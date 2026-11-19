"""
Train the PIDR reconstruction pipeline: Diffusion -> PINN refiner.

Algorithm:
1) A conditional diffusion model reconstructs the channel map from sparse CSI + mask.
2) The diffusion reconstruction is fed into a PINN refiner.

Losses (kept similar to the previous implementation):
- Diffusion: masked noise prediction MSE (DDPM-style objective).
- PINN: missing-region reconstruction L1.

This script supports caching diffusion reconstructions so PINN training does not
repeat expensive diffusion sampling every epoch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pidr.models.pinn_refiner import PINNCSIReconstructor
from pidr.config import Config
from pidr.data import CSIGridDataset
from pidr.diffusion import DiffusionManager
from pidr.models.diffusion_unet import CSIDiffusionPhysicsModel


@dataclass(frozen=True)
class PIDRPaths:
    diffusion_ckpt: str
    refiner_ckpt: str
    cache_train: str
    cache_test: str


def masked_noise_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, missing_weight: float) -> torch.Tensor:
    """Weighted MSE that emphasizes missing regions (1 - mask)."""
    missing_mask = 1.0 - mask
    weights = mask + missing_mask * float(missing_weight)
    return ((prediction - target) ** 2 * weights).mean()


def missing_region_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 computed only on missing region."""
    missing_mask = 1.0 - mask
    denom = missing_mask.sum().clamp_min(1.0)
    return (torch.abs(prediction - target) * missing_mask).sum() / denom


def missing_region_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE computed only on missing region (useful metric)."""
    missing_mask = 1.0 - mask
    denom = missing_mask.sum().clamp_min(1.0)
    return (((prediction - target) ** 2) * missing_mask).sum() / denom


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _checkpoint_signature(path: str) -> Dict[str, Any]:
    st = os.stat(path)
    return {
        "abs_path": os.path.abspath(path),
        "mtime": float(st.st_mtime),
        "size": int(st.st_size),
    }


def _resolve_existing_path(primary_path: str, legacy_path: str) -> str:
    """Prefer the new PIDR path, but fall back to the legacy cascade path if needed."""
    if os.path.exists(primary_path):
        return primary_path
    if legacy_path and os.path.exists(legacy_path):
        return legacy_path
    return primary_path


def sample_physics_guided(
    diffusion: DiffusionManager,
    model: torch.nn.Module,
    sparse_csi: torch.Tensor,
    mask: torch.Tensor,
    *,
    stride: int = 1,
) -> torch.Tensor:
    """
    Physics-guided ancestral sampling (same core logic as DiffusionManager.sample),
    but without per-call tqdm and with optional striding for speed.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    model.eval()
    device = diffusion.device
    batch_size = sparse_csi.shape[0]
    height, width = sparse_csi.shape[-2:]

    x = torch.randn((batch_size, 3, height, width), device=device)

    steps = list(range(diffusion.timesteps - 1, 0, -stride))
    if not steps or steps[-1] != 1:
        steps.append(1)

    for step in steps:
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)
        predicted_noise = model(x, t, sparse_csi, mask)

        if step % Config.PHYSICS_GUIDANCE_INTERVAL == 0:
            x0_est = diffusion.predict_x0(x, t, predicted_noise)
            x0_est = diffusion.project_to_physics(x0_est)
            x0_est = x0_est * (1 - mask) + sparse_csi * mask
            alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
            predicted_noise = (x - torch.sqrt(alpha_hat) * x0_est) / torch.sqrt(1 - alpha_hat)

        alpha = diffusion.alpha[t][:, None, None, None]
        alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
        beta = diffusion.beta[t][:, None, None, None]
        noise = torch.randn_like(x) if step > 1 else 0

        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(
            beta
        ) * noise

        x = x * (1 - mask) + sparse_csi * mask

    return x.clamp(-1.0, 1.0)


class DiffusionReconDataset(Dataset):
    """Wrap a base dataset and provide cached diffusion reconstructions as extra input."""

    def __init__(self, base: Dataset, diffusion_recon: torch.Tensor):
        self.base = base
        self.diffusion_recon = diffusion_recon

        if len(self.base) != int(self.diffusion_recon.shape[0]):
            raise ValueError(
                f"Cache size mismatch: len(base)={len(self.base)} vs diffusion_recon={tuple(self.diffusion_recon.shape)}"
            )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base[idx]
        item["diffusion_recon"] = self.diffusion_recon[idx]
        return item


def build_or_load_cache(
    *,
    cache_path: str,
    base_dataset: CSIGridDataset,
    diffusion: DiffusionManager,
    diffusion_model: torch.nn.Module,
    diffusion_ckpt_path: str,
    device: str,
    batch_size: int,
    sample_stride: int,
    force_rebuild: bool,
) -> torch.Tensor:
    """
    Cache diffusion reconstructions for a fixed-mask dataset.
    The cache is indexed in dataset order (requires loader shuffle=False).
    """
    if os.path.exists(cache_path) and not force_rebuild:
        payload = torch.load(cache_path, map_location="cpu")
        sig_expected = _checkpoint_signature(diffusion_ckpt_path)
        sig_cached = payload.get("diffusion_ckpt_signature")
        cached_recon = payload.get("diffusion_recon")
        cache_ok = (
            sig_cached == sig_expected
            and isinstance(cached_recon, torch.Tensor)
            and payload.get("sample_stride") == int(sample_stride)
            and payload.get("mask_seed") == getattr(base_dataset, "mask_seed", None)
            and payload.get("mask_strategy") == getattr(base_dataset, "mask_strategy", None)
            and tuple(cached_recon.shape)
            == (len(base_dataset), 3, Config.IMG_HEIGHT, Config.IMG_WIDTH)
        )
        if cache_ok:
            return cached_recon

    if not os.path.exists(diffusion_ckpt_path):
        raise FileNotFoundError(f"Diffusion checkpoint not found: {diffusion_ckpt_path}")

    _ensure_parent_dir(cache_path)

    diffusion_model.eval()
    for p in diffusion_model.parameters():
        p.requires_grad_(False)

    loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False)
    num_samples = len(base_dataset)
    cache = torch.empty((num_samples, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=torch.float32, device="cpu")

    offset = 0
    pbar = tqdm(loader, desc=f"Build diffusion cache -> {os.path.basename(cache_path)}")
    with torch.no_grad():
        for batch in pbar:
            sparse = batch["sparse_csi"].to(device)
            mask = batch["mask"].to(device)

            recon = sample_physics_guided(diffusion, diffusion_model, sparse, mask, stride=sample_stride)
            recon = recon.detach().to("cpu", dtype=torch.float32)

            batch_n = int(recon.shape[0])
            cache[offset : offset + batch_n].copy_(recon)
            offset += batch_n

    if offset != num_samples:
        raise RuntimeError(f"Cache fill mismatch: filled={offset} expected={num_samples}")

    torch.save(
        {
            "diffusion_recon": cache,
            "diffusion_ckpt_signature": _checkpoint_signature(diffusion_ckpt_path),
            "sample_stride": int(sample_stride),
            "mask_strategy": getattr(base_dataset, "mask_strategy", None),
            "mask_seed": getattr(base_dataset, "mask_seed", None),
        },
        cache_path,
    )
    return cache


def train_diffusion_stage(
    *,
    device: str,
    paths: PIDRPaths,
    num_epochs: int,
) -> None:
    train_dataset = CSIGridDataset(Config.DATA_PATH, mode="train", split_ratio=Config.DATA_SPLIT_RATIO)
    test_dataset = CSIGridDataset(
        Config.DATA_PATH,
        mode="test",
        split_ratio=Config.DATA_SPLIT_RATIO,
        mask_seed=Config.EVAL_MASK_SEED,
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = CSIDiffusionPhysicsModel().to(device)
    diffusion = DiffusionManager(device, train_dataset.global_min, train_dataset.global_max)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    best_val = float("inf")
    _ensure_parent_dir(paths.diffusion_ckpt)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_metric = 0.0

        pbar = tqdm(train_loader, desc=f"[PIDR-Diffusion] Epoch {epoch + 1}/{num_epochs} Train")
        for batch in pbar:
            gt = batch["gt_csi"].to(device)
            sparse = batch["sparse_csi"].to(device)
            mask = batch["mask"].to(device)

            t = torch.randint(1, Config.TIMESTEPS, (gt.shape[0],), device=device)
            x_t, noise = diffusion.noise_images(gt, t)
            pred_noise = model(x_t, t, sparse, mask)
            x0_est = diffusion.predict_x0(x_t, t, pred_noise)

            loss = masked_noise_mse(pred_noise, noise, mask, Config.MISSING_REGION_WEIGHT)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
            optimizer.step()

            train_loss += float(loss.item())
            train_metric += float(missing_region_mse(x0_est.detach(), gt, mask).item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_train_metric = train_metric / max(1, len(train_loader))

        model.eval()
        val_metric = 0.0
        with torch.no_grad():
            for batch in test_loader:
                gt = batch["gt_csi"].to(device)
                sparse = batch["sparse_csi"].to(device)
                mask = batch["mask"].to(device)

                t = torch.randint(1, Config.TIMESTEPS, (gt.shape[0],), device=device)
                x_t, _ = diffusion.noise_images(gt, t)
                pred_noise = model(x_t, t, sparse, mask)
                x0_est = diffusion.predict_x0(x_t, t, pred_noise)
                val_metric += float(missing_region_mse(x0_est, gt, mask).item())

        avg_val_metric = val_metric / max(1, len(test_loader))
        print(
            f"[PIDR-Diffusion] Epoch {epoch + 1}/{num_epochs}: "
            f"train_loss={avg_train_loss:.6f}, train_missing_mse={avg_train_metric:.6f}, val_missing_mse={avg_val_metric:.6f}"
        )

        if avg_val_metric < best_val:
            best_val = avg_val_metric
            torch.save(model.state_dict(), paths.diffusion_ckpt)
            print(f"[PIDR-Diffusion] New best saved -> {paths.diffusion_ckpt} (val_missing_mse={best_val:.6f})")


def train_pinn_stage(
    *,
    device: str,
    paths: PIDRPaths,
    num_epochs: int,
    sample_stride: int,
    force_rebuild_cache: bool,
    train_mask_seed: int,
) -> None:
    diffusion_source_ckpt = _resolve_existing_path(
        paths.diffusion_ckpt,
        Config.PIDR.LEGACY_DIFFUSION_CHECKPOINT,
    )
    if not os.path.exists(diffusion_source_ckpt):
        raise FileNotFoundError(
            f"Diffusion checkpoint is required for PIDR refiner stage, but not found: {paths.diffusion_ckpt}"
        )

    # For caching we need deterministic masks per sample.
    train_dataset = CSIGridDataset(
        Config.DATA_PATH,
        mode="train",
        split_ratio=Config.DATA_SPLIT_RATIO,
        mask_seed=train_mask_seed,
    )
    test_dataset = CSIGridDataset(
        Config.DATA_PATH,
        mode="test",
        split_ratio=Config.DATA_SPLIT_RATIO,
        mask_seed=Config.EVAL_MASK_SEED,
    )

    diffusion_model = CSIDiffusionPhysicsModel().to(device)
    diffusion_model.load_state_dict(torch.load(diffusion_source_ckpt, map_location=device))
    diffusion_model.eval()

    diffusion = DiffusionManager(device, train_dataset.global_min, train_dataset.global_max)

    diff_train = build_or_load_cache(
        cache_path=paths.cache_train,
        base_dataset=train_dataset,
        diffusion=diffusion,
        diffusion_model=diffusion_model,
        diffusion_ckpt_path=diffusion_source_ckpt,
        device=device,
        batch_size=Config.BATCH_SIZE,
        sample_stride=sample_stride,
        force_rebuild=force_rebuild_cache,
    )
    diff_test = build_or_load_cache(
        cache_path=paths.cache_test,
        base_dataset=test_dataset,
        diffusion=diffusion,
        diffusion_model=diffusion_model,
        diffusion_ckpt_path=diffusion_source_ckpt,
        device=device,
        batch_size=Config.BATCH_SIZE,
        sample_stride=sample_stride,
        force_rebuild=force_rebuild_cache,
    )

    train_ds = DiffusionReconDataset(train_dataset, diff_train)
    test_ds = DiffusionReconDataset(test_dataset, diff_test)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    pinn = PINNCSIReconstructor().to(device)
    optimizer = torch.optim.AdamW(pinn.parameters(), lr=Config.LEARNING_RATE)

    best_val = float("inf")
    _ensure_parent_dir(paths.refiner_ckpt)

    for epoch in range(num_epochs):
        pinn.train()
        train_loss = 0.0
        train_metric = 0.0

        pbar = tqdm(train_loader, desc=f"[PIDR-Refiner] Epoch {epoch + 1}/{num_epochs} Train")
        for batch in pbar:
            gt = batch["gt_csi"].to(device)
            sparse = batch["sparse_csi"].to(device)
            mask = batch["mask"].to(device)
            diff_recon = batch["diffusion_recon"].to(device)

            pred = pinn(diff_recon, mask)
            # Keep observed positions exact.
            pred = pred * (1 - mask) + sparse * mask

            loss = missing_region_l1(pred, gt, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_metric += float(missing_region_mse(pred.detach(), gt, mask).item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_train_metric = train_metric / max(1, len(train_loader))

        pinn.eval()
        val_metric = 0.0
        with torch.no_grad():
            for batch in test_loader:
                gt = batch["gt_csi"].to(device)
                sparse = batch["sparse_csi"].to(device)
                mask = batch["mask"].to(device)
                diff_recon = batch["diffusion_recon"].to(device)

                pred = pinn(diff_recon, mask)
                pred = pred * (1 - mask) + sparse * mask
                val_metric += float(missing_region_mse(pred, gt, mask).item())

        avg_val_metric = val_metric / max(1, len(test_loader))
        print(
            f"[PIDR-Refiner] Epoch {epoch + 1}/{num_epochs}: "
            f"train_loss={avg_train_loss:.6f}, train_missing_mse={avg_train_metric:.6f}, val_missing_mse={avg_val_metric:.6f}"
        )

        if avg_val_metric < best_val:
            best_val = avg_val_metric
            torch.save(pinn.state_dict(), paths.refiner_ckpt)
            print(f"[PIDR-Refiner] New best saved -> {paths.refiner_ckpt} (val_missing_mse={best_val:.6f})")


def main() -> None:
    paths = PIDRPaths(
        diffusion_ckpt=Config.PIDR.DIFFUSION_CHECKPOINT,
        refiner_ckpt=Config.PIDR.REFINER_CHECKPOINT,
        cache_train=Config.PIDR.CACHE_TRAIN,
        cache_test=Config.PIDR.CACHE_TEST,
    )

    device = Config.DEVICE
    os.makedirs(Config.PATHS.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.PATHS.CACHE_DIR, exist_ok=True)

    if Config.PIDR.TRAIN_DIFFUSION:
        train_diffusion_stage(device=device, paths=paths, num_epochs=Config.PIDR.DIFFUSION_EPOCHS)
    else:
        diffusion_source_ckpt = _resolve_existing_path(
            paths.diffusion_ckpt,
            Config.PIDR.LEGACY_DIFFUSION_CHECKPOINT,
        )
        if not os.path.exists(diffusion_source_ckpt):
            raise FileNotFoundError(
                f"train_diffusion=False but diffusion checkpoint missing: {paths.diffusion_ckpt}"
            )

    if Config.PIDR.TRAIN_PINN:
        train_pinn_stage(
            device=device,
            paths=paths,
            num_epochs=Config.PIDR.PINN_EPOCHS,
            sample_stride=Config.PIDR.SAMPLE_STRIDE,
            force_rebuild_cache=Config.PIDR.FORCE_REBUILD_CACHE,
            train_mask_seed=Config.PIDR.PINN_TRAIN_MASK_SEED,
        )
    else:
        print("train_pinn=False, skip PIDR refiner stage.")


if __name__ == "__main__":
    main()
