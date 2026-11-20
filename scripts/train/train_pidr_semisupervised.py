"""Train PIDR with partial supervision.

This script simulates an incomplete training set by splitting the observed points
of each full CSI map into two disjoint subsets:
1. input points: provided to the model as conditional observations
2. supervision points: used only for the loss

The union of these two subsets is still incomplete, so no loss is applied on the
remaining unseen region.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pidr.models.field_response import (
    apply_field_response,
    calibrate_field_response,
    configure_field_response,
    estimate_global_psd,
)
from pidr.models.pinn_refiner import PINNCSIReconstructor
from scripts.train.train_pidr import TensorBatches
from pidr.config import Config
from pidr.data import CSIGridDataset
from pidr.diffusion import DiffusionManager
from pidr.models.diffusion_unet import CSIDiffusionPhysicsModel


@dataclass(frozen=True)
class PartialPIDRPaths:
    diffusion_ckpt: str
    refiner_ckpt: str
    cache_train: str
    cache_test: str


def masked_region_mse(prediction: torch.Tensor, target: torch.Tensor, region_mask: torch.Tensor) -> torch.Tensor:
    """Average MSE over a sparse supervision region only."""
    denom = region_mask.sum().clamp_min(1.0)
    return (((prediction - target) ** 2) * region_mask).sum() / denom


def masked_region_l1(prediction: torch.Tensor, target: torch.Tensor, region_mask: torch.Tensor) -> torch.Tensor:
    """Average L1 over a sparse supervision region only."""
    denom = region_mask.sum().clamp_min(1.0)
    return (torch.abs(prediction - target) * region_mask).sum() / denom


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


def validate_partial_config() -> None:
    total_available_ratio = float(Config.PIDR_PARTIAL.TOTAL_AVAILABLE_RATIO)
    input_fraction = float(Config.PIDR_PARTIAL.INPUT_FRACTION_WITHIN_AVAILABLE)

    if not (0.0 < total_available_ratio <= 1.0):
        raise ValueError(f"TOTAL_AVAILABLE_RATIO must be in (0, 1], got {total_available_ratio}")
    if not (0.0 < input_fraction < 1.0):
        raise ValueError(f"INPUT_FRACTION_WITHIN_AVAILABLE must be in (0, 1), got {input_fraction}")


class PartialSupervisionCSIGridDataset(Dataset):
    """Wrap a fully labeled dataset and expose only partial supervision."""

    def __init__(
        self,
        base_dataset: CSIGridDataset,
        *,
        total_available_ratio: float,
        input_fraction_within_available: float,
        split_seed: int | None,
        mask_strategy: str,
    ) -> None:
        self.base_dataset = base_dataset
        self.total_available_ratio = float(total_available_ratio)
        self.input_fraction_within_available = float(input_fraction_within_available)
        self.split_seed = split_seed
        self.mask_strategy = mask_strategy

        self.global_min = base_dataset.global_min
        self.global_max = base_dataset.global_max
        self.mode = base_dataset.mode
        self.mask_seed = split_seed

        if not (0.0 < self.total_available_ratio <= 1.0):
            raise ValueError(f"total_available_ratio must be in (0, 1], got {self.total_available_ratio}")
        if not (0.0 < self.input_fraction_within_available < 1.0):
            raise ValueError(
                f"input_fraction_within_available must be in (0, 1), got {self.input_fraction_within_available}"
            )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = int(self.base_dataset.indices[idx])
        gt_csi = self.base_dataset.gt_data[sample_idx].copy()
        valid_mask = self.base_dataset.valid_masks[sample_idx, 0].copy()

        input_mask, supervision_mask, available_mask = self._split_masks(valid_mask, sample_idx)
        input_csi = gt_csi * input_mask[None, :, :]
        available_csi = gt_csi * available_mask[None, :, :]

        return {
            "input_csi": torch.from_numpy(input_csi).float(),
            "available_csi": torch.from_numpy(available_csi).float(),
            "input_mask": torch.from_numpy(input_mask[None, :, :]).float(),
            "supervision_mask": torch.from_numpy(supervision_mask[None, :, :]).float(),
            "available_mask": torch.from_numpy(available_mask[None, :, :]).float(),
        }

    def get_normalization_tensors(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self.base_dataset.get_normalization_tensors(device)

    def _split_masks(self, valid_mask: np.ndarray, sample_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        available_mask = self.base_dataset.generate_mask(
            valid_mask,
            strategy=self.mask_strategy,
            sample_idx=sample_idx,
            observed_ratio=self.total_available_ratio,
        )
        available_positions = np.argwhere(available_mask > 0.5)
        if len(available_positions) < 2:
            raise ValueError(
                "Partial supervision requires at least two available points so input and supervision can be disjoint."
            )

        input_count = int(round(len(available_positions) * self.input_fraction_within_available))
        input_count = min(max(1, input_count), len(available_positions) - 1)

        rng = self._make_rng(sample_idx)
        input_indices = rng.choice(len(available_positions), size=input_count, replace=False)
        selected = np.zeros(len(available_positions), dtype=bool)
        selected[input_indices] = True

        input_positions = available_positions[selected]
        supervision_positions = available_positions[~selected]

        input_mask = np.zeros_like(valid_mask, dtype=np.float32)
        supervision_mask = np.zeros_like(valid_mask, dtype=np.float32)

        input_mask[input_positions[:, 0], input_positions[:, 1]] = 1.0
        supervision_mask[supervision_positions[:, 0], supervision_positions[:, 1]] = 1.0
        return input_mask, supervision_mask, available_mask.astype(np.float32)

    def _make_rng(self, sample_idx: int) -> np.random.Generator:
        if self.split_seed is None:
            return np.random.default_rng()
        return np.random.default_rng(int(self.split_seed) + int(sample_idx) + 100000)


def sample_physics_guided(
    diffusion: DiffusionManager,
    model: torch.nn.Module,
    input_csi: torch.Tensor,
    input_mask: torch.Tensor,
    *,
    stride: int = 1,
) -> torch.Tensor:
    """Physics-guided sampling without tqdm spam inside cache generation."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    model.eval()
    device = diffusion.device
    batch_size = input_csi.shape[0]
    height, width = input_csi.shape[-2:]

    x = torch.randn((batch_size, 3, height, width), device=device)
    steps = list(range(diffusion.timesteps - 1, 0, -stride))
    if not steps or steps[-1] != 1:
        steps.append(1)

    for step in steps:
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)
        predicted_noise = model(x, t, input_csi, input_mask)

        if step % Config.PHYSICS_GUIDANCE_INTERVAL == 0:
            x0_est = diffusion.predict_x0(x, t, predicted_noise)
            x0_est = diffusion.project_to_physics(x0_est)
            x0_est = x0_est * (1 - input_mask) + input_csi * input_mask
            alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
            predicted_noise = (x - torch.sqrt(alpha_hat) * x0_est) / torch.sqrt(1 - alpha_hat)

        alpha = diffusion.alpha[t][:, None, None, None]
        alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
        beta = diffusion.beta[t][:, None, None, None]
        noise = torch.randn_like(x) if step > 1 else 0
        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(
            beta
        ) * noise
        x = x * (1 - input_mask) + input_csi * input_mask

    return x.clamp(-1.0, 1.0)


class DiffusionReconDataset(Dataset):
    """Attach cached diffusion reconstructions to a partial-supervision dataset."""

    def __init__(self, base: Dataset, diffusion_recon: torch.Tensor, physics: Optional[torch.Tensor] = None):
        self.base = base
        self.diffusion_recon = diffusion_recon
        self.physics = physics

        if len(self.base) != int(self.diffusion_recon.shape[0]):
            raise ValueError(
                f"Cache size mismatch: len(base)={len(self.base)} vs diffusion_recon={tuple(self.diffusion_recon.shape)}"
            )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base[idx]
        item["diffusion_recon"] = self.diffusion_recon[idx]
        if self.physics is not None:
            item["physics"] = self.physics[idx]
        return item


def build_or_load_cache(
    *,
    cache_path: str,
    base_dataset: PartialSupervisionCSIGridDataset,
    diffusion: DiffusionManager,
    diffusion_model: torch.nn.Module,
    diffusion_ckpt_path: str,
    device: str,
    batch_size: int,
    sample_stride: int,
    force_rebuild: bool,
) -> torch.Tensor:
    """Cache diffusion reconstructions for a fixed partial-supervision split."""
    if os.path.exists(cache_path) and not force_rebuild:
        payload = torch.load(cache_path, map_location="cpu")
        sig_expected = _checkpoint_signature(diffusion_ckpt_path)
        sig_cached = payload.get("diffusion_ckpt_signature")
        cached_recon = payload.get("diffusion_recon")
        cache_ok = (
            sig_cached == sig_expected
            and isinstance(cached_recon, torch.Tensor)
            and payload.get("sample_stride") == int(sample_stride)
            and payload.get("total_available_ratio") == float(base_dataset.total_available_ratio)
            and payload.get("input_fraction_within_available") == float(base_dataset.input_fraction_within_available)
            and payload.get("split_seed") == base_dataset.split_seed
            and payload.get("mask_strategy") == base_dataset.mask_strategy
            and tuple(cached_recon.shape) == (len(base_dataset), 3, Config.IMG_HEIGHT, Config.IMG_WIDTH)
        )
        if cache_ok:
            return cached_recon

    if not os.path.exists(diffusion_ckpt_path):
        raise FileNotFoundError(f"Diffusion checkpoint not found: {diffusion_ckpt_path}")

    _ensure_parent_dir(cache_path)

    diffusion_model.eval()
    for param in diffusion_model.parameters():
        param.requires_grad_(False)

    loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False)
    num_samples = len(base_dataset)
    cache = torch.empty((num_samples, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=torch.float32, device="cpu")

    offset = 0
    pbar = tqdm(loader, desc=f"Build partial diffusion cache -> {os.path.basename(cache_path)}")
    with torch.no_grad():
        for batch in pbar:
            input_csi = batch["input_csi"].to(device)
            input_mask = batch["input_mask"].to(device)

            recon = sample_physics_guided(
                diffusion,
                diffusion_model,
                input_csi,
                input_mask,
                stride=sample_stride,
            ).detach().to("cpu", dtype=torch.float32)

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
            "total_available_ratio": float(base_dataset.total_available_ratio),
            "input_fraction_within_available": float(base_dataset.input_fraction_within_available),
            "split_seed": base_dataset.split_seed,
            "mask_strategy": base_dataset.mask_strategy,
        },
        cache_path,
    )
    return cache


def create_partial_dataset(*, mode: str, split_seed: int) -> PartialSupervisionCSIGridDataset:
    base_dataset = CSIGridDataset(
        Config.DATA_PATH,
        mode=mode,
        split_ratio=Config.DATA_SPLIT_RATIO,
        mask_seed=split_seed,
    )
    return PartialSupervisionCSIGridDataset(
        base_dataset,
        total_available_ratio=Config.PIDR_PARTIAL.TOTAL_AVAILABLE_RATIO,
        input_fraction_within_available=Config.PIDR_PARTIAL.INPUT_FRACTION_WITHIN_AVAILABLE,
        split_seed=split_seed,
        mask_strategy=Config.PIDR_PARTIAL.MASK_STRATEGY,
    )


def train_diffusion_stage(
    *,
    device: str,
    paths: PartialPIDRPaths,
    num_epochs: int,
) -> None:
    train_dataset = create_partial_dataset(mode="train", split_seed=Config.PIDR_PARTIAL.TRAIN_SPLIT_SEED)
    test_dataset = create_partial_dataset(mode="test", split_seed=Config.PIDR_PARTIAL.EVAL_SPLIT_SEED)

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

        pbar = tqdm(train_loader, desc=f"[Partial-PIDR-Diffusion] Epoch {epoch + 1}/{num_epochs} Train")
        for batch in pbar:
            target = batch["available_csi"].to(device)
            input_csi = batch["input_csi"].to(device)
            input_mask = batch["input_mask"].to(device)
            supervision_mask = batch["supervision_mask"].to(device)

            t = torch.randint(1, Config.TIMESTEPS, (target.shape[0],), device=device)
            x_t, noise = diffusion.noise_images(target, t)
            pred_noise = model(x_t, t, input_csi, input_mask)
            x0_est = diffusion.predict_x0(x_t, t, pred_noise)
            x0_est = x0_est * (1 - input_mask) + input_csi * input_mask

            loss = masked_region_mse(pred_noise, noise, supervision_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
            optimizer.step()

            train_loss += float(loss.item())
            train_metric += float(masked_region_mse(x0_est.detach(), target, supervision_mask).item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_train_metric = train_metric / max(1, len(train_loader))

        model.eval()
        val_metric = 0.0
        with torch.no_grad():
            for batch in test_loader:
                target = batch["available_csi"].to(device)
                input_csi = batch["input_csi"].to(device)
                input_mask = batch["input_mask"].to(device)
                supervision_mask = batch["supervision_mask"].to(device)

                t = torch.randint(1, Config.TIMESTEPS, (target.shape[0],), device=device)
                x_t, _ = diffusion.noise_images(target, t)
                pred_noise = model(x_t, t, input_csi, input_mask)
                x0_est = diffusion.predict_x0(x_t, t, pred_noise)
                x0_est = x0_est * (1 - input_mask) + input_csi * input_mask
                val_metric += float(masked_region_mse(x0_est, target, supervision_mask).item())

        avg_val_metric = val_metric / max(1, len(test_loader))
        print(
            f"[Partial-PIDR-Diffusion] Epoch {epoch + 1}/{num_epochs}: "
            f"train_loss={avg_train_loss:.6f}, train_supervision_mse={avg_train_metric:.6f}, "
            f"val_supervision_mse={avg_val_metric:.6f}"
        )

        if avg_val_metric < best_val:
            best_val = avg_val_metric
            torch.save(model.state_dict(), paths.diffusion_ckpt)
            print(
                f"[Partial-PIDR-Diffusion] New best saved -> {paths.diffusion_ckpt} "
                f"(val_supervision_mse={best_val:.6f})"
            )


def _stack(dataset: Dataset, key: str) -> torch.Tensor:
    return torch.stack([dataset[i][key] for i in range(len(dataset))])


def prepare_field_response_partial(refiner, train_dataset, test_dataset, diff_train, diff_test):
    """Calibrate the field-response layer without complete maps and precompute its outputs.

    Only what the partial-supervision setting provides is used: the dataset wavenumber spectrum is
    estimated from the diffusion reconstructions of the training maps (which depend only on the input
    points), and the LMMSE hyper-parameters are chosen by the gain error on the held-out supervision
    points of the training maps.
    """
    layer = refiner.prior
    lo = torch.tensor(train_dataset.global_min, dtype=torch.float64).view(1, 3, 1, 1)
    hi = torch.tensor(train_dataset.global_max, dtype=torch.float64).view(1, 3, 1, 1)
    phys = (diff_train.double() + 1) / 2 * (hi - lo) + lo
    psd = estimate_global_psd(torch.complex(phys[:, 1], phys[:, 2]), Config.IMG_HEIGHT, Config.IMG_WIDTH)
    configure_field_response(layer, train_dataset.global_min, train_dataset.global_max, global_psd=psd)
    input_mask_train = _stack(train_dataset, "input_mask")
    calib = calibrate_field_response(
        layer, diff_train, input_mask_train, _stack(train_dataset, "available_csi"),
        eval_mask=_stack(train_dataset, "supervision_mask"),
    )
    print(f"[Partial-PIDR-Refiner] field-response calibration (train supervision points): {calib}", flush=True)
    phys_train = apply_field_response(layer, diff_train, input_mask_train)
    phys_test = apply_field_response(layer, diff_test, _stack(test_dataset, "input_mask"))
    return phys_train, phys_test


def train_refiner_stage(
    *,
    device: str,
    paths: PartialPIDRPaths,
    num_epochs: int,
    sample_stride: int,
    force_rebuild_cache: bool,
    refiner_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Refiner stage; ``refiner_kwargs`` selects the variant (default ``Config.PIDR.REFINER_KWARGS``)."""
    if refiner_kwargs is None:
        refiner_kwargs = dict(Config.PIDR.REFINER_KWARGS)
    if not os.path.exists(paths.diffusion_ckpt):
        raise FileNotFoundError(f"Diffusion checkpoint is required for the refiner stage: {paths.diffusion_ckpt}")

    train_dataset = create_partial_dataset(mode="train", split_seed=Config.PIDR_PARTIAL.TRAIN_SPLIT_SEED)
    test_dataset = create_partial_dataset(mode="test", split_seed=Config.PIDR_PARTIAL.EVAL_SPLIT_SEED)

    diffusion_model = CSIDiffusionPhysicsModel().to(device)
    diffusion_model.load_state_dict(torch.load(paths.diffusion_ckpt, map_location=device))
    diffusion_model.eval()

    diffusion = DiffusionManager(device, train_dataset.global_min, train_dataset.global_max)

    diff_train = build_or_load_cache(
        cache_path=paths.cache_train,
        base_dataset=train_dataset,
        diffusion=diffusion,
        diffusion_model=diffusion_model,
        diffusion_ckpt_path=paths.diffusion_ckpt,
        device=device,
        batch_size=Config.PIDR.CACHE_BATCH_SIZE,
        sample_stride=sample_stride,
        force_rebuild=force_rebuild_cache,
    )
    diff_test = build_or_load_cache(
        cache_path=paths.cache_test,
        base_dataset=test_dataset,
        diffusion=diffusion,
        diffusion_model=diffusion_model,
        diffusion_ckpt_path=paths.diffusion_ckpt,
        device=device,
        batch_size=Config.PIDR.CACHE_BATCH_SIZE,
        sample_stride=sample_stride,
        force_rebuild=force_rebuild_cache,
    )

    refiner = PINNCSIReconstructor(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH, **refiner_kwargs).to(device)
    print(f"[Partial-PIDR-Refiner] variant: {refiner_kwargs}", flush=True)
    phys_train = phys_test = None
    if refiner.prior_type == "field_response":
        phys_train, phys_test = prepare_field_response_partial(refiner, train_dataset, test_dataset, diff_train, diff_test)

    train_ds = DiffusionReconDataset(train_dataset, diff_train, phys_train)
    test_ds = DiffusionReconDataset(test_dataset, diff_test, phys_test)

    # Splits are fixed per sample, so the whole split is materialised once on the device.
    train_loader = TensorBatches(train_ds, Config.BATCH_SIZE, shuffle=True, device=device)
    test_loader = TensorBatches(test_ds, Config.BATCH_SIZE, shuffle=False, device=device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=Config.LEARNING_RATE)

    best_val = float("inf")
    _ensure_parent_dir(paths.refiner_ckpt)

    for epoch in range(num_epochs):
        refiner.train()
        train_loss = 0.0
        train_metric = 0.0

        pbar = tqdm(train_loader, desc=f"[Partial-PIDR-Refiner] Epoch {epoch + 1}/{num_epochs} Train")
        for batch in pbar:
            target = batch["available_csi"].to(device)
            input_csi = batch["input_csi"].to(device)
            input_mask = batch["input_mask"].to(device)
            supervision_mask = batch["supervision_mask"].to(device)
            diffusion_recon = batch["diffusion_recon"].to(device)

            pred = refiner(diffusion_recon, input_mask, physics=batch.get("physics"))
            pred = pred * (1 - input_mask) + input_csi * input_mask

            loss = masked_region_l1(pred, target, supervision_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_metric += float(masked_region_mse(pred.detach(), target, supervision_mask).item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_train_metric = train_metric / max(1, len(train_loader))

        refiner.eval()
        val_metric = 0.0
        with torch.no_grad():
            for batch in test_loader:
                target = batch["available_csi"].to(device)
                input_csi = batch["input_csi"].to(device)
                input_mask = batch["input_mask"].to(device)
                supervision_mask = batch["supervision_mask"].to(device)
                diffusion_recon = batch["diffusion_recon"].to(device)

                pred = refiner(diffusion_recon, input_mask, physics=batch.get("physics"))
                pred = pred * (1 - input_mask) + input_csi * input_mask
                val_metric += float(masked_region_mse(pred, target, supervision_mask).item())

        avg_val_metric = val_metric / max(1, len(test_loader))
        print(
            f"[Partial-PIDR-Refiner] Epoch {epoch + 1}/{num_epochs}: "
            f"train_loss={avg_train_loss:.6f}, train_supervision_mse={avg_train_metric:.6f}, "
            f"val_supervision_mse={avg_val_metric:.6f}"
        )

        if avg_val_metric < best_val:
            best_val = avg_val_metric
            torch.save(refiner.state_dict(), paths.refiner_ckpt)
            print(
                f"[Partial-PIDR-Refiner] New best saved -> {paths.refiner_ckpt} "
                f"(val_supervision_mse={best_val:.6f})"
            )


def main() -> None:
    validate_partial_config()

    paths = PartialPIDRPaths(
        diffusion_ckpt=Config.PIDR_PARTIAL.DIFFUSION_CHECKPOINT,
        refiner_ckpt=Config.PIDR_PARTIAL.REFINER_CHECKPOINT,
        cache_train=Config.PIDR_PARTIAL.CACHE_TRAIN,
        cache_test=Config.PIDR_PARTIAL.CACHE_TEST,
    )

    device = Config.DEVICE
    os.makedirs(Config.PATHS.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.PATHS.CACHE_DIR, exist_ok=True)

    print(
        "Partial-supervision PIDR training config: "
        f"total_available_ratio={Config.PIDR_PARTIAL.TOTAL_AVAILABLE_RATIO:.3f}, "
        f"input_fraction_within_available={Config.PIDR_PARTIAL.INPUT_FRACTION_WITHIN_AVAILABLE:.3f}, "
        f"mask_strategy={Config.PIDR_PARTIAL.MASK_STRATEGY}"
    )

    if Config.PIDR_PARTIAL.TRAIN_DIFFUSION:
        train_diffusion_stage(
            device=device,
            paths=paths,
            num_epochs=Config.PIDR_PARTIAL.DIFFUSION_EPOCHS,
        )
    else:
        if not os.path.exists(paths.diffusion_ckpt):
            raise FileNotFoundError(
                f"TRAIN_DIFFUSION=False but diffusion checkpoint is missing: {paths.diffusion_ckpt}"
            )

    if Config.PIDR_PARTIAL.TRAIN_REFINER:
        train_refiner_stage(
            device=device,
            paths=paths,
            num_epochs=Config.PIDR_PARTIAL.REFINER_EPOCHS,
            sample_stride=Config.PIDR_PARTIAL.SAMPLE_STRIDE,
            force_rebuild_cache=Config.PIDR_PARTIAL.FORCE_REBUILD_CACHE,
        )
    else:
        print("TRAIN_REFINER=False, skip partial PIDR refiner stage.")


if __name__ == "__main__":
    main()
