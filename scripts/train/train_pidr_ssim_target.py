"""Train the full EL PIDR cascade until Gain SSIM target is reached.

This trains the refiner on diffusion reconstructions, not on sparse CSI
directly. The diffusion checkpoint is treated as the first PIDR stage and the
new refiner checkpoint is saved as a plain state_dict in a unique run folder.
No existing checkpoints are overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.interpolate import griddata
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pidr.models.pinn_refiner import PINNCSIReconstructor
from pidr.config import Config
from pidr.data import CSIGridDataset
from pidr.diffusion import DiffusionManager
from pidr.models.diffusion_unet import CSIDiffusionPhysicsModel
from scripts.train.train_pidr import sample_physics_guided


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(root: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"pidr_full_el_{stamp}"
    counter = 1
    while run_dir.exists():
        run_dir = Path(root) / f"pidr_full_el_{stamp}_{counter}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def configure_data_shape(data_path: str, img_height: int | None, img_width: int | None, train_sample_count: int | None) -> None:
    if img_height is None or img_width is None:
        mat = loadmat(data_path)
        gain = np.asarray(mat["CSI_Gain"])
        gain = np.squeeze(gain)
        if gain.ndim == 2:
            inferred_h, inferred_w = gain.shape
        elif gain.ndim == 3:
            dims = list(gain.shape)
            if img_height is not None and img_height in dims:
                inferred_h = img_height
                remaining = [dim for dim in dims if dim != inferred_h]
                inferred_w = img_width if img_width is not None else min(remaining)
            elif img_width is not None and img_width in dims:
                inferred_w = img_width
                remaining = [dim for dim in dims if dim != inferred_w]
                inferred_h = min(remaining)
            else:
                spatial = sorted(dims)[:2]
                inferred_h, inferred_w = spatial[0], spatial[1]
        else:
            raise ValueError(f"Cannot infer image shape from CSI_Gain shape {gain.shape}")
        img_height = inferred_h if img_height is None else img_height
        img_width = inferred_w if img_width is None else img_width

    Config.IMG_HEIGHT = int(img_height)
    Config.IMG_WIDTH = int(img_width)
    Config.DATA.IMG_HEIGHT = int(img_height)
    Config.DATA.IMG_WIDTH = int(img_width)
    Config.TRAIN_SAMPLE_COUNT = train_sample_count
    Config.DATA.TRAIN_SAMPLE_COUNT = train_sample_count


def configure_mask_ratio(mask_ratio: float | None) -> None:
    if mask_ratio is None:
        return
    if not (0.0 <= mask_ratio < 1.0):
        raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")
    Config.MASK_RATIO = float(mask_ratio)
    Config.DATA.MASK_RATIO = float(mask_ratio)


def load_plain_or_wrapped_state_dict(path: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def checkpoint_signature(path: str) -> Dict[str, object]:
    st = os.stat(path)
    return {
        "abs_path": os.path.abspath(path),
        "mtime": float(st.st_mtime),
        "size": int(st.st_size),
    }


def infer_diffusion_base_channels(path: str) -> int:
    state_dict = load_plain_or_wrapped_state_dict(path)
    weight = state_dict.get("input_proj.weight")
    if weight is None or not hasattr(weight, "shape"):
        return int(Config.BASE_CHANNELS)
    return int(weight.shape[0])


def configure_diffusion_base_channels(diffusion_base_channels: int | None) -> int:
    if diffusion_base_channels is None:
        return int(Config.BASE_CHANNELS)
    if diffusion_base_channels <= 0:
        raise ValueError(f"diffusion_base_channels must be > 0, got {diffusion_base_channels}")
    Config.BASE_CHANNELS = int(diffusion_base_channels)
    Config.MODEL.BASE_CHANNELS = int(diffusion_base_channels)
    return int(diffusion_base_channels)


def freeze_batch_norm_modules(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)


def interpolate_sparse_batch(sparse: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill missing entries from observed sparse points using linear interpolation."""
    sparse_np = sparse.detach().cpu().numpy()
    mask_np = mask[:, 0].detach().cpu().numpy() > 0.5
    batch, channels, height, width = sparse_np.shape
    yy, xx = np.mgrid[0:height, 0:width]
    grid_points = np.column_stack([yy.ravel(), xx.ravel()])
    filled = np.empty_like(sparse_np, dtype=np.float32)

    for idx in range(batch):
        observed = np.column_stack(np.where(mask_np[idx]))
        if observed.size == 0:
            filled[idx] = sparse_np[idx]
            continue
        for channel in range(channels):
            observed_values = sparse_np[idx, channel][mask_np[idx]]
            linear = griddata(observed, observed_values, grid_points, method="linear", fill_value=np.nan).reshape(height, width)
            if np.isnan(linear).any():
                nearest = griddata(observed, observed_values, grid_points, method="nearest").reshape(height, width)
                linear = np.where(np.isnan(linear), nearest, linear)
            filled[idx, channel] = linear
    return torch.from_numpy(np.clip(filled, -1.0, 1.0))


@torch.no_grad()
def sample_ddim_physics_guided(
    diffusion: DiffusionManager,
    model: torch.nn.Module,
    sparse_csi: torch.Tensor,
    mask: torch.Tensor,
    *,
    stride: int = 20,
    return_final_x0: bool = True,
) -> torch.Tensor:
    """Deterministic DDIM-style physics-guided sampling for stronger PIDR inputs."""
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

    last_x0 = None
    for step in steps:
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)
        predicted_noise = model(x, t, sparse_csi, mask)
        x0_est = diffusion.predict_x0(x, t, predicted_noise)

        if step % Config.PHYSICS_GUIDANCE_INTERVAL == 0:
            x0_est = diffusion.project_to_physics(x0_est)
            x0_est = x0_est * (1 - mask) + sparse_csi * mask
            alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
            predicted_noise = (x - torch.sqrt(alpha_hat) * x0_est) / torch.sqrt(1 - alpha_hat)

        prev_step = max(step - stride, 0)
        prev_t = torch.full((batch_size,), prev_step, device=device, dtype=torch.long)
        alpha_hat_prev = diffusion.alpha_hat[prev_t][:, None, None, None]
        x = torch.sqrt(alpha_hat_prev) * x0_est + torch.sqrt(1 - alpha_hat_prev) * predicted_noise
        x = x * (1 - mask) + sparse_csi * mask
        last_x0 = x0_est

    if return_final_x0 and last_x0 is not None:
        x = last_x0 * (1 - mask) + sparse_csi * mask
    return x.clamp(-1.0, 1.0)


def build_or_load_diffusion_cache(
    *,
    cache_path: Path,
    dataset: CSIGridDataset,
    diffusion_ckpt: str,
    device: str,
    batch_size: int,
    sample_stride: int,
    force_rebuild: bool,
    seed: int,
    postprocess: str,
    interp_weight: float,
    diffusion_base_channels: int,
    sampling_method: str,
) -> torch.Tensor:
    expected_shape = (len(dataset), 3, Config.IMG_HEIGHT, Config.IMG_WIDTH)
    expected_signature = checkpoint_signature(diffusion_ckpt) if postprocess != "linear_interp_only" else {"linear_interp_only": True}
    expected_signature["diffusion_base_channels"] = int(diffusion_base_channels)
    if cache_path.exists() and not force_rebuild:
        payload = torch.load(cache_path, map_location="cpu")
        cached = payload.get("diffusion_recon")
        if (
            isinstance(cached, torch.Tensor)
            and tuple(cached.shape) == expected_shape
            and payload.get("diffusion_ckpt_signature") == expected_signature
            and payload.get("sample_stride") == int(sample_stride)
            and payload.get("mask_strategy") == dataset.mask_strategy
            and payload.get("mask_seed") == dataset.mask_seed
            and payload.get("seed") == seed
            and payload.get("postprocess") == postprocess
            and float(payload.get("interp_weight", -1.0)) == float(interp_weight)
            and payload.get("sampling_method", "ddpm") == sampling_method
        ):
            return cached

    set_seed(seed)
    diffusion_model = None
    diffusion = None
    if postprocess != "linear_interp_only":
        diffusion_model = CSIDiffusionPhysicsModel().to(device).eval()
        diffusion_model.load_state_dict(load_plain_or_wrapped_state_dict(diffusion_ckpt))
        for param in diffusion_model.parameters():
            param.requires_grad_(False)
        diffusion = DiffusionManager(device, dataset.global_min, dataset.global_max)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    cache = torch.empty(expected_shape, dtype=torch.float32, device="cpu")

    offset = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Build cache {cache_path.name}"):
            sparse = batch["sparse_csi"].to(device)
            mask = batch["mask"].to(device)
            if postprocess == "linear_interp_only":
                recon = interpolate_sparse_batch(sparse, mask).to(device)
            else:
                if sampling_method == "ddpm":
                    recon = sample_physics_guided(diffusion, diffusion_model, sparse, mask, stride=sample_stride)
                elif sampling_method == "ddim":
                    recon = sample_ddim_physics_guided(
                        diffusion, diffusion_model, sparse, mask, stride=sample_stride, return_final_x0=False
                    )
                elif sampling_method == "ddim_x0":
                    recon = sample_ddim_physics_guided(
                        diffusion, diffusion_model, sparse, mask, stride=sample_stride, return_final_x0=True
                    )
                else:
                    raise ValueError(f"Unsupported sampling_method: {sampling_method}")
            if postprocess == "linear_interp_blend":
                interp = interpolate_sparse_batch(sparse, mask).to(device)
                recon = (1.0 - interp_weight) * recon + interp_weight * interp
                recon = recon * (1.0 - mask) + sparse * mask
            recon = recon.detach().to("cpu", dtype=torch.float32)
            batch_n = int(recon.shape[0])
            cache[offset : offset + batch_n].copy_(recon)
            offset += batch_n

    if offset != len(dataset):
        raise RuntimeError(f"Cache fill mismatch: {offset} vs {len(dataset)}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "diffusion_recon": cache,
            "diffusion_ckpt_signature": expected_signature,
            "sample_stride": int(sample_stride),
            "mask_strategy": dataset.mask_strategy,
            "mask_seed": dataset.mask_seed,
            "seed": seed,
            "postprocess": postprocess,
            "interp_weight": float(interp_weight),
            "sampling_method": sampling_method,
        },
        cache_path,
    )
    return cache


class PIDRCacheDataset(Dataset):
    def __init__(self, base: CSIGridDataset, diffusion_recon: torch.Tensor):
        self.base = base
        self.diffusion_recon = diffusion_recon
        if len(base) != int(diffusion_recon.shape[0]):
            raise ValueError("Base dataset and diffusion cache size mismatch.")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base[idx]
        item["diffusion_recon"] = self.diffusion_recon[idx]
        return item


def weighted_mean(values: torch.Tensor, channel_weights: torch.Tensor, region: torch.Tensor | None = None) -> torch.Tensor:
    if region is None:
        region = torch.ones_like(values[:, :1])
    weighted = values * channel_weights * region
    denom = (channel_weights * region).sum().clamp_min(1.0)
    return weighted.sum() / denom


def gradient_l1(pred: torch.Tensor, target: torch.Tensor, channel: int = 0) -> torch.Tensor:
    pred_c = pred[:, channel : channel + 1]
    target_c = target[:, channel : channel + 1]
    return F.l1_loss(pred_c[:, :, :, 1:] - pred_c[:, :, :, :-1], target_c[:, :, :, 1:] - target_c[:, :, :, :-1]) + F.l1_loss(
        pred_c[:, :, 1:, :] - pred_c[:, :, :-1, :], target_c[:, :, 1:, :] - target_c[:, :, :-1, :]
    )


def differentiable_ssim_loss(pred: torch.Tensor, target: torch.Tensor, channel: int = 0, window_size: int = 3) -> torch.Tensor:
    """Metric-aligned SSIM loss for the gain channel.

    Evaluation uses skimage SSIM with win_size=3 and a per-sample data_range.
    Computing the same data_range on the normalized target is equivalent to
    doing it after linear inverse normalization, so this stays differentiable
    and avoids CPU round trips during training.
    """
    pred_c = pred[:, channel : channel + 1]
    target_c = target[:, channel : channel + 1]
    padding = window_size // 2
    pred_pad = F.pad(pred_c, (padding, padding, padding, padding), mode="reflect")
    target_pad = F.pad(target_c, (padding, padding, padding, padding), mode="reflect")
    mu_x = F.avg_pool2d(pred_pad, window_size, stride=1)
    mu_y = F.avg_pool2d(target_pad, window_size, stride=1)
    sigma_x = F.avg_pool2d(pred_pad * pred_pad, window_size, stride=1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target_pad * target_pad, window_size, stride=1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred_pad * target_pad, window_size, stride=1) - mu_x * mu_y
    cov_norm = (window_size * window_size) / max(window_size * window_size - 1, 1)
    sigma_x = sigma_x * cov_norm
    sigma_y = sigma_y * cov_norm
    sigma_xy = sigma_xy * cov_norm
    data_range = (target_c.amax(dim=(2, 3), keepdim=True) - target_c.amin(dim=(2, 3), keepdim=True)).clamp_min(1e-6)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        ((mu_x * mu_x) + (mu_y * mu_y) + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return 1.0 - ssim_map.mean()


def differentiable_multichannel_ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_weights: torch.Tensor,
    window_size: int = 3,
) -> torch.Tensor:
    weights = channel_weights.flatten()
    weights = weights / weights.sum().clamp_min(1e-8)
    losses = [
        weights[channel] * differentiable_ssim_loss(pred, target, channel=channel, window_size=window_size)
        for channel in range(pred.shape[1])
    ]
    return torch.stack(losses).sum()


def complex_gain_consistency_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    dataset: CSIGridDataset,
    region: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep CSI_Gain aligned with the predicted complex channel.

    The project stores CSI_Gain as 10*log10(|h|), i.e.
    5*log10(Re^2 + Im^2). This loss does not change the model structure; it
    only discourages the three output channels from becoming physically
    inconsistent during refiner training.
    """
    device = pred.device
    dtype = pred.dtype
    c_min = torch.tensor(dataset.global_min, dtype=dtype, device=device).view(1, 3, 1, 1)
    c_max = torch.tensor(dataset.global_max, dtype=dtype, device=device).view(1, 3, 1, 1)
    scale = (c_max - c_min).clamp_min(1e-12)

    pred_phys = (pred + 1.0) * 0.5 * scale + c_min
    target_phys = (target + 1.0) * 0.5 * scale + c_min
    gain_from_pred_complex = 5.0 * torch.log10(
        pred_phys[:, 1:2] * pred_phys[:, 1:2] + pred_phys[:, 2:3] * pred_phys[:, 2:3] + 1e-15
    )
    gain_from_pred_complex_norm = ((gain_from_pred_complex - c_min[:, 0:1]) / scale[:, 0:1]).clamp(0.0, 1.0)
    gain_from_pred_complex_norm = gain_from_pred_complex_norm * 2.0 - 1.0

    pred_gain_consistency = F.l1_loss(pred[:, 0:1] * region, gain_from_pred_complex_norm * region, reduction="sum")
    target_gain_consistency = F.l1_loss(
        target[:, 0:1] * region,
        gain_from_pred_complex_norm * region,
        reduction="sum",
    )
    denom = region.sum().clamp_min(1.0)
    return pred_gain_consistency / denom, target_gain_consistency / denom


def train_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    channel_weights: torch.Tensor,
    dataset: CSIGridDataset,
    *,
    missing_mse_weight: float,
    full_mse_weight: float,
    missing_l1_weight: float,
    gradient_weight: float,
    ssim_weight: float,
    complex_gain_weight: float,
    complex_target_gain_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    missing = 1.0 - mask
    sq = (pred - target) ** 2
    abs_err = torch.abs(pred - target)
    missing_mse = weighted_mean(sq, channel_weights, missing)
    full_mse = weighted_mean(sq, channel_weights, None)
    missing_l1 = weighted_mean(abs_err, channel_weights, missing)
    grad = gradient_l1(pred, target, channel=0)
    ssim_loss = differentiable_multichannel_ssim_loss(pred, target, channel_weights)
    pred_gain_consistency, target_gain_consistency = complex_gain_consistency_losses(pred, target, dataset, missing)
    total = (
        missing_mse_weight * missing_mse
        + full_mse_weight * full_mse
        + missing_l1_weight * missing_l1
        + gradient_weight * grad
        + ssim_weight * ssim_loss
        + complex_gain_weight * pred_gain_consistency
        + complex_target_gain_weight * target_gain_consistency
    )
    return total, {
        "missing_mse": float(missing_mse.detach().cpu()),
        "full_mse": float(full_mse.detach().cpu()),
        "missing_l1": float(missing_l1.detach().cpu()),
        "gradient_l1": float(grad.detach().cpu()),
        "ssim_loss": float(ssim_loss.detach().cpu()),
        "pred_gain_consistency": float(pred_gain_consistency.detach().cpu()),
        "target_gain_consistency": float(target_gain_consistency.detach().cpu()),
    }


def channel_ssim(dataset: CSIGridDataset, gt_norm: np.ndarray, pred_norm: np.ndarray, channel: int) -> float:
    gt_gain = dataset.inverse_normalize_channel(gt_norm, channel)
    pred_gain = dataset.inverse_normalize_channel(pred_norm, channel)
    data_range = float(np.nanmax(gt_gain) - np.nanmin(gt_gain))
    if data_range <= 0:
        data_range = 1.0
    return float(ssim(gt_gain.astype(np.float32), pred_gain.astype(np.float32), data_range=data_range, win_size=3))


@torch.no_grad()
def evaluate(
    model: PINNCSIReconstructor,
    loader: DataLoader,
    dataset: CSIGridDataset,
    device: str,
    output_input_blend: float,
) -> Dict[str, float]:
    model.eval()
    ssim_values = {0: [], 1: [], 2: []}
    mse_values = []
    for batch in loader:
        gt = batch["gt_csi"].to(device)
        sparse = batch["sparse_csi"].to(device)
        mask = batch["mask"].to(device)
        diff_recon = batch["diffusion_recon"].to(device)
        pred = model(diff_recon, mask)
        if output_input_blend > 0.0:
            pred = (1.0 - output_input_blend) * pred + output_input_blend * diff_recon
        pred = pred * (1.0 - mask) + sparse * mask
        gt_np = gt.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        missing_np = (1.0 - mask[:, 0]).detach().cpu().numpy()
        for idx in range(gt_np.shape[0]):
            for channel in range(3):
                ssim_values[channel].append(channel_ssim(dataset, gt_np[idx, channel], pred_np[idx, channel], channel))
            gt_gain = dataset.inverse_normalize_channel(gt_np[idx, 0], 0)
            pred_gain = dataset.inverse_normalize_channel(pred_np[idx, 0], 0)
            denom = max(float(missing_np[idx].sum()), 1.0)
            mse_values.append(float(((pred_gain - gt_gain) ** 2 * missing_np[idx]).sum() / denom))
    gain_ssim = float(np.mean(ssim_values[0])) if ssim_values[0] else float("nan")
    real_ssim = float(np.mean(ssim_values[1])) if ssim_values[1] else float("nan")
    imag_ssim = float(np.mean(ssim_values[2])) if ssim_values[2] else float("nan")
    return {
        "gain_ssim": gain_ssim,
        "real_ssim": real_ssim,
        "imag_ssim": imag_ssim,
        "min_ssim": min(gain_ssim, real_ssim, imag_ssim),
        "mean_ssim": float(np.mean([gain_ssim, real_ssim, imag_ssim])),
        "gain_mse_missing_phys": float(np.mean(mse_values)) if mse_values else float("nan"),
    }


def save_plain_checkpoint(path: Path, model: PINNCSIReconstructor, metadata: Dict[str, object]) -> None:
    torch.save(model.state_dict(), path)
    path.with_suffix(path.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_report(path: Path, config: Dict[str, object], best: Dict[str, object], status: str) -> None:
    lines = [
        "# Full PIDR EL Training Report",
        "",
        f"Status: {status}",
        "",
        "## Target",
        "",
        f"- Target train minimum channel SSIM: {config['target_train_ssim']}",
        f"- Target test minimum channel SSIM: {config['target_test_ssim']}",
        "- Success condition: all three channels reach the target on both train and test after diffusion -> refiner.",
        "",
        "## Best Result",
        "",
        f"- Best train Gain SSIM: {best.get('best_train_ssim')}",
        f"- Best test Gain SSIM: {best.get('best_test_ssim')}",
        f"- Best train min-channel SSIM: {best.get('best_train_min_ssim')}",
        f"- Best test min-channel SSIM: {best.get('best_test_min_ssim')}",
        f"- Best epoch: {best.get('best_epoch')}",
        f"- Diffusion checkpoint: {config['diffusion_ckpt']}",
        f"- Refiner checkpoint: {best.get('best_refiner_checkpoint')}",
        "",
        "## Configuration",
        "",
    ]
    for key, value in config.items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    configure_data_shape(args.data_path, args.img_height, args.img_width, args.train_sample_count)
    configure_mask_ratio(args.mask_ratio)
    if args.diffusion_base_channels is None and args.cache_postprocess != "linear_interp_only":
        args.diffusion_base_channels = infer_diffusion_base_channels(args.diffusion_ckpt)
    diffusion_base_channels = configure_diffusion_base_channels(args.diffusion_base_channels)
    set_seed(args.seed)
    device = args.device
    run_dir = make_run_dir(args.run_root)
    print(f"Run directory: {run_dir}")

    train_dataset = CSIGridDataset(
        args.data_path,
        mode="train",
        split_ratio=args.split_ratio,
        seed=args.split_seed,
        gain_in_db=True,
        mask_strategy=args.mask_strategy,
        mask_seed=args.train_mask_seed,
    )
    test_dataset = CSIGridDataset(
        args.data_path,
        mode="test",
        split_ratio=args.split_ratio,
        seed=args.split_seed,
        gain_in_db=True,
        mask_strategy=args.mask_strategy,
        mask_seed=args.eval_mask_seed,
    )

    cache_dir = Path(args.reuse_cache_dir) if args.reuse_cache_dir else run_dir / "cache"
    train_cache = build_or_load_diffusion_cache(
        cache_path=cache_dir / "diffusion_train.pt",
        dataset=train_dataset,
        diffusion_ckpt=args.diffusion_ckpt,
        device=device,
        batch_size=args.cache_batch_size,
        sample_stride=args.sample_stride,
        force_rebuild=args.force_rebuild_cache,
        seed=args.diffusion_sample_seed,
        postprocess=args.cache_postprocess,
        interp_weight=args.interp_weight,
        diffusion_base_channels=diffusion_base_channels,
        sampling_method=args.sampling_method,
    )
    test_cache = build_or_load_diffusion_cache(
        cache_path=cache_dir / "diffusion_test.pt",
        dataset=test_dataset,
        diffusion_ckpt=args.diffusion_ckpt,
        device=device,
        batch_size=args.cache_batch_size,
        sample_stride=args.sample_stride,
        force_rebuild=args.force_rebuild_cache,
        seed=args.diffusion_sample_seed,
        postprocess=args.cache_postprocess,
        interp_weight=args.interp_weight,
        diffusion_base_channels=diffusion_base_channels,
        sampling_method=args.sampling_method,
    )

    train_ds = PIDRCacheDataset(train_dataset, train_cache)
    test_ds = PIDRCacheDataset(test_dataset, test_cache)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    train_eval_loader = DataLoader(train_ds, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False)

    model = PINNCSIReconstructor().to(device)
    if args.init_refiner_ckpt:
        model.load_state_dict(load_plain_or_wrapped_state_dict(args.init_refiner_ckpt))
        print(f"Initialized refiner from: {args.init_refiner_ckpt}")
    if args.freeze_bn:
        freeze_batch_norm_modules(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_decay_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    channel_weights = torch.tensor([args.gain_weight, args.real_weight, args.imag_weight], device=device).view(1, 3, 1, 1)

    config = vars(args).copy()
    config.update(
        {
            "run_dir": str(run_dir),
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "mask_ratio": Config.MASK_RATIO,
            "train_cache": str(cache_dir / "diffusion_train.pt"),
            "test_cache": str(cache_dir / "diffusion_test.pt"),
            "reuse_cache_dir": args.reuse_cache_dir,
            "diffusion_signature": checkpoint_signature(args.diffusion_ckpt)
            if args.cache_postprocess != "linear_interp_only"
            else {"linear_interp_only": True},
            "diffusion_base_channels": diffusion_base_channels,
            "checkpoint_format": "plain_state_dict",
            "pipeline": "diffusion_reconstruction -> PINNCSIReconstructor",
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history_path = run_dir / "history.csv"
    fields = [
        "epoch",
        "lr",
        "train_loss",
        "missing_mse",
        "full_mse",
        "missing_l1",
        "gradient_l1",
        "ssim_loss",
        "pred_gain_consistency",
        "target_gain_consistency",
        "train_gain_ssim",
        "train_real_ssim",
        "train_imag_ssim",
        "train_min_ssim",
        "train_mean_ssim",
        "test_gain_ssim",
        "test_real_ssim",
        "test_imag_ssim",
        "test_min_ssim",
        "test_mean_ssim",
        "train_gain_mse_missing_phys",
        "test_gain_mse_missing_phys",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv.DictWriter(csv_file, fieldnames=fields).writeheader()

    best_train_ssim = -1.0
    best_test_ssim = -1.0
    best_train_min_ssim = -1.0
    best_test_min_ssim = -1.0
    best_epoch = 0
    best_refiner_checkpoint = None
    status = "not_reached"

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_bn:
            freeze_batch_norm_modules(model)
        loss_sum = 0.0
        part_sums = {
            key: 0.0
            for key in [
                "missing_mse",
                "full_mse",
                "missing_l1",
                "gradient_l1",
                "ssim_loss",
                "pred_gain_consistency",
                "target_gain_consistency",
            ]
        }
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            gt = batch["gt_csi"].to(device)
            sparse = batch["sparse_csi"].to(device)
            mask = batch["mask"].to(device)
            diff_recon = batch["diffusion_recon"].to(device)
            pred = model(diff_recon, mask)
            if args.output_input_blend > 0.0:
                pred = (1.0 - args.output_input_blend) * pred + args.output_input_blend * diff_recon
            pred = pred * (1.0 - mask) + sparse * mask
            loss, parts = train_loss(
                pred,
                gt,
                mask,
                channel_weights,
                train_dataset,
                missing_mse_weight=args.missing_mse_weight,
                full_mse_weight=args.full_mse_weight,
                missing_l1_weight=args.missing_l1_weight,
                gradient_weight=args.gradient_weight,
                ssim_weight=args.ssim_weight,
                complex_gain_weight=args.complex_gain_weight,
                complex_target_gain_weight=args.complex_target_gain_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            for key, value in parts.items():
                part_sums[key] += value

        avg_loss = loss_sum / max(1, len(train_loader))
        avg_parts = {key: value / max(1, len(train_loader)) for key, value in part_sums.items()}

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            train_eval = evaluate(model, train_eval_loader, train_dataset, device, args.output_input_blend)
            test_eval = evaluate(model, test_loader, test_dataset, device, args.output_input_blend)
            train_ssim = train_eval["gain_ssim"]
            test_ssim = test_eval["gain_ssim"]
            train_min_ssim = train_eval["min_ssim"]
            test_min_ssim = test_eval["min_ssim"]
            scheduler.step(max(train_min_ssim, test_min_ssim))
            print(
                f"Epoch {epoch}: loss={avg_loss:.6f}, "
                f"train_ssim(G/R/I/min)={train_eval['gain_ssim']:.4f}/{train_eval['real_ssim']:.4f}/"
                f"{train_eval['imag_ssim']:.4f}/{train_min_ssim:.4f}, "
                f"test_ssim(G/R/I/min)={test_eval['gain_ssim']:.4f}/{test_eval['real_ssim']:.4f}/"
                f"{test_eval['imag_ssim']:.4f}/{test_min_ssim:.4f}, "
                f"train_mse={train_eval['gain_mse_missing_phys']:.4g}, "
                f"test_mse={test_eval['gain_mse_missing_phys']:.4g}"
            )

            improved = train_min_ssim > best_train_min_ssim or test_min_ssim > best_test_min_ssim
            if improved:
                best_train_ssim = max(best_train_ssim, train_ssim)
                best_test_ssim = max(best_test_ssim, test_ssim)
                best_train_min_ssim = max(best_train_min_ssim, train_min_ssim)
                best_test_min_ssim = max(best_test_min_ssim, test_min_ssim)
                best_epoch = epoch
                best_refiner_checkpoint = run_dir / (
                    f"best_refiner_epoch_{epoch:04d}_trainmin{train_min_ssim:.4f}_testmin{test_min_ssim:.4f}.pth"
                )
                save_plain_checkpoint(
                    best_refiner_checkpoint,
                    model,
                    {"epoch": epoch, "train_eval": train_eval, "test_eval": test_eval, "config": config},
                )

            with history_path.open("a", newline="", encoding="utf-8") as csv_file:
                csv.DictWriter(csv_file, fieldnames=fields).writerow(
                    {
                        "epoch": epoch,
                        "lr": optimizer.param_groups[0]["lr"],
                        "train_loss": avg_loss,
                        **avg_parts,
                        "train_gain_ssim": train_ssim,
                        "train_real_ssim": train_eval["real_ssim"],
                        "train_imag_ssim": train_eval["imag_ssim"],
                        "train_min_ssim": train_min_ssim,
                        "train_mean_ssim": train_eval["mean_ssim"],
                        "test_gain_ssim": test_ssim,
                        "test_real_ssim": test_eval["real_ssim"],
                        "test_imag_ssim": test_eval["imag_ssim"],
                        "test_min_ssim": test_min_ssim,
                        "test_mean_ssim": test_eval["mean_ssim"],
                        "train_gain_mse_missing_phys": train_eval["gain_mse_missing_phys"],
                        "test_gain_mse_missing_phys": test_eval["gain_mse_missing_phys"],
                    }
                )

            if train_min_ssim >= args.target_train_ssim and test_min_ssim >= args.target_test_ssim:
                status = "target_reached"
                target_checkpoint = run_dir / (
                    f"target_refiner_epoch_{epoch:04d}_trainmin{train_min_ssim:.4f}_testmin{test_min_ssim:.4f}.pth"
                )
                save_plain_checkpoint(
                    target_checkpoint,
                    model,
                    {"epoch": epoch, "train_eval": train_eval, "test_eval": test_eval, "config": config},
                )
                best_refiner_checkpoint = target_checkpoint
                print(f"Target reached. Saved: {target_checkpoint}")
                break

    if args.copy_diffusion_to_run:
        copied = run_dir / Path(args.diffusion_ckpt).name
        if not copied.exists():
            shutil.copy2(args.diffusion_ckpt, copied)
            (run_dir / f"{copied.name}.source.json").write_text(
                json.dumps({"source": args.diffusion_ckpt, "signature": checkpoint_signature(args.diffusion_ckpt)}, indent=2),
                encoding="utf-8",
            )

    best = {
        "best_train_ssim": best_train_ssim,
        "best_test_ssim": best_test_ssim,
        "best_train_min_ssim": best_train_min_ssim,
        "best_test_min_ssim": best_test_min_ssim,
        "best_epoch": best_epoch,
        "diffusion_checkpoint": args.diffusion_ckpt,
        "best_refiner_checkpoint": str(best_refiner_checkpoint) if best_refiner_checkpoint else None,
    }
    (run_dir / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    write_report(run_dir / "report.md", config, best, status)
    print(f"Final status: {status}")
    print(f"Report: {run_dir / 'report.md'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train full PIDR refiner on cached diffusion outputs for EL data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", default="./data/el.mat")
    parser.add_argument("--run-root", default="./runs/pidr_ssim_target")
    parser.add_argument("--img-height", type=int, default=None)
    parser.add_argument("--img-width", type=int, default=None)
    parser.add_argument("--train-sample-count", type=int, default=None)
    parser.add_argument("--device", default=Config.DEVICE)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-ratio", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--mask-ratio", type=float, default=None, help="Missing ratio. Defaults to Config.MASK_RATIO.")
    parser.add_argument("--mask-strategy", default="random", choices=["random", "uniform", "extreme_uneven"])
    parser.add_argument("--train-mask-seed", type=int, default=12345)
    parser.add_argument("--eval-mask-seed", type=int, default=0)
    parser.add_argument("--diffusion-ckpt", default="./checkpoints/el/final/diffusion.pth")
    parser.add_argument("--diffusion-base-channels", type=int, default=None)
    parser.add_argument("--init-refiner-ckpt", default="")
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--sampling-method", default="ddpm", choices=["ddpm", "ddim", "ddim_x0"])
    parser.add_argument("--diffusion-sample-seed", type=int, default=0)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--reuse-cache-dir", default="", help="Directory containing diffusion_train.pt and diffusion_test.pt.")
    parser.add_argument("--cache-postprocess", default="none", choices=["none", "linear_interp_blend", "linear_interp_only"])
    parser.add_argument("--interp-weight", type=float, default=0.0)
    parser.add_argument("--copy-diffusion-to-run", action="store_true")
    parser.add_argument("--cache-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=216)
    parser.add_argument("--eval-batch-size", type=int, default=216)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--freeze-bn", action="store_true", help="Keep refiner BatchNorm running stats and affine params fixed.")
    parser.add_argument("--gain-weight", type=float, default=16.0)
    parser.add_argument("--real-weight", type=float, default=0.05)
    parser.add_argument("--imag-weight", type=float, default=0.05)
    parser.add_argument("--missing-mse-weight", type=float, default=0.04)
    parser.add_argument("--full-mse-weight", type=float, default=0.02)
    parser.add_argument("--missing-l1-weight", type=float, default=0.01)
    parser.add_argument("--gradient-weight", type=float, default=0.002)
    parser.add_argument("--ssim-weight", type=float, default=14.0)
    parser.add_argument("--complex-gain-weight", type=float, default=0.0)
    parser.add_argument("--complex-target-gain-weight", type=float, default=0.0)
    parser.add_argument("--output-input-blend", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--target-train-ssim", type=float, default=0.9)
    parser.add_argument("--target-test-ssim", type=float, default=0.9)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
