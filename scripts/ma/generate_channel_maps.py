"""Batch PIDR channel map reconstruction with heatmap comparisons.

This script reconstructs PIDR channel maps for a
multi-sample .mat file. It loads CSI_Gain, CSI_Real and CSI_Imag, applies a
sparse observation mask to each sample, runs the PIDR diffusion + refiner
pipeline, saves reconstructed maps, and plots GT / PIDR / absolute error
heatmaps for visual comparison.

Example:
    python -m scripts.ma.generate_channel_maps ^
        --input ./data/test/el.mat ^
        --output ./data/test/el_pidr_batch.mat ^
        --diffusion-ckpt ./checkpoints/el/final/diffusion.pth ^
        --refiner-ckpt ./checkpoints/el/final/refiner.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata
from scipy.io import loadmat, savemat
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from pidr.models.pinn_refiner import PINNCSIReconstructor
from pidr.config import Config
from pidr.diffusion import DiffusionManager
from pidr.models.diffusion_unet import CSIDiffusionPhysicsModel


CHANNEL_FIELDS = ("CSI_Gain", "CSI_Real", "CSI_Imag")
CHANNEL_LABELS = ("Gain", "Real", "Imag")


def load_checkpoint_metadata(checkpoint_path: str) -> Dict[str, object]:
    metadata_path = f"{checkpoint_path}.metadata.json"
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    return metadata if isinstance(metadata, dict) else {}


def load_plain_or_wrapped_state_dict(checkpoint_path: str, device: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def infer_diffusion_base_channels(checkpoint_path: str) -> int:
    state_dict = load_plain_or_wrapped_state_dict(checkpoint_path, "cpu")
    weight = state_dict.get("input_proj.weight") if isinstance(state_dict, dict) else None
    if weight is None or not hasattr(weight, "shape"):
        return int(Config.BASE_CHANNELS)
    return int(weight.shape[0])


def configure_diffusion_base_channels(base_channels: int | None) -> int:
    if base_channels is None:
        return int(Config.BASE_CHANNELS)
    if base_channels <= 0:
        raise ValueError(f"diffusion_base_channels must be > 0, got {base_channels}")
    Config.BASE_CHANNELS = int(base_channels)
    Config.MODEL.BASE_CHANNELS = int(base_channels)
    return int(base_channels)


def configure_data_shape_from_mat(mat_path: str) -> None:
    """Infer H/W from CSI_Gain and update Config before parsing fields."""
    mat_data = loadmat(mat_path)
    gain = np.squeeze(_extract_field_case_insensitive(mat_data, "CSI_Gain"))
    if gain.ndim == 2:
        height, width = gain.shape
    elif gain.ndim == 3:
        shape = tuple(int(dim) for dim in gain.shape)
        current_hw = (int(Config.IMG_HEIGHT), int(Config.IMG_WIDTH))
        if shape[0] == shape[1]:
            height, width = shape[0], shape[1]
        elif shape[0:2] == current_hw:
            height, width = shape[0], shape[1]
        elif shape[1:3] == current_hw:
            height, width = shape[1], shape[2]
        elif (shape[0], shape[2]) == current_hw:
            height, width = shape[0], shape[2]
        else:
            spatial = sorted(shape, reverse=True)[:2]
            height, width = sorted(spatial)
    else:
        raise ValueError(f"Cannot infer image shape from CSI_Gain shape {gain.shape}")

    Config.IMG_HEIGHT = int(height)
    Config.IMG_WIDTH = int(width)
    Config.DATA.IMG_HEIGHT = int(height)
    Config.DATA.IMG_WIDTH = int(width)


def sample_physics_guided(
    diffusion: DiffusionManager,
    model: torch.nn.Module,
    sparse_csi: torch.Tensor,
    mask: torch.Tensor,
    *,
    stride: int = 1,
) -> torch.Tensor:
    """Physics-guided ancestral sampling without per-step progress output."""
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
        x = 1 / torch.sqrt(alpha) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
        ) + torch.sqrt(beta) * noise
        x = x * (1 - mask) + sparse_csi * mask

    return x.clamp(-1.0, 1.0)


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
    """Deterministic DDIM-style physics-guided sampling."""
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


def _extract_field_case_insensitive(mat_data: Dict[str, np.ndarray], field_name: str) -> np.ndarray:
    if field_name in mat_data:
        return np.asarray(mat_data[field_name])

    lower_map = {str(key).lower(): key for key in mat_data.keys()}
    lower_name = field_name.lower()
    if lower_name in lower_map:
        return np.asarray(mat_data[lower_map[lower_name]])

    available = ", ".join(sorted(str(key) for key in mat_data.keys() if not str(key).startswith("__")))
    raise KeyError(f"Missing field '{field_name}'. Available fields: {available}")


def _to_sample_first(array: np.ndarray, field_name: str) -> np.ndarray:
    """Convert a 2D/3D CSI field to shape [N, H, W]."""
    array = np.asarray(array, dtype=np.float32)
    array = np.squeeze(array)
    img_height = Config.IMG_HEIGHT
    img_width = Config.IMG_WIDTH

    if array.ndim == 2:
        if array.shape != (img_height, img_width):
            raise ValueError(f"{field_name} 2D shape must be ({img_height}, {img_width}), got {array.shape}")
        return array[None, :, :]

    if array.ndim != 3:
        raise ValueError(f"{field_name} must be 2D or 3D, got shape {array.shape}")

    if array.shape[0] == img_height and array.shape[1] == img_width:
        return np.moveaxis(array, -1, 0)

    if array.shape[1] == img_height and array.shape[2] == img_width:
        return array

    if array.shape[0] == img_height and array.shape[2] == img_width:
        return np.moveaxis(array, 1, 0)

    raise ValueError(
        f"{field_name} has unsupported shape {array.shape}; expected (H,W,N), (N,H,W), or (H,N,W) "
        f"with H={img_height}, W={img_width}."
    )


def load_multi_sample_input(mat_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load CSI fields and return raw samples [N, 3, H, W] plus valid masks [N, H, W]."""
    mat_data = loadmat(mat_path)
    blocks = []

    for field in CHANNEL_FIELDS:
        block = _to_sample_first(_extract_field_case_insensitive(mat_data, field), field)
        blocks.append(block)

    if not (blocks[0].shape == blocks[1].shape == blocks[2].shape):
        raise ValueError(
            "Field shapes do not match after sample-first conversion: "
            f"Gain={blocks[0].shape}, Real={blocks[1].shape}, Imag={blocks[2].shape}"
        )

    raw = np.stack(blocks, axis=1).astype(np.float32)
    valid_masks = np.all(np.isfinite(raw), axis=1).astype(np.float32)
    return raw, valid_masks


def compute_normalization_stats(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute file-wide min/max for each channel."""
    global_min = []
    global_max = []

    for channel_idx in range(raw.shape[1]):
        values = raw[:, channel_idx]
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise ValueError(f"Channel {channel_idx} does not contain finite values.")

        c_min = float(finite_values.min())
        c_max = float(finite_values.max())
        if np.isclose(c_min, c_max):
            c_max = c_min + 1.0
        global_min.append(c_min)
        global_max.append(c_max)

    return np.asarray(global_min, dtype=np.float32), np.asarray(global_max, dtype=np.float32)


def load_normalization_stats(stats_mat_path: str | None) -> Tuple[np.ndarray, np.ndarray] | None:
    if not stats_mat_path:
        return None
    stats_raw, _stats_valid = load_multi_sample_input(stats_mat_path)
    return compute_normalization_stats(stats_raw)


def normalize(raw: np.ndarray, global_min: np.ndarray, global_max: np.ndarray) -> np.ndarray:
    scale = np.maximum(global_max - global_min, 1e-12).reshape(1, 3, 1, 1)
    safe_raw = np.where(np.isfinite(raw), raw, global_min.reshape(1, 3, 1, 1))
    normalized = (safe_raw - global_min.reshape(1, 3, 1, 1)) / scale
    normalized = np.clip(normalized, 0.0, 1.0) * 2.0 - 1.0
    return normalized.astype(np.float32)


def inverse_normalize(norm_map: np.ndarray, global_min: np.ndarray, global_max: np.ndarray) -> np.ndarray:
    scale = np.maximum(global_max - global_min, 1e-12).reshape(1, 3, 1, 1)
    rescaled = (norm_map + 1.0) / 2.0
    return (rescaled * scale + global_min.reshape(1, 3, 1, 1)).astype(np.float32)


def generate_mask(
    valid_mask: np.ndarray,
    mask_ratio: float,
    strategy: str = "random",
    seed: int | None = None,
) -> np.ndarray:
    """Generate a single observation mask, where 1 means observed."""
    if not (0.0 <= mask_ratio < 1.0):
        raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")

    observed_ratio = 1.0 - mask_ratio
    valid_positions = np.argwhere(valid_mask > 0.5)
    if len(valid_positions) == 0:
        return np.zeros_like(valid_mask, dtype=np.float32)

    target_count = max(1, int(round(len(valid_positions) * observed_ratio)))
    rng = np.random.default_rng(seed)

    if strategy == "random":
        indices = rng.choice(len(valid_positions), size=min(target_count, len(valid_positions)), replace=False)
        selected = valid_positions[indices]
    elif strategy == "uniform":
        height, width = valid_mask.shape
        aspect_ratio = max(width / max(height, 1), 1e-6)
        points_y = max(1, int(round(np.sqrt(target_count / aspect_ratio))))
        points_x = max(1, int(np.ceil(target_count / points_y)))
        ys = np.linspace(0, height - 1, points_y)
        xs = np.linspace(0, width - 1, points_x)
        grid_positions = np.array([(int(round(y)), int(round(x))) for y in ys for x in xs], dtype=np.int32)
        grid_positions = np.unique(grid_positions, axis=0)

        valid_set = {tuple(position) for position in valid_positions.tolist()}
        selected_list = [position for position in grid_positions.tolist() if tuple(position) in valid_set]

        if len(selected_list) > target_count:
            sample_indices = np.linspace(0, len(selected_list) - 1, target_count, dtype=int)
            selected = np.asarray(selected_list, dtype=np.int32)[sample_indices]
        else:
            selected_set = {tuple(position) for position in selected_list}
            remaining = np.asarray(
                [position for position in valid_positions.tolist() if tuple(position) not in selected_set],
                dtype=np.int32,
            )
            fill_count = min(target_count - len(selected_list), len(remaining))
            if fill_count > 0:
                fill_indices = rng.choice(len(remaining), size=fill_count, replace=False)
                selected_list.extend(remaining[fill_indices].tolist())
            selected = np.asarray(selected_list, dtype=np.int32)
    elif strategy in {"extreme_uneven", "very_uneven"}:
        height, width = valid_mask.shape
        center_y = rng.uniform(0.15 * height, 0.85 * height)
        center_x = rng.uniform(0.15 * width, 0.85 * width)
        distances = np.sqrt((valid_positions[:, 0] - center_y) ** 2 + (valid_positions[:, 1] - center_x) ** 2)
        weights = np.exp(-distances / max(height, width) * 5.0)
        weights = weights / weights.sum()
        indices = rng.choice(
            len(valid_positions),
            size=min(target_count, len(valid_positions)),
            replace=False,
            p=weights,
        )
        selected = valid_positions[indices]
    else:
        raise ValueError(f"Unsupported mask strategy: {strategy}")

    mask = np.zeros_like(valid_mask, dtype=np.float32)
    mask[selected[:, 0], selected[:, 1]] = 1.0
    return mask


def interpolate_sparse_batch(sparse: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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


def plot_heatmap_comparison(
    *,
    save_path: str,
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    sparse_map: np.ndarray,
    mask: np.ndarray,
    sample_idx: int,
    channel_label: str,
) -> None:
    error_map = np.abs(pred_map - gt_map)
    sparse_display = sparse_map.copy()
    sparse_display[mask < 0.5] = np.nan

    vmin = float(np.nanmin(gt_map))
    vmax = float(np.nanmax(gt_map))
    err_max = float(np.nanmax(error_map))
    if err_max <= 0:
        err_max = 1.0

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    panels = [
        (gt_map, "Ground Truth", "jet", vmin, vmax),
        (sparse_display, "Sparse Input", "jet", vmin, vmax),
        (pred_map, "PIDR Output", "jet", vmin, vmax),
        (error_map, "Absolute Error", "magma", 0.0, err_max),
    ]

    for ax, (data, title, cmap, panel_vmin, panel_vmax) in zip(axes, panels):
        data = np.ma.masked_invalid(np.asarray(data, dtype=np.float32))
        height, width = data.shape
        mesh = ax.pcolormesh(
            np.arange(width + 1),
            np.arange(height + 1),
            data,
            cmap=cmap,
            vmin=panel_vmin,
            vmax=panel_vmax,
            shading="flat",
        )
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Sample {sample_idx} - {channel_label}", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def calc_metrics(gt_map: np.ndarray, pred_map: np.ndarray) -> Tuple[float, float, float]:
    diff = pred_map - gt_map
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    denom = float(np.mean(gt_map * gt_map))
    nmse_db = float(10.0 * np.log10(mse / (denom + 1e-10)))
    data_range = float(np.nanmax(gt_map) - np.nanmin(gt_map))
    if data_range <= 0:
        data_range = 1.0
    ssim_val = float(ssim(gt_map.astype(np.float32), pred_map.astype(np.float32), data_range=data_range, win_size=3))
    return mse, mae, nmse_db, ssim_val


def load_models(diffusion_ckpt: str, refiner_ckpt: str, device: str, diffusion_base_channels: int | None) -> Tuple[torch.nn.Module, torch.nn.Module]:
    if not os.path.exists(diffusion_ckpt):
        raise FileNotFoundError(f"Diffusion checkpoint not found: {diffusion_ckpt}")
    if not os.path.exists(refiner_ckpt):
        raise FileNotFoundError(f"Refiner checkpoint not found: {refiner_ckpt}")

    diffusion_base_channels = configure_diffusion_base_channels(diffusion_base_channels)
    print(f"Diffusion base channels: {diffusion_base_channels}")
    diffusion_model = CSIDiffusionPhysicsModel().to(device).eval()
    diffusion_model.load_state_dict(load_plain_or_wrapped_state_dict(diffusion_ckpt, device))

    refiner_model = PINNCSIReconstructor().to(device).eval()
    refiner_model.load_state_dict(load_plain_or_wrapped_state_dict(refiner_ckpt, device))
    return diffusion_model, refiner_model


def generate_pidr_maps_batch(
    *,
    input_path: str,
    output_path: str,
    heatmap_dir: str,
    metrics_path: str,
    diffusion_ckpt: str,
    refiner_ckpt: str,
    mask_ratio: float,
    mask_strategy: str,
    mask_seed: int | None,
    device: str,
    diffusion_base_channels: int | None,
    sample_stride: int,
    sampling_method: str,
    seed: int,
    batch_size: int,
    num_samples: int | None,
    max_plots: int | None,
    plot_channel: int,
    enforce_observed: bool,
    norm_stats_mat: str | None,
    cache_postprocess: str,
    interp_weight: float,
    output_input_blend: float,
) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    configure_data_shape_from_mat(input_path)
    raw, valid_masks = load_multi_sample_input(input_path)
    if num_samples is not None:
        raw = raw[:num_samples]
        valid_masks = valid_masks[:num_samples]

    if raw.shape[0] == 0:
        raise ValueError("No samples available for reconstruction.")
    if not (0 <= plot_channel < len(CHANNEL_LABELS)):
        raise ValueError(f"plot_channel must be 0, 1 or 2, got {plot_channel}")

    norm_stats = load_normalization_stats(norm_stats_mat)
    if norm_stats is None:
        global_min, global_max = compute_normalization_stats(raw)
        print("Normalization stats: input file")
    else:
        global_min, global_max = norm_stats
        print(f"Normalization stats: {norm_stats_mat}")
    gt_norm = normalize(raw, global_min, global_max)
    gt_phys = inverse_normalize(gt_norm, global_min, global_max)

    print(f"Input: {input_path}")
    print(f"Samples: {raw.shape[0]}, map size: {raw.shape[-2]}x{raw.shape[-1]}")
    print(f"Global min: {global_min.tolist()}")
    print(f"Global max: {global_max.tolist()}")
    print(f"Device: {device}")
    print(
        f"Inference settings: sampling_method={sampling_method}, sample_stride={sample_stride}, "
        f"cache_postprocess={cache_postprocess}, "
        f"interp_weight={interp_weight}, output_input_blend={output_input_blend}, seed={seed}"
    )

    diffusion_model, refiner_model = load_models(diffusion_ckpt, refiner_ckpt, device, diffusion_base_channels)
    diffusion_mgr = DiffusionManager(device, global_min, global_max)

    masks = np.empty((raw.shape[0], 1, raw.shape[-2], raw.shape[-1]), dtype=np.float32)
    for idx in range(raw.shape[0]):
        sample_seed = None if mask_seed is None else int(mask_seed) + idx
        masks[idx, 0] = generate_mask(valid_masks[idx], mask_ratio, mask_strategy, seed=sample_seed)

    sparse_norm = gt_norm * masks
    recon_norm = np.empty_like(gt_norm, dtype=np.float32)

    for start in tqdm(range(0, raw.shape[0], batch_size), desc="PIDR batch reconstruction"):
        end = min(start + batch_size, raw.shape[0])
        sparse_tensor = torch.from_numpy(sparse_norm[start:end]).float().to(device)
        mask_tensor = torch.from_numpy(masks[start:end]).float().to(device)

        with torch.no_grad():
            if sampling_method == "ddpm":
                diff_out = sample_physics_guided(
                    diffusion_mgr,
                    diffusion_model,
                    sparse_tensor,
                    mask_tensor,
                    stride=sample_stride,
                )
            elif sampling_method == "ddim":
                diff_out = sample_ddim_physics_guided(
                    diffusion_mgr,
                    diffusion_model,
                    sparse_tensor,
                    mask_tensor,
                    stride=sample_stride,
                    return_final_x0=False,
                )
            elif sampling_method == "ddim_x0":
                diff_out = sample_ddim_physics_guided(
                    diffusion_mgr,
                    diffusion_model,
                    sparse_tensor,
                    mask_tensor,
                    stride=sample_stride,
                    return_final_x0=True,
                )
            else:
                raise ValueError(f"Unsupported sampling_method: {sampling_method}")
            if cache_postprocess == "linear_interp_blend":
                interp = interpolate_sparse_batch(sparse_tensor, mask_tensor).to(device)
                diff_out = (1.0 - interp_weight) * diff_out + interp_weight * interp
                diff_out = diff_out * (1.0 - mask_tensor) + sparse_tensor * mask_tensor

            pidr_out = refiner_model(diff_out, mask_tensor)
            if output_input_blend > 0.0:
                pidr_out = (1.0 - output_input_blend) * pidr_out + output_input_blend * diff_out
            if enforce_observed:
                pidr_out = pidr_out * (1 - mask_tensor) + sparse_tensor * mask_tensor

        recon_norm[start:end] = pidr_out.detach().cpu().numpy().astype(np.float32)

    recon_phys = inverse_normalize(recon_norm, global_min, global_max)
    sparse_phys = inverse_normalize(sparse_norm, global_min, global_max)

    output_payload = {
        "CSI_Gain": np.moveaxis(recon_phys[:, 0], 0, -1),
        "CSI_Real": np.moveaxis(recon_phys[:, 1], 0, -1),
        "CSI_Imag": np.moveaxis(recon_phys[:, 2], 0, -1),
        "mask": np.moveaxis(masks[:, 0], 0, -1),
        "global_min": global_min,
        "global_max": global_max,
        "norm_stats_mat": norm_stats_mat or "",
        "cache_postprocess": cache_postprocess,
        "sampling_method": sampling_method,
        "diffusion_base_channels": int(Config.BASE_CHANNELS),
        "sample_stride": int(sample_stride),
        "interp_weight": float(interp_weight),
        "output_input_blend": float(output_input_blend),
        "diffusion_checkpoint": diffusion_ckpt,
        "refiner_checkpoint": refiner_ckpt,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    savemat(output_path, output_payload)

    rows = []
    plot_count = raw.shape[0] if max_plots is None else min(raw.shape[0], max(0, int(max_plots)))
    os.makedirs(heatmap_dir, exist_ok=True)

    for idx in range(raw.shape[0]):
        for channel_idx, channel_label in enumerate(CHANNEL_LABELS):
            mse, mae, nmse_db, ssim_val = calc_metrics(gt_phys[idx, channel_idx], recon_phys[idx, channel_idx])
            rows.append(
                {
                    "sample_idx": idx,
                    "channel": channel_label,
                    "observed_count": int(masks[idx, 0].sum()),
                    "mse": mse,
                    "mae": mae,
                    "nmse_db": nmse_db,
                    "ssim": ssim_val,
                }
            )

        if idx < plot_count:
            plot_heatmap_comparison(
                save_path=os.path.join(heatmap_dir, f"sample_{idx}_{CHANNEL_LABELS[plot_channel].lower()}.png"),
                gt_map=gt_phys[idx, plot_channel],
                pred_map=recon_phys[idx, plot_channel],
                sparse_map=sparse_phys[idx, plot_channel],
                mask=masks[idx, 0],
                sample_idx=idx,
                channel_label=CHANNEL_LABELS[plot_channel],
            )

    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    with open(metrics_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["sample_idx", "channel", "observed_count", "mse", "mae", "nmse_db", "ssim"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Reconstructed maps saved to: {output_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Heatmaps saved to: {heatmap_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch PIDR reconstruction for multi-sample CSI .mat files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, default="./data/test/fd.mat")
    parser.add_argument("--output", "-o", type=str, default="./data/test/fd_pidr_batch.mat")
    parser.add_argument("--heatmap-dir", type=str, default="./data/test/fd_pidr_heatmaps")
    parser.add_argument("--metrics", type=str, default="./data/test/fd_pidr_metrics.csv")
    parser.add_argument("--diffusion-ckpt", type=str, default="./checkpoints/fd/final/diffusion.pth")
    parser.add_argument("--refiner-ckpt", type=str, default="./checkpoints/fd/final/refiner.pth")
    parser.add_argument("--mask-ratio", type=float, default=Config.MASK_RATIO)
    parser.add_argument("--mask-strategy", type=str, default=None, choices=["random", "uniform", "extreme_uneven"])
    parser.add_argument("--mask-seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=Config.DEVICE)
    parser.add_argument("--diffusion-base-channels", type=int, default=None)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--sampling-method", default=None, choices=["ddpm", "ddim", "ddim_x0"])
    parser.add_argument("--seed", type=int, default=0, help="Seed for diffusion sampling noise.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=None, help="Limit the number of samples to reconstruct.")
    parser.add_argument("--max-plots", type=int, default=None, help="Limit saved heatmaps; default plots all samples.")
    parser.add_argument("--plot-channel", type=int, default=0, choices=[0, 1, 2], help="0=Gain, 1=Real, 2=Imag.")
    parser.add_argument("--norm-stats-mat", type=str, default=None, help="MAT file used to compute normalization min/max. Defaults to metadata data_path when available.")
    parser.add_argument("--cache-postprocess", default=None, choices=["none", "linear_interp_blend"])
    parser.add_argument("--interp-weight", type=float, default=None)
    parser.add_argument("--output-input-blend", type=float, default=None)
    parser.add_argument("--no-enforce-observed", action="store_true")
    args = parser.parse_args()

    metadata = load_checkpoint_metadata(args.refiner_ckpt)
    metadata_config = metadata.get("config", {}) if isinstance(metadata.get("config", {}), dict) else {}
    if metadata_config:
        print(f"Loaded refiner metadata: {args.refiner_ckpt}.metadata.json")
    if args.mask_strategy is None:
        args.mask_strategy = str(metadata_config.get("mask_strategy", "random"))
    if args.mask_seed is None:
        args.mask_seed = int(metadata_config.get("eval_mask_seed", Config.EVAL_MASK_SEED))
    if args.mask_ratio == Config.MASK_RATIO and "mask_ratio" in metadata_config:
        args.mask_ratio = float(metadata_config["mask_ratio"])
    if args.sample_stride == 1 and "sample_stride" in metadata_config:
        args.sample_stride = int(metadata_config["sample_stride"])
    if args.diffusion_base_channels is None and "diffusion_base_channels" in metadata_config:
        args.diffusion_base_channels = int(metadata_config["diffusion_base_channels"])
    if args.diffusion_base_channels is None:
        args.diffusion_base_channels = infer_diffusion_base_channels(args.diffusion_ckpt)
    if args.sampling_method is None:
        args.sampling_method = str(metadata_config.get("sampling_method", "ddpm"))
    if args.norm_stats_mat is None:
        args.norm_stats_mat = str(metadata_config.get("data_path", "")) or None
    if args.cache_postprocess is None:
        args.cache_postprocess = str(metadata_config.get("cache_postprocess", "none"))
    if args.interp_weight is None:
        args.interp_weight = float(metadata_config.get("interp_weight", 0.0))
    if args.output_input_blend is None:
        args.output_input_blend = float(metadata_config.get("output_input_blend", 0.0))

    generate_pidr_maps_batch(
        input_path=args.input,
        output_path=args.output,
        heatmap_dir=args.heatmap_dir,
        metrics_path=args.metrics,
        diffusion_ckpt=args.diffusion_ckpt,
        refiner_ckpt=args.refiner_ckpt,
        mask_ratio=args.mask_ratio,
        mask_strategy=args.mask_strategy,
        mask_seed=args.mask_seed,
        device=args.device,
        diffusion_base_channels=args.diffusion_base_channels,
        sample_stride=args.sample_stride,
        sampling_method=args.sampling_method,
        seed=args.seed,
        batch_size=max(1, int(args.batch_size)),
        num_samples=args.num_samples,
        max_plots=args.max_plots,
        plot_channel=args.plot_channel,
        enforce_observed=not args.no_enforce_observed,
        norm_stats_mat=args.norm_stats_mat,
        cache_postprocess=args.cache_postprocess,
        interp_weight=args.interp_weight,
        output_input_blend=args.output_input_blend,
    )


if __name__ == "__main__":
    main()
