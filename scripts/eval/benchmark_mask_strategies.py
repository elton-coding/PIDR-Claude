"""Compare mask strategies across interpolation baselines, PINN, Diffusion and PIDR-Net.

Outputs:
- detailed_metrics.csv
- summary_metrics.csv
- per-sample comparison heatmaps in PDF (vector format)
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import griddata
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from pidr.models.pinn_refiner import PINNCSIReconstructor, build_refiner_from_state_dict
from pidr.config import Config
from pidr.data import CSIGridDataset
from pidr.diffusion import DiffusionManager
from pidr.models.diffusion_unet import CSIDiffusionPhysicsModel


MASK_STRATEGIES = [
    ("random", "Random"),
    ("uniform", "Uniform"),
    ("extreme_uneven", "Extreme Uneven"),
]

INTERP_METHODS = [
    ("Nearest", "nearest"),
    ("Linear", "linear"),
    ("Cubic", "cubic"),
]

PIDR_NAME = "PIDR-Net"


def resolve_checkpoint(candidates: List[str], name: str) -> str:
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"{name} checkpoint not found. Tried: {candidates}")


def calc_metrics(gt_map: np.ndarray, pred_map: np.ndarray) -> Tuple[float, float, float, float]:
    pred_map = np.asarray(pred_map, dtype=np.float32)
    gt_map = np.asarray(gt_map, dtype=np.float32)

    if np.isnan(pred_map).any():
        pred_map = np.nan_to_num(pred_map, nan=float(np.nanmin(gt_map)))

    mse_val = float(np.mean((gt_map - pred_map) ** 2))
    norm = float(np.mean(gt_map**2))
    nmse_db = float(10.0 * np.log10(mse_val / (norm + 1e-10)))

    data_range = float(np.nanmax(gt_map) - np.nanmin(gt_map))
    if data_range <= 0:
        data_range = 1.0

    psnr_val = float(psnr(gt_map, pred_map, data_range=data_range))
    ssim_val = float(ssim(gt_map, pred_map, data_range=data_range, win_size=3))
    return mse_val, nmse_db, psnr_val, ssim_val


def classical_interp(sparse_map: np.ndarray, mask: np.ndarray, method: str) -> np.ndarray:
    height, width = sparse_map.shape
    points_y, points_x = np.where(mask > 0.5)
    values = sparse_map[points_y, points_x]
    if len(values) == 0:
        return np.zeros_like(sparse_map, dtype=np.float32)

    grid_y, grid_x = np.mgrid[0:height, 0:width]
    result = griddata(
        (points_y, points_x),
        values,
        (grid_y, grid_x),
        method=method,
        fill_value=float(np.min(values)),
    )
    return np.nan_to_num(result).astype(np.float32)


def sample_physics_guided(
    diffusion: DiffusionManager,
    model: torch.nn.Module,
    sparse_csi: torch.Tensor,
    mask: torch.Tensor,
    *,
    stride: int = 1,
) -> torch.Tensor:
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


def draw_panel(
    ax: plt.Axes,
    data: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    data = np.asarray(data, dtype=np.float32)
    data = np.ma.masked_invalid(data)
    height, width = data.shape
    mesh = ax.pcolormesh(
        np.arange(width + 1),
        np.arange(height + 1),
        data,
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        shading="flat",
    )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)


def plot_mask_strategy_heatmap(
    sample_id: int,
    gt_map: np.ndarray,
    row_payloads: List[Dict[str, np.ndarray]],
    save_path: str,
) -> None:
    # Columns: GT, Sparse, Nearest, Linear, Cubic, PINN, Diffusion, PIDR-Net
    fig, axes = plt.subplots(len(row_payloads), 8, figsize=(34, 4.8 * len(row_payloads)))
    if len(row_payloads) == 1:
        axes = np.expand_dims(axes, axis=0)

    vmin, vmax = float(np.nanmin(gt_map)), float(np.nanmax(gt_map))

    for row_idx, payload in enumerate(row_payloads):
        plots = [
            (gt_map, "Ground Truth"),
            (payload["sparse"], f"{payload['label']} Sparse"),
            (payload["nearest"], f"Nearest\nMSE: {payload['nearest_mse']:.2f}"),
            (payload["linear"], f"Linear\nMSE: {payload['linear_mse']:.2f}"),
            (payload["cubic"], f"Cubic\nMSE: {payload['cubic_mse']:.2f}"),
            (payload["pinn"], f"PINN\nMSE: {payload['pinn_mse']:.2f}"),
            (payload["diffusion"], f"Diffusion\nMSE: {payload['diffusion_mse']:.2f}"),
            (payload["pidr"], f"{PIDR_NAME}\nMSE: {payload['pidr_mse']:.2f}"),
        ]

        for col_idx, (plot_data, title) in enumerate(plots):
            draw_panel(axes[row_idx, col_idx], plot_data, title, vmin, vmax)

    fig.suptitle(f"Mask Strategy Comparison (Sample {sample_id})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def run_mask_strategy_benchmark(
    *,
    diffusion_path: str,
    pinn_path: str,
    pidr_diffusion_path: str,
    pidr_refiner_path: str,
    output_dir: str = Config.BENCHMARK.MASK_STRATEGIES.OUTPUT_DIR,
    num_eval_samples: int = Config.BENCHMARK.MASK_STRATEGIES.NUM_EVAL_SAMPLES,
) -> None:
    device = Config.DEVICE
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "heatmaps"), exist_ok=True)

    eval_mask_seed = Config.EVAL_MASK_SEED
    sampling_stride = Config.BENCHMARK.MASK_STRATEGIES.SAMPLING_STRIDE
    enforce_observed = Config.BENCHMARK.MASK_STRATEGIES.ENFORCE_OBSERVED

    dataset = CSIGridDataset(
        Config.DATA_PATH,
        mode="test",
        split_ratio=Config.DATA_SPLIT_RATIO,
        mask_seed=eval_mask_seed,
    )

    diffusion_model = CSIDiffusionPhysicsModel().to(device).eval()
    diffusion_model.load_state_dict(torch.load(diffusion_path, map_location=device))

    pinn_model = PINNCSIReconstructor().to(device).eval()
    pinn_model.load_state_dict(torch.load(pinn_path, map_location=device))

    pidr_diffusion_model = CSIDiffusionPhysicsModel().to(device).eval()
    pidr_diffusion_model.load_state_dict(torch.load(pidr_diffusion_path, map_location=device))

    pidr_refiner_model = build_refiner_from_state_dict(torch.load(pidr_refiner_path, map_location="cpu")).to(device).eval()

    diffusion_mgr = DiffusionManager(device, dataset.global_min, dataset.global_max)

    all_records: List[Dict[str, float]] = []
    observed_ratio = 1.0 - Config.MASK_RATIO
    num_eval = min(max(0, int(num_eval_samples)), len(dataset))
    if num_eval == 0:
        raise ValueError("num_eval_samples must be > 0.")

    for dataset_idx in tqdm(range(num_eval), desc=f"Mask strategy benchmark ({num_eval} samples)"):
        sample_idx = int(dataset.indices[dataset_idx])
        gt_norm = dataset.gt_data[sample_idx].copy()
        valid_mask = dataset.valid_masks[sample_idx, 0].copy()
        gt_phys = dataset.inverse_normalize_channel(gt_norm[0], 0)
        row_payloads = []

        for strategy_offset, (strategy_key, strategy_label) in enumerate(MASK_STRATEGIES):
            mask_np = dataset.generate_mask(
                valid_mask,
                strategy=strategy_key,
                sample_idx=sample_idx + strategy_offset * 10000,
                observed_ratio=observed_ratio,
            )
            sparse_norm = gt_norm * mask_np[None, :, :]

            sparse_tensor = torch.from_numpy(sparse_norm[None]).float().to(device)
            mask_tensor = torch.from_numpy(mask_np[None, None]).float().to(device)

            with torch.no_grad():
                pinn_out = pinn_model(sparse_tensor, mask_tensor)
                if enforce_observed:
                    pinn_out = pinn_out * (1 - mask_tensor) + sparse_tensor * mask_tensor

                diffusion_out = sample_physics_guided(
                    diffusion_mgr,
                    diffusion_model,
                    sparse_tensor,
                    mask_tensor,
                    stride=sampling_stride,
                )

                pidr_diff_out = sample_physics_guided(
                    diffusion_mgr,
                    pidr_diffusion_model,
                    sparse_tensor,
                    mask_tensor,
                    stride=sampling_stride,
                )
                pidr_out = pidr_refiner_model(pidr_diff_out, mask_tensor)
                if enforce_observed:
                    pidr_out = pidr_out * (1 - mask_tensor) + sparse_tensor * mask_tensor

            sparse_phys = dataset.inverse_normalize_channel(sparse_norm[0], 0)
            sparse_display = sparse_phys.copy()
            sparse_display[mask_np < 0.5] = np.nan

            interp_maps: Dict[str, np.ndarray] = {}
            for display_name, method in INTERP_METHODS:
                interp_maps[display_name] = classical_interp(sparse_phys, mask_np, method=method)

            pinn_phys = dataset.inverse_normalize_channel(pinn_out[0, 0].cpu().numpy(), 0)
            diffusion_phys = dataset.inverse_normalize_channel(diffusion_out[0, 0].cpu().numpy(), 0)
            pidr_phys = dataset.inverse_normalize_channel(pidr_out[0, 0].cpu().numpy(), 0)

            method_maps = [
                ("Nearest", interp_maps["Nearest"]),
                ("Linear", interp_maps["Linear"]),
                ("Cubic", interp_maps["Cubic"]),
                ("PINN", pinn_phys),
                ("Diffusion", diffusion_phys),
                (PIDR_NAME, pidr_phys),
            ]

            observed_count = int(mask_np.sum())
            for method_name, pred_map in method_maps:
                mse_val, nmse_db, psnr_val, ssim_val = calc_metrics(gt_phys, pred_map)
                all_records.append(
                    {
                        "sample_id": sample_idx,
                        "strategy": strategy_key,
                        "method": method_name,
                        "observed_count": observed_count,
                        "mse": mse_val,
                        "nmse_db": nmse_db,
                        "psnr_db": psnr_val,
                        "ssim": ssim_val,
                    }
                )

            row_payloads.append(
                {
                    "label": strategy_label,
                    "sparse": sparse_display,
                    "nearest": interp_maps["Nearest"],
                    "linear": interp_maps["Linear"],
                    "cubic": interp_maps["Cubic"],
                    "pinn": pinn_phys,
                    "diffusion": diffusion_phys,
                    "pidr": pidr_phys,
                    "nearest_mse": calc_metrics(gt_phys, interp_maps["Nearest"])[0],
                    "linear_mse": calc_metrics(gt_phys, interp_maps["Linear"])[0],
                    "cubic_mse": calc_metrics(gt_phys, interp_maps["Cubic"])[0],
                    "pinn_mse": calc_metrics(gt_phys, pinn_phys)[0],
                    "diffusion_mse": calc_metrics(gt_phys, diffusion_phys)[0],
                    "pidr_mse": calc_metrics(gt_phys, pidr_phys)[0],
                }
            )

        heatmap_path = os.path.join(output_dir, "heatmaps", f"sample_{sample_idx}.pdf")
        plot_mask_strategy_heatmap(sample_idx, gt_phys, row_payloads, heatmap_path)

    df = pd.DataFrame(all_records)
    summary = (
        df.groupby(["strategy", "method"], as_index=False)[["observed_count", "mse", "nmse_db", "psnr_db", "ssim"]]
        .mean()
        .sort_values(["strategy", "method"])
    )

    df.to_csv(os.path.join(output_dir, "detailed_metrics.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)

    print("\nMask strategy comparison summary:")
    print(summary.to_string(index=False))
    print(f"\nDetailed results saved to {output_dir}")


if __name__ == "__main__":
    run_mask_strategy_benchmark(
        diffusion_path=resolve_checkpoint(
            Config.BENCHMARK.MASK_STRATEGIES.DIFFUSION_CHECKPOINT_CANDIDATES,
            "Diffusion",
        ),
        pinn_path=resolve_checkpoint(
            Config.BENCHMARK.MASK_STRATEGIES.PINN_CHECKPOINT_CANDIDATES,
            "PINN",
        ),
        pidr_diffusion_path=resolve_checkpoint(
            Config.BENCHMARK.MASK_STRATEGIES.PIDR_DIFFUSION_CHECKPOINT_CANDIDATES,
            "PIDR-Diffusion",
        ),
        pidr_refiner_path=resolve_checkpoint(
            Config.BENCHMARK.MASK_STRATEGIES.PIDR_REFINER_CHECKPOINT_CANDIDATES,
            "PIDR-Refiner",
        ),
    )
