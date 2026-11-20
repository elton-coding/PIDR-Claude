"""Compare PIDR reconstruction quality under different mask ratios.

Outputs:
- detailed_metrics.csv
- summary_metrics.csv
- per-sample comparison heatmaps in PDF (vector format by default)
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from pidr.models.pinn_refiner import build_refiner_from_state_dict
from pidr.config import Config
from pidr.data import CSIGridDataset
from pidr.diffusion import DiffusionManager
from pidr.models.diffusion_unet import CSIDiffusionPhysicsModel


PIDR_NAME = "PIDR-Net"


def resolve_checkpoint(candidates: Sequence[str], name: str) -> str:
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"{name} checkpoint not found. Tried: {list(candidates)}")


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


def plot_mask_ratio_heatmap(
    sample_id: int,
    gt_map: np.ndarray,
    row_payloads: List[Dict[str, object]],
    save_path: str,
) -> None:
    # Columns: GT, Sparse, PIDR-Net
    fig, axes = plt.subplots(len(row_payloads), 3, figsize=(14, 4.8 * len(row_payloads)))
    if len(row_payloads) == 1:
        axes = np.expand_dims(axes, axis=0)

    vmin, vmax = float(np.nanmin(gt_map)), float(np.nanmax(gt_map))

    for row_idx, payload in enumerate(row_payloads):
        plots = [
            (gt_map, "Ground Truth"),
            (
                payload["sparse"],
                f"{payload['label']} Sparse\nObserved: {payload['observed_ratio'] * 100:.0f}%",
            ),
            (
                payload["pidr"],
                f"{PIDR_NAME}\nMSE: {payload['mse']:.2f}\nNMSE: {payload['nmse_db']:.2f} dB",
            ),
        ]

        for col_idx, (plot_data, title) in enumerate(plots):
            draw_panel(axes[row_idx, col_idx], plot_data, title, vmin, vmax)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def validate_experiments(experiments: Sequence[Dict[str, object]]) -> None:
    if not experiments:
        raise ValueError("Config.BENCHMARK.PIDR_MASK_RATIOS.EXPERIMENTS must not be empty.")

    seen_names = set()
    for experiment in experiments:
        name = str(experiment["name"])
        if name in seen_names:
            raise ValueError(f"Duplicate experiment name: {name}")
        seen_names.add(name)

        mask_ratio = float(experiment["mask_ratio"])
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio} for experiment {name}")


def load_pidr_experiments(
    *,
    device: str,
    experiments: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    loaded_experiments: List[Dict[str, object]] = []

    for experiment in experiments:
        name = str(experiment["name"])
        label = str(experiment["label"])
        mask_ratio = float(experiment["mask_ratio"])

        diffusion_ckpt = resolve_checkpoint(
            experiment["pidr_diffusion_checkpoint_candidates"],
            f"{name} PIDR-Diffusion",
        )
        refiner_ckpt = resolve_checkpoint(
            experiment["pidr_refiner_checkpoint_candidates"],
            f"{name} PIDR-Refiner",
        )

        diffusion_model = CSIDiffusionPhysicsModel().to(device).eval()
        diffusion_model.load_state_dict(torch.load(diffusion_ckpt, map_location=device))

        refiner_model = build_refiner_from_state_dict(torch.load(refiner_ckpt, map_location="cpu")).to(device).eval()

        loaded_experiments.append(
            {
                "name": name,
                "label": label,
                "mask_ratio": mask_ratio,
                "diffusion_checkpoint": diffusion_ckpt,
                "refiner_checkpoint": refiner_ckpt,
                "diffusion_model": diffusion_model,
                "refiner_model": refiner_model,
            }
        )

    return loaded_experiments


def run_pidr_mask_ratio_benchmark() -> None:
    benchmark_cfg = Config.BENCHMARK.PIDR_MASK_RATIOS
    validate_experiments(benchmark_cfg.EXPERIMENTS)

    device = Config.DEVICE
    os.makedirs(benchmark_cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(benchmark_cfg.OUTPUT_DIR, "heatmaps"), exist_ok=True)

    dataset = CSIGridDataset(
        Config.DATA_PATH,
        mode="test",
        split_ratio=Config.DATA_SPLIT_RATIO,
        mask_seed=Config.EVAL_MASK_SEED,
    )
    diffusion_mgr = DiffusionManager(device, dataset.global_min, dataset.global_max)
    experiments = load_pidr_experiments(device=device, experiments=benchmark_cfg.EXPERIMENTS)

    all_records: List[Dict[str, object]] = []
    num_eval = min(max(0, int(benchmark_cfg.NUM_EVAL_SAMPLES)), len(dataset))
    if num_eval == 0:
        raise ValueError("NUM_EVAL_SAMPLES must be > 0.")

    for dataset_idx in tqdm(range(num_eval), desc=f"PIDR mask ratio benchmark ({num_eval} samples)"):
        sample_idx = int(dataset.indices[dataset_idx])
        gt_norm = dataset.gt_data[sample_idx].copy()
        valid_mask = dataset.valid_masks[sample_idx, 0].copy()
        gt_phys = dataset.gain_db(gt_norm[0])
        row_payloads: List[Dict[str, object]] = []

        for ratio_offset, experiment in enumerate(experiments):
            mask_ratio = float(experiment["mask_ratio"])
            observed_ratio = 1.0 - mask_ratio
            mask_np = dataset.generate_mask(
                valid_mask,
                strategy=benchmark_cfg.MASK_STRATEGY,
                sample_idx=sample_idx + ratio_offset * 10000,
                observed_ratio=observed_ratio,
            )
            sparse_norm = gt_norm * mask_np[None, :, :]

            sparse_tensor = torch.from_numpy(sparse_norm[None]).float().to(device)
            mask_tensor = torch.from_numpy(mask_np[None, None]).float().to(device)

            with torch.no_grad():
                pidr_diff_out = sample_physics_guided(
                    diffusion_mgr,
                    experiment["diffusion_model"],
                    sparse_tensor,
                    mask_tensor,
                    stride=benchmark_cfg.SAMPLING_STRIDE,
                )
                pidr_out = experiment["refiner_model"](pidr_diff_out, mask_tensor)
                if benchmark_cfg.ENFORCE_OBSERVED:
                    pidr_out = pidr_out * (1 - mask_tensor) + sparse_tensor * mask_tensor

            sparse_phys = dataset.gain_db(sparse_norm[0])
            sparse_display = sparse_phys.copy()
            sparse_display[mask_np < 0.5] = np.nan

            pidr_phys = dataset.gain_db(pidr_out[0, 0].cpu().numpy())
            mse_val, nmse_db, psnr_val, ssim_val = calc_metrics(gt_phys, pidr_phys)
            observed_count = int(mask_np.sum())

            all_records.append(
                {
                    "sample_id": sample_idx,
                    "experiment": experiment["name"],
                    "label": experiment["label"],
                    "mask_ratio": mask_ratio,
                    "observed_ratio": observed_ratio,
                    "observed_count": observed_count,
                    "mask_strategy": benchmark_cfg.MASK_STRATEGY,
                    "pidr_diffusion_checkpoint": experiment["diffusion_checkpoint"],
                    "pidr_refiner_checkpoint": experiment["refiner_checkpoint"],
                    "mse": mse_val,
                    "nmse_db": nmse_db,
                    "psnr_db": psnr_val,
                    "ssim": ssim_val,
                }
            )

            row_payloads.append(
                {
                    "label": experiment["label"],
                    "observed_ratio": observed_ratio,
                    "sparse": sparse_display,
                    "pidr": pidr_phys,
                    "mse": mse_val,
                    "nmse_db": nmse_db,
                }
            )

        heatmap_path = os.path.join(
            benchmark_cfg.OUTPUT_DIR,
            "heatmaps",
            f"sample_{sample_idx}.{benchmark_cfg.PLOT_FORMAT}",
        )
        plot_mask_ratio_heatmap(sample_idx, gt_phys, row_payloads, heatmap_path)

    df = pd.DataFrame(all_records)
    summary = (
        df.groupby(
            ["experiment", "label", "mask_ratio", "observed_ratio", "mask_strategy"],
            as_index=False,
        )[["observed_count", "mse", "nmse_db", "psnr_db", "ssim"]]
        .mean()
        .sort_values(["mask_ratio", "experiment"])
    )

    df.to_csv(os.path.join(benchmark_cfg.OUTPUT_DIR, "detailed_metrics.csv"), index=False)
    summary.to_csv(os.path.join(benchmark_cfg.OUTPUT_DIR, "summary_metrics.csv"), index=False)

    print("\nPIDR mask ratio comparison summary:")
    print(summary.to_string(index=False))
    print(f"\nDetailed results saved to {benchmark_cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse

    cfg = Config.BENCHMARK.PIDR_MASK_RATIOS
    parser = argparse.ArgumentParser(description="PIDR missing-ratio benchmark (defaults: Config.BENCHMARK.PIDR_MASK_RATIOS)")
    parser.add_argument("--refiner-root", default=None,
                        help="directory holding rho<ratio>/refiner.pth (e.g. runs/sr/mask_ratio_fr); default: curated checkpoints")
    parser.add_argument("--output-dir", default=cfg.OUTPUT_DIR)
    parser.add_argument("--num-eval-samples", type=int, default=cfg.NUM_EVAL_SAMPLES)
    args = parser.parse_args()
    cfg.OUTPUT_DIR = args.output_dir
    cfg.NUM_EVAL_SAMPLES = args.num_eval_samples
    if args.refiner_root:
        for exp in cfg.EXPERIMENTS:
            exp["pidr_refiner_checkpoint_candidates"] = [
                os.path.join(args.refiner_root, f"rho{float(exp['mask_ratio']):.2f}", "refiner.pth")
            ]
    run_pidr_mask_ratio_benchmark()
