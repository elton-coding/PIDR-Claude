"""Evaluate partial-supervision PIDR against complete CSI maps.

The inference input follows the partial-supervision setting:
- input points are sparse observations provided to the model
- supervision points are hidden from the model during inference

Metrics are computed against the full ground-truth channel map, following the
project's existing benchmark convention on the first channel (power map).
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from pidr.models.pinn_refiner import build_refiner_from_state_dict
from pidr.config import Config
from pidr.data import CSIGridDataset
from pidr.diffusion import DiffusionManager
from pidr.models.diffusion_unet import CSIDiffusionPhysicsModel
from scripts.train.train_pidr_semisupervised import PartialSupervisionCSIGridDataset, sample_physics_guided


PIDR_NAME = "PIDR-Net"


def resolve_checkpoint(candidates: Sequence[str], name: str) -> str:
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"{name} checkpoint not found. Tried: {list(candidates)}")


def calc_metrics(gt_map: np.ndarray, pred_map: np.ndarray) -> Tuple[float, float, float]:
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

    ssim_val = float(ssim(gt_map, pred_map, data_range=data_range, win_size=3))
    return mse_val, nmse_db, ssim_val


def draw_panel(
    ax: plt.Axes,
    data: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    *,
    cmap: str = "jet",
) -> None:
    data = np.asarray(data, dtype=np.float32)
    data = np.ma.masked_invalid(data)
    height, width = data.shape
    mesh = ax.pcolormesh(
        np.arange(width + 1),
        np.arange(height + 1),
        data,
        cmap=cmap,
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


def plot_comparison(
    *,
    save_path: str,
    gt_map: np.ndarray,
    sparse_map: np.ndarray,
    pred_map: np.ndarray,
    error_map: np.ndarray,
    mse_val: float,
    nmse_db: float,
    ssim_val: float,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    vmin, vmax = float(np.nanmin(gt_map)), float(np.nanmax(gt_map))
    err_max = float(np.nanmax(error_map))
    if err_max <= 0:
        err_max = 1.0

    draw_panel(axes[0], gt_map, "Ground Truth", vmin, vmax)
    draw_panel(axes[1], sparse_map, "Sparse Input", vmin, vmax)
    draw_panel(
        axes[2],
        pred_map,
        f"{PIDR_NAME}\nMSE: {mse_val:.2f}\nNMSE: {nmse_db:.2f} dB\nSSIM: {ssim_val:.3f}",
        vmin,
        vmax,
    )
    draw_panel(axes[3], error_map, "Absolute Error", 0.0, err_max, cmap="magma")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def build_eval_dataset() -> PartialSupervisionCSIGridDataset:
    benchmark_cfg = Config.BENCHMARK.PIDR_PARTIAL_SUPERVISION
    base_dataset = CSIGridDataset(
        Config.DATA_PATH,
        mode="test",
        split_ratio=Config.DATA_SPLIT_RATIO,
        mask_seed=benchmark_cfg.EVAL_SPLIT_SEED,
    )
    return PartialSupervisionCSIGridDataset(
        base_dataset,
        total_available_ratio=benchmark_cfg.TOTAL_AVAILABLE_RATIO,
        input_fraction_within_available=benchmark_cfg.INPUT_FRACTION_WITHIN_AVAILABLE,
        split_seed=benchmark_cfg.EVAL_SPLIT_SEED,
        mask_strategy=benchmark_cfg.MASK_STRATEGY,
    )


def run_benchmark() -> None:
    benchmark_cfg = Config.BENCHMARK.PIDR_PARTIAL_SUPERVISION
    device = Config.DEVICE

    os.makedirs(benchmark_cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(benchmark_cfg.OUTPUT_DIR, "heatmaps"), exist_ok=True)

    dataset = build_eval_dataset()
    base_dataset = dataset.base_dataset

    diffusion_ckpt = resolve_checkpoint(
        benchmark_cfg.DIFFUSION_CHECKPOINT_CANDIDATES,
        "Partial PIDR-Diffusion",
    )
    refiner_ckpt = resolve_checkpoint(
        benchmark_cfg.REFINER_CHECKPOINT_CANDIDATES,
        "Partial PIDR-Refiner",
    )

    diffusion_mgr = DiffusionManager(device, dataset.global_min, dataset.global_max)

    diffusion_model = CSIDiffusionPhysicsModel().to(device).eval()
    diffusion_model.load_state_dict(torch.load(diffusion_ckpt, map_location=device))

    refiner_model = build_refiner_from_state_dict(torch.load(refiner_ckpt, map_location="cpu")).to(device).eval()

    num_eval = min(max(0, int(benchmark_cfg.NUM_EVAL_SAMPLES)), len(dataset))
    if num_eval == 0:
        raise ValueError("NUM_EVAL_SAMPLES must be > 0.")

    records: List[Dict[str, object]] = []

    for idx in tqdm(range(num_eval), desc=f"Partial PIDR benchmark ({num_eval} samples)"):
        sample = dataset[idx]
        sample_idx = int(base_dataset.indices[idx])
        gt_norm = base_dataset.gt_data[sample_idx].copy()

        input_csi = sample["input_csi"]
        input_mask = sample["input_mask"]
        supervision_mask = sample["supervision_mask"]
        available_mask = sample["available_mask"]

        gt_phys = base_dataset.gain_db(gt_norm[0])
        sparse_phys = base_dataset.gain_db(input_csi[0].numpy())
        sparse_display = sparse_phys.copy()
        sparse_display[input_mask[0].numpy() < 0.5] = np.nan

        with torch.no_grad():
            input_csi_batch = input_csi.unsqueeze(0).to(device)
            input_mask_batch = input_mask.unsqueeze(0).to(device)

            diff_out = sample_physics_guided(
                diffusion_mgr,
                diffusion_model,
                input_csi_batch,
                input_mask_batch,
                stride=benchmark_cfg.SAMPLING_STRIDE,
            )
            pred = refiner_model(diff_out, input_mask_batch)
            if benchmark_cfg.ENFORCE_OBSERVED:
                pred = pred * (1 - input_mask_batch) + input_csi_batch * input_mask_batch

        pred_phys = base_dataset.gain_db(pred[0, 0].cpu().numpy())
        error_map = np.abs(pred_phys - gt_phys)
        mse_val, nmse_db, ssim_val = calc_metrics(gt_phys, pred_phys)

        input_count = int(input_mask.sum().item())
        supervision_count = int(supervision_mask.sum().item())
        available_count = int(available_mask.sum().item())

        records.append(
            {
                "sample_id": sample_idx,
                "input_count": input_count,
                "supervision_count": supervision_count,
                "available_count": available_count,
                "total_available_ratio": float(benchmark_cfg.TOTAL_AVAILABLE_RATIO),
                "input_fraction_within_available": float(benchmark_cfg.INPUT_FRACTION_WITHIN_AVAILABLE),
                "mask_strategy": benchmark_cfg.MASK_STRATEGY,
                "diffusion_checkpoint": diffusion_ckpt,
                "refiner_checkpoint": refiner_ckpt,
                "mse": mse_val,
                "nmse_db": nmse_db,
                "ssim": ssim_val,
            }
        )

        if idx < benchmark_cfg.MAX_PLOTS:
            save_path = os.path.join(
                benchmark_cfg.OUTPUT_DIR,
                "heatmaps",
                f"sample_{sample_idx}.{benchmark_cfg.PLOT_FORMAT}",
            )
            plot_comparison(
                save_path=save_path,
                gt_map=gt_phys,
                sparse_map=sparse_display,
                pred_map=pred_phys,
                error_map=error_map,
                mse_val=mse_val,
                nmse_db=nmse_db,
                ssim_val=ssim_val,
            )

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(benchmark_cfg.OUTPUT_DIR, "detailed_metrics.csv"), index=False)

    summary = pd.DataFrame(
        [
            {
                "Method": PIDR_NAME,
                "Avg_Input_Count": float(df["input_count"].mean()),
                "Avg_Supervision_Count": float(df["supervision_count"].mean()),
                "Avg_Available_Count": float(df["available_count"].mean()),
                "Avg_MSE": float(df["mse"].mean()),
                "Avg_NMSE_dB": float(df["nmse_db"].mean()),
                "Avg_SSIM": float(df["ssim"].mean()),
            }
        ]
    )
    summary.to_csv(os.path.join(benchmark_cfg.OUTPUT_DIR, "summary_metrics.csv"), index=False)

    print("\nPartial-supervision PIDR benchmark summary:")
    print(summary.to_string(index=False))
    print(f"\nDetailed results saved to {benchmark_cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse

    cfg = Config.BENCHMARK.PIDR_PARTIAL_SUPERVISION
    parser = argparse.ArgumentParser(description="Semi-supervised PIDR benchmark (defaults: Config.BENCHMARK.PIDR_PARTIAL_SUPERVISION)")
    parser.add_argument("--alpha", type=float, default=None, help="offline available ratio (default: config)")
    parser.add_argument("--diffusion-ckpt", default=None, help="default: checkpoints/<ds>/semisup/alpha<a>/diffusion.pth")
    parser.add_argument("--refiner-ckpt", default=None, help="default: checkpoints/<ds>/semisup/alpha<a>/refiner.pth")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.alpha is not None:
        cfg.TOTAL_AVAILABLE_RATIO = args.alpha
        tag = f"alpha{args.alpha:g}"
        root = os.path.join(Config.PATHS.CHECKPOINT_DIR, Config.DATASET, "semisup", tag)
        cfg.DIFFUSION_CHECKPOINT_CANDIDATES = [os.path.join(root, "diffusion.pth")]
        cfg.REFINER_CHECKPOINT_CANDIDATES = [os.path.join(root, "refiner.pth")]
    if args.diffusion_ckpt:
        cfg.DIFFUSION_CHECKPOINT_CANDIDATES = [args.diffusion_ckpt]
    if args.refiner_ckpt:
        cfg.REFINER_CHECKPOINT_CANDIDATES = [args.refiner_ckpt]
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    run_benchmark()
