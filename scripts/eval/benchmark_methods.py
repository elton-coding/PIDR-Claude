"""
Benchmark comparison: Diffusion vs PINN vs PIDR.

Outputs:
- per-sample metrics CSV
- summary metrics CSV
- vector visual comparison plots (PDF)

Metrics:
- MSE
- NMSE (dB): 10*log10( MSE / mean(gt^2) )
- PSNR
- SSIM
"""

from __future__ import annotations

import argparse
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

# Project method display name used in figure subtitles.
PROPOSED_METHOD_NAME = "PIDR-Net"


def resolve_checkpoint(candidates: List[str], name: str) -> str:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"{name} checkpoint not found. Tried: {candidates}")


def load_plain_or_wrapped_state_dict(path: str, device: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


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


def calc_metrics(gt_map: np.ndarray, pred_map: np.ndarray) -> Tuple[float, float, float, float]:
    pred_map = np.asarray(pred_map, dtype=np.float32)
    gt_map = np.asarray(gt_map, dtype=np.float32)

    if np.isnan(pred_map).any():
        pred_map = np.nan_to_num(pred_map, nan=float(np.min(gt_map)))

    mse_val = float(np.mean((gt_map - pred_map) ** 2))
    norm = float(np.mean(gt_map**2))
    nmse_db = float(10.0 * np.log10(mse_val / (norm + 1e-10)))

    data_range = float(np.max(gt_map) - np.min(gt_map))
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
    """
    Physics-guided ancestral sampling (same core logic as DiffusionManager.sample),
    but without the internal tqdm progress bar.
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


def plot_comparison(
    *,
    save_path: str,
    gt_map: np.ndarray,
    sparse_map: np.ndarray,
    diff_map: np.ndarray,
    pinn_map: np.ndarray,
    pidr_map: np.ndarray,
    titles: Dict[str, str],
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    vmin, vmax = float(np.nanmin(gt_map)), float(np.nanmax(gt_map))

    items = [
        (gt_map, titles["gt"]),
        (sparse_map, titles["sparse"]),
        (diff_map, titles["diffusion"]),
        (pinn_map, titles["pinn"]),
        (pidr_map, titles["pidr"]),
    ]

    for ax, (data, title) in zip(axes, items):
        data = np.asarray(data, dtype=np.float32)
        data = np.ma.masked_invalid(data)
        h, w = data.shape
        mesh = ax.pcolormesh(
            np.arange(w + 1),
            np.arange(h + 1),
            data,
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title(title, fontsize=11, pad=10)
        plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def apply_cli_overrides(args: argparse.Namespace) -> None:
    if args.data_path:
        Config.DATA_PATH = args.data_path
    if args.img_height is not None:
        Config.IMG_HEIGHT = int(args.img_height)
        Config.DATA.IMG_HEIGHT = int(args.img_height)
    if args.img_width is not None:
        Config.IMG_WIDTH = int(args.img_width)
        Config.DATA.IMG_WIDTH = int(args.img_width)
    if args.mask_ratio is not None:
        Config.MASK_RATIO = float(args.mask_ratio)
        Config.DATA.MASK_RATIO = float(args.mask_ratio)
    if args.mask_seed is not None:
        Config.EVAL_MASK_SEED = int(args.mask_seed)
        Config.DATA.EVAL_MASK_SEED = int(args.mask_seed)
    if args.diffusion_base_channels is not None:
        Config.BASE_CHANNELS = int(args.diffusion_base_channels)
        Config.MODEL.BASE_CHANNELS = int(args.diffusion_base_channels)

    benchmark_cfg = Config.BENCHMARK.DIFFUSION_PINN_PIDR
    if args.output_dir:
        benchmark_cfg.OUTPUT_DIR = args.output_dir
    if args.diffusion_ckpt:
        benchmark_cfg.DIFFUSION_CHECKPOINT_CANDIDATES = [args.diffusion_ckpt]
    if args.pinn_ckpt:
        benchmark_cfg.PINN_CHECKPOINT_CANDIDATES = [args.pinn_ckpt]
    if args.pidr_diffusion_ckpt:
        benchmark_cfg.PIDR_DIFFUSION_CHECKPOINT_CANDIDATES = [args.pidr_diffusion_ckpt]
    if args.pidr_refiner_ckpt:
        benchmark_cfg.PIDR_REFINER_CHECKPOINT_CANDIDATES = [args.pidr_refiner_ckpt]
    if args.sample_stride is not None:
        benchmark_cfg.SAMPLING_STRIDE = int(args.sample_stride)
    if args.num_eval_samples is not None:
        benchmark_cfg.NUM_EVAL_SAMPLES = int(args.num_eval_samples)
    if args.max_plots is not None:
        benchmark_cfg.MAX_PLOTS = int(args.max_plots)
    if args.plot_format:
        benchmark_cfg.PLOT_FORMAT = args.plot_format
    benchmark_cfg.PIDR_SAMPLING_METHOD = args.sampling_method
    benchmark_cfg.PIDR_CACHE_POSTPROCESS = args.cache_postprocess
    benchmark_cfg.PIDR_INTERP_WEIGHT = float(args.interp_weight)
    benchmark_cfg.PIDR_OUTPUT_INPUT_BLEND = float(args.output_input_blend)


def run_benchmark() -> None:
    benchmark_cfg = Config.BENCHMARK.DIFFUSION_PINN_PIDR
    out_dir = benchmark_cfg.OUTPUT_DIR
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    # Deterministic masks for fair comparison.
    eval_mask_seed = Config.EVAL_MASK_SEED

    # If True, overwrite observed positions with sparse measurements for PINN outputs.
    # Diffusion already enforces this during sampling.
    enforce_observed = benchmark_cfg.ENFORCE_OBSERVED

    # Reduce this to speed up diffusion sampling (may degrade quality).
    sampling_stride = benchmark_cfg.SAMPLING_STRIDE
    pidr_sampling_method = getattr(benchmark_cfg, "PIDR_SAMPLING_METHOD", "ddpm")
    pidr_cache_postprocess = getattr(benchmark_cfg, "PIDR_CACHE_POSTPROCESS", "none")
    pidr_interp_weight = float(getattr(benchmark_cfg, "PIDR_INTERP_WEIGHT", 0.0))
    pidr_output_input_blend = float(getattr(benchmark_cfg, "PIDR_OUTPUT_INPUT_BLEND", 0.0))
    if pidr_sampling_method != "ddpm":
        raise ValueError("benchmark_methods currently supports --sampling-method ddpm.")
    if pidr_cache_postprocess not in {"none", "linear_interp_blend"}:
        raise ValueError(f"Unsupported cache postprocess: {pidr_cache_postprocess}")

    max_plots = benchmark_cfg.MAX_PLOTS
    num_eval_samples = benchmark_cfg.NUM_EVAL_SAMPLES
    plot_format = benchmark_cfg.PLOT_FORMAT

    diffusion_ckpt = resolve_checkpoint(
        benchmark_cfg.DIFFUSION_CHECKPOINT_CANDIDATES,
        "Diffusion",
    )

    pinn_ckpt = resolve_checkpoint(
        benchmark_cfg.PINN_CHECKPOINT_CANDIDATES,
        "PINN",
    )

    pidr_diff_ckpt = resolve_checkpoint(
        benchmark_cfg.PIDR_DIFFUSION_CHECKPOINT_CANDIDATES,
        "PIDR-Diffusion",
    )

    pidr_refiner_ckpt = resolve_checkpoint(
        benchmark_cfg.PIDR_REFINER_CHECKPOINT_CANDIDATES,
        "PIDR-Refiner",
    )

    device = Config.DEVICE
    # Evaluate on the held-out test split (the paper's protocol). Before 2026-11 this read the
    # training split, which is what the original main-table numbers were computed on.
    dataset = CSIGridDataset(
        Config.DATA_PATH,
        mode="test",
        split_ratio=Config.DATA_SPLIT_RATIO,
        mask_seed=eval_mask_seed,
    )

    diffusion_mgr = DiffusionManager(device, dataset.global_min, dataset.global_max)

    diff = CSIDiffusionPhysicsModel().to(device).eval()
    diff.load_state_dict(load_plain_or_wrapped_state_dict(diffusion_ckpt, device))

    pinn = PINNCSIReconstructor().to(device).eval()
    pinn.load_state_dict(load_plain_or_wrapped_state_dict(pinn_ckpt, device))

    pidr_diff = CSIDiffusionPhysicsModel().to(device).eval()
    pidr_diff.load_state_dict(load_plain_or_wrapped_state_dict(pidr_diff_ckpt, device))

    pidr_refiner = build_refiner_from_state_dict(load_plain_or_wrapped_state_dict(pidr_refiner_ckpt, device)).to(device).eval()

    print(
        "Benchmark settings: "
        f"data_path={Config.DATA_PATH}, mask_ratio={Config.MASK_RATIO}, mask_seed={eval_mask_seed}, "
        f"diffusion_base_channels={Config.BASE_CHANNELS}, sample_stride={sampling_stride}, "
        f"pidr_sampling_method={pidr_sampling_method}, pidr_cache_postprocess={pidr_cache_postprocess}, "
        f"pidr_interp_weight={pidr_interp_weight}, pidr_output_input_blend={pidr_output_input_blend}"
    )
    print(f"Diffusion ckpt: {diffusion_ckpt}")
    print(f"PINN ckpt: {pinn_ckpt}")
    print(f"PIDR diffusion ckpt: {pidr_diff_ckpt}")
    print(f"PIDR refiner ckpt: {pidr_refiner_ckpt}")

    records: List[Dict[str, float]] = []
    num_eval = min(num_eval_samples, len(dataset))

    for idx in tqdm(range(num_eval), desc="Benchmark"):
        sample = dataset[idx]
        gt = sample["gt_csi"]
        sparse = sample["sparse_csi"]
        mask = sample["mask"]

        gt_phys = dataset.inverse_normalize_channel(gt[0].numpy(), 0)
        sparse_phys = dataset.inverse_normalize_channel(sparse[0].numpy(), 0)
        mask_np = mask[0].numpy()

        sparse_display = sparse_phys.copy()
        sparse_display[mask_np == 0] = np.nan

        with torch.no_grad():
            s_in = sparse.unsqueeze(0).to(device)
            m_in = mask.unsqueeze(0).to(device)

            diff_out = sample_physics_guided(diffusion_mgr, diff, s_in, m_in, stride=sampling_stride)

            pinn_out = pinn(s_in, m_in)
            if enforce_observed:
                pinn_out = pinn_out * (1 - m_in) + s_in * m_in

            pidr_diff_out = sample_physics_guided(diffusion_mgr, pidr_diff, s_in, m_in, stride=sampling_stride)
            if pidr_cache_postprocess == "linear_interp_blend":
                interp = interpolate_sparse_batch(s_in, m_in).to(device)
                pidr_diff_out = (1.0 - pidr_interp_weight) * pidr_diff_out + pidr_interp_weight * interp
                pidr_diff_out = pidr_diff_out * (1 - m_in) + s_in * m_in
            pidr_out = pidr_refiner(pidr_diff_out, m_in)
            if pidr_output_input_blend > 0.0:
                pidr_out = (1.0 - pidr_output_input_blend) * pidr_out + pidr_output_input_blend * pidr_diff_out
            if enforce_observed:
                pidr_out = pidr_out * (1 - m_in) + s_in * m_in

        diff_phys = dataset.inverse_normalize_channel(diff_out[0, 0].cpu().numpy(), 0)
        pinn_phys = dataset.inverse_normalize_channel(pinn_out[0, 0].cpu().numpy(), 0)
        pidr_phys = dataset.inverse_normalize_channel(pidr_out[0, 0].cpu().numpy(), 0)

        diff_mse, diff_nmse, diff_psnr, diff_ssim = calc_metrics(gt_phys, diff_phys)
        pinn_mse, pinn_nmse, pinn_psnr, pinn_ssim = calc_metrics(gt_phys, pinn_phys)
        pidr_mse, pidr_nmse, pidr_psnr, pidr_ssim = calc_metrics(gt_phys, pidr_phys)

        records.append(
            {
                "Sample_ID": int(idx),
                "Diffusion_MSE": diff_mse,
                "Diffusion_NMSE_dB": diff_nmse,
                "Diffusion_PSNR_dB": diff_psnr,
                "Diffusion_SSIM": diff_ssim,
                "PINN_MSE": pinn_mse,
                "PINN_NMSE_dB": pinn_nmse,
                "PINN_PSNR_dB": pinn_psnr,
                "PINN_SSIM": pinn_ssim,
                "PIDR_MSE": pidr_mse,
                "PIDR_NMSE_dB": pidr_nmse,
                "PIDR_PSNR_dB": pidr_psnr,
                "PIDR_SSIM": pidr_ssim,
            }
        )

        if idx < max_plots:
            titles = {
                "gt": "Ground Truth",
                "sparse": "Sparse (Observed)",
                "diffusion": f"Diffusion\nMSE: {diff_mse:.5f}",
                "pinn": f"PINN\nMSE: {pinn_mse:.5f}",
                "pidr": f"{PROPOSED_METHOD_NAME}\nMSE: {pidr_mse:.5f}",
            }
            plot_comparison(
                save_path=os.path.join(out_dir, "plots", f"sample_{idx}.{plot_format}"),
                gt_map=gt_phys,
                sparse_map=sparse_display,
                diff_map=diff_phys,
                pinn_map=pinn_phys,
                pidr_map=pidr_phys,
                titles=titles,
            )

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, "per_sample_metrics.csv"), index=False)

    summary_rows = []
    for method in ("Diffusion", "PINN", "PIDR"):
        summary_rows.append(
            {
                "Method": method,
                "Avg_MSE": float(df[f"{method}_MSE"].mean()),
                "Avg_NMSE_dB": float(df[f"{method}_NMSE_dB"].mean()),
                "Avg_PSNR_dB": float(df[f"{method}_PSNR_dB"].mean()),
                "Avg_SSIM": float(df[f"{method}_SSIM"].mean()),
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)

    print("\n" + "=" * 60)
    print(f"Diffusion vs PINN vs {PROPOSED_METHOD_NAME} Summary")
    print("=" * 60)
    print(df_summary.to_string(index=False))
    print("=" * 60)
    print(f"Saved to: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Diffusion vs PINN vs PIDR on the Gain channel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--img-height", type=int, default=None)
    parser.add_argument("--img-width", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--diffusion-ckpt", default=None)
    parser.add_argument("--pinn-ckpt", default=None)
    parser.add_argument("--pidr-diffusion-ckpt", default=None)
    parser.add_argument("--pidr-refiner-ckpt", default=None)
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--mask-seed", type=int, default=None)
    parser.add_argument("--diffusion-base-channels", type=int, default=None)
    parser.add_argument("--sample-stride", type=int, default=None)
    parser.add_argument("--sampling-method", default="ddpm", choices=["ddpm"])
    parser.add_argument("--cache-postprocess", default="none", choices=["none", "linear_interp_blend"])
    parser.add_argument("--interp-weight", type=float, default=0.0)
    parser.add_argument("--output-input-blend", type=float, default=0.0)
    parser.add_argument("--num-eval-samples", type=int, default=None)
    parser.add_argument("--max-plots", type=int, default=None)
    parser.add_argument("--plot-format", default=None)
    return parser


if __name__ == "__main__":
    apply_cli_overrides(build_parser().parse_args())
    run_benchmark()
