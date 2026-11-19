"""Evaluate PIDR reconstruction on a held-out split of an EL CSI dataset.

This script intentionally uses the same CSIGridDataset loader as the training
and benchmark scripts, so it is useful for checking whether a generated map
problem comes from inference code or from the trained PIDR weights themselves.

Example:
    python -m scripts.eval.evaluate_pidr ^
        --data-path ./data/el.mat ^
        --diffusion-ckpt ./checkpoints/el/final/diffusion.pth ^
        --refiner-ckpt ./checkpoints/el/final/refiner.pth ^
        --eval-fraction 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.interpolate import griddata
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from pidr.models.pinn_refiner import build_refiner_from_state_dict
from pidr.config import Config
from pidr.data import CSIGridDataset
from pidr.diffusion import DiffusionManager
from pidr.models.diffusion_unet import CSIDiffusionPhysicsModel


CHANNEL_NAMES = ("Gain", "Real", "Imag")


def configure_data_shape(data_path: str, img_height: int | None, img_width: int | None, train_sample_count: int | None) -> None:
    if img_height is None or img_width is None:
        mat = loadmat(data_path)
        gain = np.squeeze(np.asarray(mat["CSI_Gain"]))
        if gain.ndim == 2:
            inferred_h, inferred_w = gain.shape
        elif gain.ndim == 3:
            spatial = sorted(list(gain.shape))[:2]
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


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calc_metrics(gt_map: np.ndarray, pred_map: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Return MSE, MAE, NMSE(dB), PSNR and SSIM for one 2D map."""
    gt_map = np.asarray(gt_map, dtype=np.float32)
    pred_map = np.asarray(pred_map, dtype=np.float32)

    if np.isnan(pred_map).any():
        pred_map = np.nan_to_num(pred_map, nan=float(np.nanmin(gt_map)))

    diff = pred_map - gt_map
    mse_val = float(np.mean(diff * diff))
    mae_val = float(np.mean(np.abs(diff)))
    norm = float(np.mean(gt_map * gt_map))
    nmse_db = float(10.0 * np.log10(mse_val / (norm + 1e-10)))

    data_range = float(np.nanmax(gt_map) - np.nanmin(gt_map))
    if data_range <= 0:
        data_range = 1.0

    psnr_val = float(psnr(gt_map, pred_map, data_range=data_range))
    ssim_val = float(ssim(gt_map, pred_map, data_range=data_range, win_size=3))
    return mse_val, mae_val, nmse_db, psnr_val, ssim_val


def sample_physics_guided(
    diffusion: DiffusionManager,
    model: torch.nn.Module,
    sparse_csi: torch.Tensor,
    mask: torch.Tensor,
    *,
    stride: int = 1,
) -> torch.Tensor:
    """Physics-guided ancestral sampling, matching the existing benchmark logic."""
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


def _load_plain_or_wrapped_state_dict(checkpoint_path: str, device: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def infer_diffusion_base_channels(checkpoint_path: str) -> int:
    state_dict = _load_plain_or_wrapped_state_dict(checkpoint_path, "cpu")
    weight = state_dict.get("input_proj.weight") if isinstance(state_dict, dict) else None
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


def load_checkpoint_metadata(checkpoint_path: str) -> Dict[str, object]:
    metadata_path = f"{checkpoint_path}.metadata.json"
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    return metadata if isinstance(metadata, dict) else {}


def resolve_evaluation_settings(args: argparse.Namespace) -> Dict[str, object]:
    metadata = load_checkpoint_metadata(args.refiner_ckpt)
    metadata_config = metadata.get("config", {}) if isinstance(metadata.get("config", {}), dict) else {}

    if args.refiner_input == "auto":
        pipeline = str(metadata_config.get("pipeline", "")).lower()
        if "diffusion" in pipeline:
            args.refiner_input = "diffusion"
        elif metadata_config:
            args.refiner_input = "sparse"
        else:
            args.refiner_input = "diffusion"

    if args.mask_strategy is None:
        args.mask_strategy = str(metadata_config.get("mask_strategy", "random"))

    if args.mask_seed is None:
        if metadata_config and args.mode == "train":
            args.mask_seed = int(metadata_config.get("train_mask_seed", Config.EVAL_MASK_SEED))
        elif metadata_config:
            args.mask_seed = int(metadata_config.get("eval_mask_seed", Config.EVAL_MASK_SEED))
        else:
            args.mask_seed = Config.EVAL_MASK_SEED

    if metadata_config:
        if args.mask_ratio is None and "mask_ratio" in metadata_config:
            args.mask_ratio = float(metadata_config["mask_ratio"])
        if args.diffusion_base_channels is None and "diffusion_base_channels" in metadata_config:
            args.diffusion_base_channels = int(metadata_config["diffusion_base_channels"])
        if args.sample_stride == 1 and "sample_stride" in metadata_config:
            args.sample_stride = int(metadata_config["sample_stride"])
        if args.cache_postprocess == "none":
            args.cache_postprocess = str(metadata_config.get("cache_postprocess", args.cache_postprocess))
        if args.interp_weight == 0.0:
            args.interp_weight = float(metadata_config.get("interp_weight", args.interp_weight))
        if args.output_input_blend == 0.0:
            args.output_input_blend = float(metadata_config.get("output_input_blend", args.output_input_blend))
        if args.sampling_method == "ddpm":
            args.sampling_method = str(metadata_config.get("sampling_method", args.sampling_method))

    return metadata


def load_models(
    diffusion_ckpt: str,
    refiner_ckpt: str,
    device: str,
    *,
    load_diffusion: bool = True,
) -> Tuple[torch.nn.Module | None, torch.nn.Module]:
    diffusion_model = None
    if load_diffusion:
        if not os.path.exists(diffusion_ckpt):
            raise FileNotFoundError(f"Diffusion checkpoint not found: {diffusion_ckpt}")
        diffusion_model = CSIDiffusionPhysicsModel().to(device).eval()
        diffusion_model.load_state_dict(_load_plain_or_wrapped_state_dict(diffusion_ckpt, device))

    if not os.path.exists(refiner_ckpt):
        raise FileNotFoundError(f"Refiner checkpoint not found: {refiner_ckpt}")

    refiner_model = build_refiner_from_state_dict(_load_plain_or_wrapped_state_dict(refiner_ckpt, device)).to(device).eval()
    return diffusion_model, refiner_model


def draw_heatmap_panel(
    ax: plt.Axes,
    data: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    *,
    cmap: str = "jet",
) -> None:
    data = np.ma.masked_invalid(np.asarray(data, dtype=np.float32))
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


def plot_sample(
    *,
    save_path: str,
    sample_id: int,
    channel_name: str,
    gt_map: np.ndarray,
    sparse_map: np.ndarray,
    pred_map: np.ndarray,
    mask: np.ndarray,
    metrics: Tuple[float, float, float, float, float],
) -> None:
    sparse_display = sparse_map.copy()
    sparse_display[mask < 0.5] = np.nan
    error_map = np.abs(pred_map - gt_map)

    vmin = float(np.nanmin(gt_map))
    vmax = float(np.nanmax(gt_map))
    err_max = float(np.nanmax(error_map))
    if err_max <= 0:
        err_max = 1.0

    mse_val, _mae_val, nmse_db, _psnr_val, ssim_val = metrics
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    draw_heatmap_panel(axes[0], gt_map, "Ground Truth", vmin, vmax)
    draw_heatmap_panel(axes[1], sparse_display, "Sparse Input", vmin, vmax)
    draw_heatmap_panel(
        axes[2],
        pred_map,
        f"PIDR Output\nMSE: {mse_val:.4g}\nNMSE: {nmse_db:.2f} dB\nSSIM: {ssim_val:.3f}",
        vmin,
        vmax,
    )
    draw_heatmap_panel(axes[3], error_map, "Absolute Error", 0.0, err_max, cmap="magma")
    fig.suptitle(f"Sample {sample_id} - {channel_name}", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def choose_eval_count(total: int, eval_fraction: float, num_samples: int | None) -> int:
    if not (0.0 < eval_fraction <= 1.0):
        raise ValueError(f"eval_fraction must be in (0, 1], got {eval_fraction}")
    count = max(1, int(round(total * eval_fraction)))
    if num_samples is not None:
        count = min(count, max(1, int(num_samples)))
    return min(count, total)


def run_evaluation(args: argparse.Namespace) -> None:
    configure_data_shape(args.data_path, args.img_height, args.img_width, args.train_sample_count)
    set_seed(args.seed)
    device = args.device
    metadata = resolve_evaluation_settings(args)
    if args.diffusion_base_channels is None and args.cache_postprocess != "linear_interp_only":
        args.diffusion_base_channels = infer_diffusion_base_channels(args.diffusion_ckpt)
    diffusion_base_channels = configure_diffusion_base_channels(args.diffusion_base_channels)
    if args.mask_ratio is not None:
        if not (0.0 <= args.mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in [0, 1), got {args.mask_ratio}")
        Config.MASK_RATIO = float(args.mask_ratio)
        Config.DATA.MASK_RATIO = float(args.mask_ratio)
    if metadata:
        print(f"Loaded refiner metadata: {args.refiner_ckpt}.metadata.json")
    print(
        "Resolved evaluation settings: "
        f"refiner_input={args.refiner_input}, "
        f"mask_strategy={args.mask_strategy}, "
        f"mask_ratio={Config.MASK_RATIO}, "
        f"mask_seed={args.mask_seed}, "
        f"diffusion_base_channels={diffusion_base_channels}, "
        f"sampling_method={args.sampling_method}, "
        f"cache_postprocess={args.cache_postprocess}, "
        f"interp_weight={args.interp_weight}, "
        f"output_input_blend={args.output_input_blend}"
    )

    dataset = CSIGridDataset(
        args.data_path,
        mode=args.mode,
        split_ratio=args.split_ratio,
        seed=args.split_seed,
        gain_in_db=not args.gain_linear,
        mask_strategy=args.mask_strategy,
        mask_seed=args.mask_seed,
    )

    num_eval = choose_eval_count(len(dataset), args.eval_fraction, args.num_samples)
    print(
        f"Evaluating {num_eval}/{len(dataset)} samples from {args.data_path}; "
        f"mode={args.mode}, split_ratio={args.split_ratio}, gain_in_db={not args.gain_linear}"
    )

    use_diffusion = args.refiner_input == "diffusion" and args.cache_postprocess != "linear_interp_only"
    use_interpolation_input = args.refiner_input == "diffusion" and args.cache_postprocess == "linear_interp_only"
    diffusion_model, refiner_model = load_models(
        args.diffusion_ckpt,
        args.refiner_ckpt,
        device,
        load_diffusion=use_diffusion,
    )
    diffusion_mgr = DiffusionManager(device, dataset.global_min, dataset.global_max)

    os.makedirs(args.output_dir, exist_ok=True)
    heatmap_dir = os.path.join(args.output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    records: List[Dict[str, object]] = []
    plot_count = min(num_eval, max(0, int(args.max_plots)))

    batch_size = max(1, int(args.batch_size))
    for batch_start in tqdm(range(0, num_eval, batch_size), desc="Evaluate PIDR"):
        batch_indices = list(range(batch_start, min(num_eval, batch_start + batch_size)))
        samples = [dataset[dataset_idx] for dataset_idx in batch_indices]
        gt = torch.stack([sample["gt_csi"] for sample in samples], dim=0)
        sparse = torch.stack([sample["sparse_csi"] for sample in samples], dim=0)
        mask = torch.stack([sample["mask"] for sample in samples], dim=0)

        with torch.no_grad():
            sparse_batch = sparse.to(device)
            mask_batch = mask.to(device)
            if use_interpolation_input:
                diff_out = interpolate_sparse_batch(sparse_batch, mask_batch).to(device)
                diff_out = diff_out * (1.0 - mask_batch) + sparse_batch * mask_batch
                refiner_input = diff_out
            elif use_diffusion:
                if args.sampling_method == "ddpm":
                    diff_out = sample_physics_guided(
                        diffusion_mgr,
                        diffusion_model,
                        sparse_batch,
                        mask_batch,
                        stride=args.sample_stride,
                    )
                elif args.sampling_method == "ddim":
                    diff_out = sample_ddim_physics_guided(
                        diffusion_mgr,
                        diffusion_model,
                        sparse_batch,
                        mask_batch,
                        stride=args.sample_stride,
                        return_final_x0=False,
                    )
                elif args.sampling_method == "ddim_x0":
                    diff_out = sample_ddim_physics_guided(
                        diffusion_mgr,
                        diffusion_model,
                        sparse_batch,
                        mask_batch,
                        stride=args.sample_stride,
                        return_final_x0=True,
                    )
                else:
                    raise ValueError(f"Unsupported sampling_method: {args.sampling_method}")
                if args.cache_postprocess == "linear_interp_blend":
                    interp = interpolate_sparse_batch(sparse_batch, mask_batch).to(device)
                    diff_out = (1.0 - args.interp_weight) * diff_out + args.interp_weight * interp
                    diff_out = diff_out * (1.0 - mask_batch) + sparse_batch * mask_batch
                refiner_input = diff_out
            else:
                refiner_input = sparse_batch
            pidr_out = refiner_model(refiner_input, mask_batch)
            if args.output_input_blend > 0.0:
                pidr_out = (1.0 - args.output_input_blend) * pidr_out + args.output_input_blend * refiner_input
            if args.enforce_observed:
                pidr_out = pidr_out * (1 - mask_batch) + sparse_batch * mask_batch

        pred_norm_batch = pidr_out.detach().cpu().numpy()
        gt_norm_batch = gt.numpy()
        sparse_norm_batch = sparse.numpy()
        mask_np_batch = mask[:, 0].numpy()

        gt_phys_batch = dataset.inverse_normalize_np(gt_norm_batch)
        sparse_phys_batch = dataset.inverse_normalize_np(sparse_norm_batch)
        pred_phys_batch = dataset.inverse_normalize_np(pred_norm_batch)

        for batch_offset, dataset_idx in enumerate(batch_indices):
            sample_id = int(dataset.indices[dataset_idx])
            mask_np = mask_np_batch[batch_offset]
            gt_phys_all = gt_phys_batch[batch_offset]
            sparse_phys_all = sparse_phys_batch[batch_offset]
            pred_phys_all = pred_phys_batch[batch_offset]

            for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
                metric_values = calc_metrics(gt_phys_all[channel_idx], pred_phys_all[channel_idx])
                mse_val, mae_val, nmse_db, psnr_val, ssim_val = metric_values
                records.append(
                    {
                        "sample_id": sample_id,
                        "dataset_idx": dataset_idx,
                        "channel": channel_name,
                        "observed_count": int(mask_np.sum()),
                        "mse": mse_val,
                        "mae": mae_val,
                        "nmse_db": nmse_db,
                        "psnr_db": psnr_val,
                        "ssim": ssim_val,
                    }
                )

                if dataset_idx < plot_count and channel_idx == args.plot_channel:
                    plot_sample(
                        save_path=os.path.join(heatmap_dir, f"sample_{sample_id}_{channel_name.lower()}.png"),
                        sample_id=sample_id,
                        channel_name=channel_name,
                        gt_map=gt_phys_all[channel_idx],
                        sparse_map=sparse_phys_all[channel_idx],
                        pred_map=pred_phys_all[channel_idx],
                        mask=mask_np,
                        metrics=metric_values,
                    )

    detailed = pd.DataFrame(records)
    summary = (
        detailed.groupby("channel", as_index=False)[["observed_count", "mse", "mae", "nmse_db", "psnr_db", "ssim"]]
        .mean()
        .sort_values("channel")
    )

    detailed_path = os.path.join(args.output_dir, "detailed_metrics.csv")
    summary_path = os.path.join(args.output_dir, "summary_metrics.csv")
    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nPIDR evaluation summary:")
    print(summary.to_string(index=False))
    print(f"\nDetailed metrics saved to: {detailed_path}")
    print(f"Summary metrics saved to: {summary_path}")
    print(f"Heatmaps saved to: {heatmap_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PIDR output against GT maps on a configurable test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", type=str, default="./data/el.mat")
    parser.add_argument("--img-height", type=int, default=None)
    parser.add_argument("--img-width", type=int, default=None)
    parser.add_argument("--train-sample-count", type=int, default=None)
    parser.add_argument("--diffusion-ckpt", type=str, default="./checkpoints/el/final/diffusion.pth")
    parser.add_argument("--diffusion-base-channels", type=int, default=None)
    parser.add_argument("--refiner-ckpt", type=str, default="./checkpoints/el/final/refiner.pth")
    parser.add_argument("--output-dir", type=str, default="./pidr_0.8el_recovered_testset_eval")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--split-ratio", type=float, default=Config.DATA_SPLIT_RATIO)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--mask-ratio", type=float, default=None, help="Missing ratio. Defaults to checkpoint metadata or Config.MASK_RATIO.")
    parser.add_argument("--eval-fraction", type=float, default=1.0, help="Fraction of selected split to evaluate.")
    parser.add_argument("--num-samples", type=int, default=None, help="Optional hard cap after eval-fraction.")
    parser.add_argument(
        "--mask-strategy",
        type=str,
        default=None,
        choices=["random", "uniform", "extreme_uneven"],
        help="Mask strategy. Defaults to refiner checkpoint metadata when available, otherwise random.",
    )
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=None,
        help="Mask seed. Defaults to train/eval seed from refiner metadata when available, otherwise Config.EVAL_MASK_SEED.",
    )
    parser.add_argument("--gain-linear", action="store_true", help="Use only if CSI_Gain is linear, not dB.")
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--sampling-method", default="ddpm", choices=["ddpm", "ddim", "ddim_x0"])
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation diffusion/refiner inference.")
    parser.add_argument("--cache-postprocess", default="none", choices=["none", "linear_interp_blend", "linear_interp_only"])
    parser.add_argument("--interp-weight", type=float, default=0.0)
    parser.add_argument("--output-input-blend", type=float, default=0.0)
    parser.add_argument(
        "--refiner-input",
        type=str,
        default="auto",
        choices=["auto", "diffusion", "sparse"],
        help="auto uses sparse input for refiner-only checkpoints with metadata, otherwise diffusion.",
    )
    parser.add_argument("--device", type=str, default=Config.DEVICE)
    parser.add_argument("--seed", type=int, default=0, help="Seed for diffusion sampling noise.")
    parser.add_argument("--max-plots", type=int, default=10)
    parser.add_argument("--plot-channel", type=int, default=0, choices=[0, 1, 2], help="0=Gain, 1=Real, 2=Imag.")
    parser.add_argument("--no-enforce-observed", dest="enforce_observed", action="store_false")
    parser.set_defaults(enforce_observed=True)
    return parser


def main() -> None:
    parser = build_parser()
    run_evaluation(parser.parse_args())


if __name__ == "__main__":
    main()
