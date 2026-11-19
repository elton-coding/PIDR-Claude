"""Evaluate the refiner ablation (scripts/train/train_refiner_ablation.py) on the test split.

Two paired evaluations (every variant refines exactly the same diffusion estimates):

1. ``cache``: all test maps, using the cached diffusion reconstructions from training
   (eval mask seed, stride 1). Deterministic.
2. ``protocol``: the main-table protocol, i.e. the first N test maps with fresh diffusion sampling,
   repeated over several seeds; within a seed all variants share the same diffusion output.

Gain-channel metrics are reported in the paper's units (10*log10|h|^2, i.e. twice the gain stored
in the current data files), so MSE is comparable with the paper; NMSE/PSNR/SSIM are unit-free.

Usage (repo root):
    PIDR_DATASET=sr python -m scripts.eval.eval_refiner_ablation [--seeds 0 1 2 3 4] [--n 50]
Writes results/refiner_ablation_<dataset>/{summary.csv,per_sample_cache.csv,learned_prior.csv}.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from pidr.config import Config
from pidr.data import CSIGridDataset
from pidr.diffusion import DiffusionManager
from pidr.models.diffusion_unet import CSIDiffusionPhysicsModel
from pidr.models.pinn_refiner import PINNCSIReconstructor
from scripts.eval.benchmark_methods import calc_metrics, sample_physics_guided
from scripts.train.train_refiner_ablation import VARIANTS

GAIN_TO_PAPER_UNITS = 2.0  # data: 10*log10|h|; paper: 10*log10|h|^2


def gain_metrics(dataset: CSIGridDataset, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """(N, 4) array of MSE, NMSE, PSNR, SSIM on the gain channel in paper units."""
    out = []
    for i in range(gt.shape[0]):
        g = dataset.inverse_normalize_channel(gt[i, 0], 0) * GAIN_TO_PAPER_UNITS
        p = dataset.inverse_normalize_channel(pred[i, 0], 0) * GAIN_TO_PAPER_UNITS
        out.append(calc_metrics(g, p))
    return np.asarray(out)


def refine(model: torch.nn.Module, coarse: torch.Tensor, sparse: torch.Tensor, mask: torch.Tensor, batch: int = 256) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for s in range(0, coarse.shape[0], batch):
            c, x, m = coarse[s:s + batch], sparse[s:s + batch], mask[s:s + batch]
            outs.append(model(c, m) * (1 - m) + x * m)
    return torch.cat(outs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join("./runs", Config.DATASET, "ablation_refiner"))
    parser.add_argument("--diffusion-ckpt", default=Config.PATHS.PIDR_DIFFUSION_CHECKPOINT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--out-dir", default=os.path.join("results", f"refiner_ablation_{Config.DATASET}"))
    args = parser.parse_args()
    device = Config.DEVICE
    os.makedirs(args.out_dir, exist_ok=True)

    ds = CSIGridDataset(Config.DATA_PATH, mode="test", split_ratio=Config.DATA_SPLIT_RATIO, mask_seed=Config.EVAL_MASK_SEED)
    items = [ds[i] for i in range(len(ds))]
    gt = torch.stack([it["gt_csi"] for it in items])
    sparse = torch.stack([it["sparse_csi"] for it in items]).to(device)
    mask = torch.stack([it["mask"] for it in items]).to(device)

    models = {}
    for key, (name, flags) in VARIANTS.items():
        path = os.path.join(args.root, name, "refiner.pth")
        if os.path.exists(path):
            m = PINNCSIReconstructor(**flags).to(device).eval()
            m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            models[name] = m
    if not models:
        raise FileNotFoundError(f"no trained variants under {args.root}")
    print("variants:", list(models))

    rows, per_sample = [], {}

    # 1) all test maps, cached diffusion estimates
    cache = torch.load(os.path.join(args.root, "cache", "diffusion_recon_test.pt"), map_location="cpu", weights_only=False)
    coarse = cache["diffusion_recon"].float().to(device)
    assert coarse.shape[0] == len(ds), "cache does not match the test split"
    outputs = {"Diffusion only": coarse}
    outputs.update({name: refine(m, coarse, sparse, mask) for name, m in models.items()})
    for name, pred in outputs.items():
        met = gain_metrics(ds, gt.numpy(), pred.cpu().numpy())
        per_sample[name] = met
        rows.append({"evaluation": f"cache (all {len(ds)} test maps)", "method": name, "runs": 1,
                     **{k: met[:, i].mean() for i, k in enumerate(["MSE", "NMSE_dB", "PSNR_dB", "SSIM"])},
                     **{f"{k}_std_over_seeds": np.nan for k in ["NMSE_dB", "SSIM"]}})
    ref = per_sample.get("A_full")
    if ref is not None:
        for name, met in per_sample.items():
            if name != "A_full":
                rows.append({"evaluation": "cache: fraction of maps where A_full has lower MSE", "method": name, "runs": 1,
                             "MSE": float(np.mean(ref[:, 0] < met[:, 0]))})
    pd.DataFrame({f"{name}_{k}": met[:, i] for name, met in per_sample.items()
                  for i, k in enumerate(["MSE", "NMSE_dB", "PSNR_dB", "SSIM"])}).to_csv(
        os.path.join(args.out_dir, "per_sample_cache.csv"), index_label="test_index")

    # 2) main-table protocol: first N maps, fresh diffusion sampling per seed (shared by variants)
    diff_model = CSIDiffusionPhysicsModel().to(device).eval()
    diff_model.load_state_dict(torch.load(args.diffusion_ckpt, map_location=device, weights_only=True))
    mgr = DiffusionManager(device, ds.global_min, ds.global_max)
    n = min(args.n, len(ds))
    seed_results = {name: [] for name in ["Diffusion only", *models]}
    for seed in args.seeds:
        torch.manual_seed(seed)
        with torch.no_grad():
            c = sample_physics_guided(mgr, diff_model, sparse[:n], mask[:n], stride=1)
        outs = {"Diffusion only": c}
        outs.update({name: refine(m, c, sparse[:n], mask[:n]) for name, m in models.items()})
        for name, pred in outs.items():
            seed_results[name].append(gain_metrics(ds, gt[:n].numpy(), pred.cpu().numpy()).mean(0))
        print(f"seed {seed}: " + ", ".join(f"{k} NMSE {v[-1][1]:.3f}" for k, v in seed_results.items()), flush=True)
    for name, vals in seed_results.items():
        v = np.asarray(vals)
        rows.append({"evaluation": f"protocol (first {n} test maps, {len(args.seeds)} seeds)", "method": name,
                     "runs": len(args.seeds),
                     **{k: v[:, i].mean() for i, k in enumerate(["MSE", "NMSE_dB", "PSNR_dB", "SSIM"])},
                     "NMSE_dB_std_over_seeds": v[:, 1].std(ddof=1) if len(v) > 1 else np.nan,
                     "SSIM_std_over_seeds": v[:, 3].std(ddof=1) if len(v) > 1 else np.nan})

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(4).to_string(index=False))

    if "A_full" in models:
        p = models["A_full"].prior
        prior = {"tx_x": torch.sigmoid(p.tx_x_raw).item(), "tx_y": torch.sigmoid(p.tx_y_raw).item(),
                 "pathloss_exponent_n": p.get_pathloss_exponent().item(), "wave_number_k": p.get_wave_number().item(),
                 "amp_scale": p.amp_scale.item(), "amp_bias": p.amp_bias.item(), "phase_bias": p.phase_bias.item()}
        pd.DataFrame([prior]).to_csv(os.path.join(args.out_dir, "learned_prior.csv"), index=False)
        print("learned prior (A_full):", {k: round(v, 4) for k, v in prior.items()})


if __name__ == "__main__":
    main()
