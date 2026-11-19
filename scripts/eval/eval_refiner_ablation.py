"""Evaluate the refiner ablation (scripts/train/train_refiner_ablation.py) on the test split.

Two paired evaluations (every variant refines exactly the same diffusion estimates):

1. ``cache``: all test maps, using the cached diffusion reconstructions from training
   (eval mask seed, stride 1). Deterministic.
2. ``protocol``: the main-table protocol, i.e. the first N test maps with fresh diffusion sampling,
   repeated over several sampling seeds; within a seed all variants share the same diffusion output.

Variants trained with several seeds (``<variant>``, ``<variant>_s1``, ...) are reported per training
seed and as mean +- std over training seeds. "Physics only" is the output of the field-response
LMMSE layer alone (no network), taken from the first field-response refiner found.

Gain-channel metrics are reported in the paper's units (10*log10|h|^2, i.e. twice the gain stored
in the current data files), so MSE is comparable with the paper; NMSE/PSNR/SSIM are unit-free.

Usage (repo root):
    PIDR_DATASET=sr python -m scripts.eval.eval_refiner_ablation [--root runs/sr/ablation_refiner]
        [--variants FR NP] [--seeds 0 1 2 3 4] [--n 50] [--out-dir results/refiner_ablation_sr]
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
import torch

from pidr.config import Config
from pidr.data import CSIGridDataset
from pidr.diffusion import DiffusionManager
from pidr.models.diffusion_unet import CSIDiffusionPhysicsModel
from pidr.models.pinn_refiner import load_refiner
from scripts.eval.benchmark_methods import calc_metrics, sample_physics_guided

GAIN_TO_PAPER_UNITS = 2.0  # data: 10*log10|h|; paper: 10*log10|h|^2
METRICS = ["MSE", "NMSE_dB", "PSNR_dB", "SSIM"]


def gain_metrics(dataset: CSIGridDataset, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """(N, 4) array of MSE, NMSE, PSNR, SSIM on the gain channel in paper units."""
    out = []
    for i in range(gt.shape[0]):
        g = dataset.inverse_normalize_channel(gt[i, 0], 0) * GAIN_TO_PAPER_UNITS
        p = dataset.inverse_normalize_channel(pred[i, 0], 0) * GAIN_TO_PAPER_UNITS
        out.append(calc_metrics(g, p))
    return np.asarray(out)


def refine(model, coarse, sparse, mask, batch: int = 128) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for s in range(0, coarse.shape[0], batch):
            c, x, m = coarse[s:s + batch], sparse[s:s + batch], mask[s:s + batch]
            outs.append(model(c, m) * (1 - m) + x * m)
    return torch.cat(outs)


def physics_only(model, coarse, sparse, mask, batch: int = 128) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for s in range(0, coarse.shape[0], batch):
            c, x, m = coarse[s:s + batch], sparse[s:s + batch], mask[s:s + batch]
            outs.append(model.physics_features(c, m)[:, :3] * (1 - m) + x * m)
    return torch.cat(outs)


def discover(root: str, wanted):
    """{run_name: (variant, train_seed, path)} for every trained refiner under root."""
    runs = {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "refiner.pth")
        if not os.path.exists(path):
            continue
        m = re.fullmatch(r"(.+?)(?:_s(\d+))?", name)
        variant, seed = m.group(1), int(m.group(2) or 0)
        if wanted and variant not in wanted:
            continue
        runs[name] = (variant, seed, path)
    return runs


def summarise(rows_per_run, runs, evaluation):
    """Per-variant mean/std over training seeds."""
    out = []
    by_variant = {}
    for run_name, vals in rows_per_run.items():
        variant = runs[run_name][0] if run_name in runs else run_name
        by_variant.setdefault(variant, []).append(vals)
    for variant, vals in by_variant.items():
        v = np.asarray(vals)
        row = {"evaluation": evaluation, "method": variant, "train_seeds": len(v)}
        row.update({k: v[:, i].mean() for i, k in enumerate(METRICS)})
        row.update({f"{k}_std_train_seeds": (v[:, i].std(ddof=1) if len(v) > 1 else np.nan) for i, k in enumerate(METRICS)})
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join("./runs", Config.DATASET, "ablation_refiner"))
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--diffusion-ckpt", default=Config.PATHS.PIDR_DIFFUSION_CHECKPOINT)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4], help="diffusion sampling seeds")
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

    runs = discover(args.root, set(args.variants) if args.variants else None)
    if not runs:
        raise FileNotFoundError(f"no trained variants under {args.root}")
    models = {name: load_refiner(path, device) for name, (_v, _s, path) in runs.items()}
    fr_model = next((m for m in models.values() if m.prior_type == "field_response"), None)
    print("runs:", {k: (v[0], v[1]) for k, v in runs.items()}, flush=True)

    rows, per_run_rows = [], []

    # 1) all test maps, cached diffusion estimates
    cache = torch.load(os.path.join(args.root, "cache", "diffusion_recon_test.pt"), map_location="cpu", weights_only=False)
    coarse = cache["diffusion_recon"].float().to(device)
    assert coarse.shape[0] == len(ds), "cache does not match the test split"
    outputs = {"Diffusion only": coarse}
    if fr_model is not None:
        outputs["Physics only"] = physics_only(fr_model, coarse, sparse, mask)
    outputs.update({name: refine(m, coarse, sparse, mask) for name, m in models.items()})
    per_sample, means = {}, {}
    for name, pred in outputs.items():
        met = gain_metrics(ds, gt.numpy(), pred.cpu().numpy())
        per_sample[name] = met
        means[name] = met.mean(0)
        per_run_rows.append({"evaluation": "cache", "run": name, **dict(zip(METRICS, met.mean(0)))})
    rows += summarise(means, runs, f"cache (all {len(ds)} test maps)")
    pd.DataFrame({f"{name}_{k}": met[:, i] for name, met in per_sample.items()
                  for i, k in enumerate(METRICS)}).to_csv(os.path.join(args.out_dir, "per_sample_cache.csv"), index_label="test_index")

    # paired win rates of each physics variant against each non-physics variant (same training seed)
    for a in models:
        for b in models:
            va, sa, _ = runs[a]
            vb, sb, _ = runs[b]
            if sa == sb and models[a].prior_type == "field_response" and models[b].prior_type != "field_response":
                rows.append({"evaluation": "cache: fraction of maps with lower MSE", "method": f"{va} vs {vb} (seed {sa})",
                             "train_seeds": 1, "MSE": float(np.mean(per_sample[a][:, 0] < per_sample[b][:, 0]))})

    # 2) main-table protocol: first N maps, fresh diffusion sampling per seed (shared by variants)
    if args.seeds:
        diff_model = CSIDiffusionPhysicsModel().to(device).eval()
        diff_model.load_state_dict(torch.load(args.diffusion_ckpt, map_location=device, weights_only=True))
        mgr = DiffusionManager(device, ds.global_min, ds.global_max)
        n = min(args.n, len(ds))
        seed_results = {}
        for seed in args.seeds:
            torch.manual_seed(seed)
            with torch.no_grad():
                c = sample_physics_guided(mgr, diff_model, sparse[:n], mask[:n], stride=1)
            outs = {"Diffusion only": c}
            if fr_model is not None:
                outs["Physics only"] = physics_only(fr_model, c, sparse[:n], mask[:n])
            outs.update({name: refine(m, c, sparse[:n], mask[:n]) for name, m in models.items()})
            for name, pred in outs.items():
                seed_results.setdefault(name, []).append(gain_metrics(ds, gt[:n].numpy(), pred.cpu().numpy()).mean(0))
            print(f"sampling seed {seed}: " + ", ".join(f"{k} {v[-1][1]:.3f}" for k, v in seed_results.items()), flush=True)
        proto_means = {name: np.asarray(v).mean(0) for name, v in seed_results.items()}
        for name, v in seed_results.items():
            per_run_rows.append({"evaluation": f"protocol (first {n}, {len(args.seeds)} sampling seeds)", "run": name,
                                 **dict(zip(METRICS, np.asarray(v).mean(0))),
                                 "NMSE_dB_std_sampling_seeds": np.asarray(v)[:, 1].std(ddof=1) if len(v) > 1 else np.nan})
        rows += summarise(proto_means, runs, f"protocol (first {n} test maps, {len(args.seeds)} sampling seeds)")

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    pd.DataFrame(per_run_rows).to_csv(os.path.join(args.out_dir, "per_run.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(summary.round(4).to_string(index=False))

    if fr_model is not None:
        layer = fr_model.prior
        calib = {"beta": float(layer.beta), "nugget": float(layer.nugget), "iterations": int(layer.iterations)}
        pd.DataFrame([calib]).to_csv(os.path.join(args.out_dir, "physics_layer.csv"), index=False)
        print("field-response layer:", calib)


if __name__ == "__main__":
    main()
