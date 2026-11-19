"""Refiner ablation: does the physics-driven input branch help the PIDR refiner?

Every variant runs the standard refiner stage of ``train_pidr.py`` (missing-region L1 loss, best epoch
kept) on the same frozen diffusion model and the same diffusion-reconstruction cache, with the same
random seed; only the refiner variant differs (see pidr.models.pinn_refiner.PINNCSIReconstructor).

Variants:
    FR          field-response LMMSE physics branch + coordinates, residual on the physics estimate
                (proposed refiner)
    FR_coarse   field-response physics branch + coordinates, residual on the diffusion estimate
    FR_direct   field-response physics branch + coordinates, direct (tanh) output
    NP          no physics branch, coordinates, residual on the diffusion estimate
                (ablation counterpart of FR / FR_coarse)
    NP_direct   no physics branch, coordinates, direct output
    RAD         original global radial template (paper v13 refiner), direct output
    PLAIN       no physics branch, no coordinates, direct output

Usage (repo root):
    PIDR_DATASET=sr python -m scripts.train.train_refiner_ablation --variants FR NP --seeds 0 1 2 [--epochs 1000]
Outputs: <out-root>/<variant>[_s<seed>]/refiner.pth; the diffusion cache is shared in <out-root>/cache.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from pidr.config import Config
from scripts.train import train_pidr

VARIANTS = {
    "FR": dict(prior="field_response", use_coords=True, residual="physics"),
    "FR_coarse": dict(prior="field_response", use_coords=True, residual="coarse"),
    "FR_direct": dict(prior="field_response", use_coords=True, residual="none"),
    "NP": dict(prior="none", use_coords=True, residual="coarse"),
    "NP_direct": dict(prior="none", use_coords=True, residual="none"),
    "RAD": dict(prior="radial", use_coords=True, residual="none"),
    "PLAIN": dict(prior="none", use_coords=False, residual="none"),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_dir(out_root: str, variant: str, seed: int) -> str:
    return os.path.join(out_root, variant if seed == 0 else f"{variant}_s{seed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=["FR", "NP"], choices=sorted(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=Config.PIDR.PINN_EPOCHS)
    parser.add_argument("--diffusion-ckpt", default=Config.PATHS.PIDR_DIFFUSION_CHECKPOINT,
                        help="frozen first stage (default: the preset's PIDR-Net diffusion model)")
    parser.add_argument("--out-root", default=os.path.join("./runs", Config.DATASET, "ablation_refiner"))
    args = parser.parse_args()

    if not os.path.exists(args.diffusion_ckpt):
        raise FileNotFoundError(args.diffusion_ckpt)
    cache_dir = os.path.join(args.out_root, "cache")
    print(f"dataset={Config.DATASET} diffusion={args.diffusion_ckpt} epochs={args.epochs} seeds={args.seeds}", flush=True)

    for seed in args.seeds:
        for name in args.variants:
            out_dir = run_dir(args.out_root, name, seed)
            print(f"\n===== Variant {name} seed {seed}: {VARIANTS[name]} -> {out_dir}", flush=True)
            seed_everything(seed)
            paths = train_pidr.PIDRPaths(
                diffusion_ckpt=args.diffusion_ckpt,
                refiner_ckpt=os.path.join(out_dir, "refiner.pth"),
                cache_train=os.path.join(cache_dir, "diffusion_recon_train.pt"),
                cache_test=os.path.join(cache_dir, "diffusion_recon_test.pt"),
            )
            train_pidr.train_pinn_stage(
                device=Config.DEVICE,
                paths=paths,
                num_epochs=args.epochs,
                sample_stride=Config.PIDR.SAMPLE_STRIDE,
                force_rebuild_cache=False,
                train_mask_seed=Config.PIDR.PINN_TRAIN_MASK_SEED,
                refiner_kwargs=VARIANTS[name],
            )


if __name__ == "__main__":
    main()
