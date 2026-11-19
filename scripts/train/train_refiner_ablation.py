"""Refiner ablation: does the physics-driven input (propagation prior) help the PIDR refiner?

Variants (all other settings identical; see pidr.models.pinn_refiner.PINNCSIReconstructor):
    A  full      diffusion estimate + mask + propagation prior + coordinates  (paper model)
    B  no_prior  diffusion estimate + mask + coordinates
    C  plain     diffusion estimate + mask

Every variant runs the standard refiner stage of ``train_pidr.py`` (missing-region L1 loss,
Config.PIDR.PINN_EPOCHS epochs, best epoch kept) on the same frozen diffusion model and the same
diffusion-reconstruction cache, with the same random seed.

Usage (repo root):
    PIDR_DATASET=sr python -m scripts.train.train_refiner_ablation [--variants A B C] [--seed 0]
Outputs: runs/<dataset>/ablation_refiner/{cache,A_full,B_no_prior,C_plain}/
"""

from __future__ import annotations

import argparse
import functools
import os
import random

import numpy as np
import torch

from pidr.config import Config
from pidr.models.pinn_refiner import PINNCSIReconstructor
from scripts.train import train_pidr

VARIANTS = {
    "A": ("A_full", dict(use_prior=True, use_coords=True)),
    "B": ("B_no_prior", dict(use_prior=False, use_coords=True)),
    "C": ("C_plain", dict(use_prior=False, use_coords=False)),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=["A", "B", "C"], choices=sorted(VARIANTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diffusion-ckpt", default=Config.PATHS.PIDR_DIFFUSION_CHECKPOINT,
                        help="frozen first stage (default: the preset's PIDR-Net diffusion model)")
    parser.add_argument("--out-root", default=os.path.join("./runs", Config.DATASET, "ablation_refiner"))
    args = parser.parse_args()

    if not os.path.exists(args.diffusion_ckpt):
        raise FileNotFoundError(args.diffusion_ckpt)
    cache_dir = os.path.join(args.out_root, "cache")
    print(f"dataset={Config.DATASET} diffusion={args.diffusion_ckpt} epochs={Config.PIDR.PINN_EPOCHS} seed={args.seed}")

    original_cls = train_pidr.PINNCSIReconstructor
    try:
        for key in args.variants:
            name, flags = VARIANTS[key]
            out_dir = os.path.join(args.out_root, name)
            print(f"\n===== Variant {name}: {flags} -> {out_dir}")
            # train_pinn_stage builds the refiner with PINNCSIReconstructor(); swap in the variant.
            train_pidr.PINNCSIReconstructor = functools.partial(PINNCSIReconstructor, **flags)
            seed_everything(args.seed)
            paths = train_pidr.PIDRPaths(
                diffusion_ckpt=args.diffusion_ckpt,
                refiner_ckpt=os.path.join(out_dir, "refiner.pth"),
                cache_train=os.path.join(cache_dir, "diffusion_recon_train.pt"),
                cache_test=os.path.join(cache_dir, "diffusion_recon_test.pt"),
            )
            train_pidr.train_pinn_stage(
                device=Config.DEVICE,
                paths=paths,
                num_epochs=Config.PIDR.PINN_EPOCHS,
                sample_stride=Config.PIDR.SAMPLE_STRIDE,
                force_rebuild_cache=False,
                train_mask_seed=Config.PIDR.PINN_TRAIN_MASK_SEED,
            )
    finally:
        train_pidr.PINNCSIReconstructor = original_cls


if __name__ == "__main__":
    main()
