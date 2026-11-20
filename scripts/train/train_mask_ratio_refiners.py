"""Train the field-response refiner for the missing-ratio experiment (SR).

For every missing ratio rho the frozen first stage is the curated diffusion model of that setting
(``checkpoints/<ds>/mask_ratio/rho<rho>/diffusion.pth``); the refiner is trained exactly as in
``train_pidr.py`` (missing-region L1, 1000 epochs) with train/eval masks drawn at the same rho.
The field-response physics layer is calibrated on the training split of each setting.

Usage (repo root):
    PIDR_DATASET=sr python -m scripts.train.train_mask_ratio_refiners [--ratios 0.7 0.8 0.9 0.95]
Outputs: runs/<ds>/mask_ratio_fr/rho<rho>/{refiner.pth, cache/}
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from pidr.config import Config
from scripts.train import train_pidr


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.7, 0.8, 0.9, 0.95])
    parser.add_argument("--epochs", type=int, default=Config.PIDR.PINN_EPOCHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", default=os.path.join("./runs", Config.DATASET, "mask_ratio_fr"))
    args = parser.parse_args()

    for ratio in args.ratios:
        tag = f"rho{ratio:.2f}"
        diffusion_ckpt = os.path.join(Config.PATHS.CHECKPOINT_DIR, Config.DATASET, "mask_ratio", tag, "diffusion.pth")
        if not os.path.exists(diffusion_ckpt):
            raise FileNotFoundError(diffusion_ckpt)
        Config.MASK_RATIO = ratio
        Config.DATA.MASK_RATIO = ratio
        out_dir = os.path.join(args.out_root, tag)
        print(f"\n===== missing ratio {ratio} -> {out_dir} (diffusion {diffusion_ckpt})", flush=True)
        seed_everything(args.seed)
        paths = train_pidr.PIDRPaths(
            diffusion_ckpt=diffusion_ckpt,
            refiner_ckpt=os.path.join(out_dir, "refiner.pth"),
            cache_train=os.path.join(out_dir, "cache", "diffusion_recon_train.pt"),
            cache_test=os.path.join(out_dir, "cache", "diffusion_recon_test.pt"),
        )
        train_pidr.train_pinn_stage(
            device=Config.DEVICE,
            paths=paths,
            num_epochs=args.epochs,
            sample_stride=Config.PIDR.SAMPLE_STRIDE,
            force_rebuild_cache=False,
            train_mask_seed=Config.PIDR.PINN_TRAIN_MASK_SEED,
            refiner_kwargs=dict(prior="field_response", use_coords=True, residual="physics"),
        )


if __name__ == "__main__":
    main()
