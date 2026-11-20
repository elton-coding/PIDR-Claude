"""Train the field-response refiner for the training-measurement-budget (semi-supervised) experiment.

For every available ratio alpha the frozen first stage is the curated semi-supervised diffusion model
(``checkpoints/<ds>/semisup/alpha<a>/diffusion.pth``); the refiner stage of
``train_pidr_semisupervised.py`` is run with the field-response refiner. No complete training map is
used: the loss and the physics-layer calibration only use the held-out supervision points.

Usage (repo root):
    PIDR_DATASET=sr python -m scripts.train.train_semisup_refiners [--alphas 0.1 0.2 0.3 0.4 0.5 0.6]
Outputs: runs/<ds>/semisup_fr/alpha<a>/{refiner.pth, cache/}
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from pidr.config import Config
from scripts.train import train_pidr_semisupervised as semisup


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--epochs", type=int, default=Config.PIDR_PARTIAL.REFINER_EPOCHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", default=os.path.join("./runs", Config.DATASET, "semisup_fr"))
    args = parser.parse_args()

    for alpha in args.alphas:
        tag = f"alpha{alpha:g}"
        diffusion_ckpt = os.path.join(Config.PATHS.CHECKPOINT_DIR, Config.DATASET, "semisup", tag, "diffusion.pth")
        if not os.path.exists(diffusion_ckpt):
            raise FileNotFoundError(diffusion_ckpt)
        Config.PIDR_PARTIAL.TOTAL_AVAILABLE_RATIO = alpha
        semisup.validate_partial_config()
        out_dir = os.path.join(args.out_root, tag)
        print(f"\n===== alpha {alpha} -> {out_dir} (diffusion {diffusion_ckpt})", flush=True)
        seed_everything(args.seed)
        paths = semisup.PartialPIDRPaths(
            diffusion_ckpt=diffusion_ckpt,
            refiner_ckpt=os.path.join(out_dir, "refiner.pth"),
            cache_train=os.path.join(out_dir, "cache", "diffusion_recon_train.pt"),
            cache_test=os.path.join(out_dir, "cache", "diffusion_recon_test.pt"),
        )
        semisup.train_refiner_stage(
            device=Config.DEVICE,
            paths=paths,
            num_epochs=args.epochs,
            sample_stride=Config.PIDR_PARTIAL.SAMPLE_STRIDE,
            force_rebuild_cache=False,
            refiner_kwargs=dict(prior="field_response", use_coords=True, residual="physics"),
        )


if __name__ == "__main__":
    main()
