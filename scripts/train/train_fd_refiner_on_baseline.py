"""Train a PIDR refiner on top of the field-response baseline diffusion model.

The cross-dataset re-evaluation (docs/checkpoints.md) showed that every existing FD refiner
was trained on the outputs of a different, weaker diffusion model, so none of them improves
``checkpoints/fd/baselines/diffusion.pth``. This script runs the standard PIDR refiner stage of
``train_pidr.py`` (same loss, epochs and cache logic) with that diffusion model as the frozen
first stage. Output goes to ``runs/fd/<timestamp>/pidr/refiner.pth``.

Usage (repo root):
    PIDR_DATASET=fd python -m scripts.train.train_fd_refiner_on_baseline
"""

import os

from pidr.config import Config
from scripts.train import train_pidr

if Config.DATASET != "fd":
    raise SystemExit("Run with PIDR_DATASET=fd.")

Config.PIDR.TRAIN_DIFFUSION = False
Config.PIDR.TRAIN_PINN = True
Config.PIDR.LEGACY_DIFFUSION_CHECKPOINT = os.path.join(Config.PATHS.CHECKPOINT_DIR, "fd", "baselines", "diffusion.pth")
print(f"Frozen diffusion stage: {Config.PIDR.LEGACY_DIFFUSION_CHECKPOINT}")
print(f"Refiner output: {Config.PIDR.REFINER_CHECKPOINT}")

if __name__ == "__main__":
    train_pidr.main()
