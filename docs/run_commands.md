# Final Model Metadata and Run Commands

This file records the final PIDR-Net models for the EL, Field/FD and SR datasets
(weights in `checkpoints/final/{el,fd,sr}/`). These models produced the reconstructed
maps in `data/test/*_pidr_*batch.mat` that the paper's movable-antenna experiments use.
Run all commands from the repository root in PowerShell (activate the conda env first).

```powershell
conda activate ma_diffusion
```

## Environment

Python environment (see `requirements.txt`):

```powershell
conda activate ma_diffusion
```

## EL Model

### Metadata

- Train dataset: `.\data\el.mat`
- Test/generation input example: `.\data\test\el.mat`
- Diffusion checkpoint: `.\checkpoints\final\el\pidr_diffusion_el.pth`
- Refiner checkpoint: `.\checkpoints\final\el\pidr_refiner_el.pth`
- Refiner format: plain `state_dict`
- Split ratio: `0.9`
- Split seed: `42`
- Mask ratio: `0.8`
- Mask strategy: `random`
- Train mask seed: `12345`
- Eval mask seed: `0`
- Diffusion base channels: `64`
- Sampling method: `ddpm`
- Sample stride: `1`
- Cache postprocess: `linear_interp_blend`
- Interp weight: `0.95`
- Output input blend: `0.98`
- Train SSIM, Gain/Real/Imag/min: `0.979207 / 0.968757 / 0.971697 / 0.968757`
- Test SSIM, Gain/Real/Imag/min: `0.966836 / 0.966257 / 0.963295 / 0.963295`

### Generate Channel Maps

```powershell
python -m scripts.ma.generate_channel_maps `
  --input .\data\test\el.mat `
  --output .\data\test\el_pidr_el_model_batch.mat `
  --heatmap-dir .\data\test\el_pidr_el_model_heatmaps `
  --metrics .\data\test\el_pidr_el_model_metrics.csv `
  --diffusion-ckpt .\checkpoints\final\el\pidr_diffusion_el.pth `
  --refiner-ckpt .\checkpoints\final\el\pidr_refiner_el.pth `
  --norm-stats-mat .\data\el.mat `
  --mask-ratio 0.8 `
  --mask-strategy random `
  --mask-seed 0 `
  --diffusion-base-channels 64 `
  --sample-stride 1 `
  --sampling-method ddpm `
  --cache-postprocess linear_interp_blend `
  --interp-weight 0.95 `
  --output-input-blend 0.98 `
  --batch-size 16 `
  --plot-channel 0 `
  --max-plots 50
```

### Evaluate Test Set

```powershell
python -m scripts.eval.evaluate_pidr `
  --data-path .\data\el.mat `
  --diffusion-ckpt .\checkpoints\final\el\pidr_diffusion_el.pth `
  --refiner-ckpt .\checkpoints\final\el\pidr_refiner_el.pth `
  --mode test `
  --split-ratio 0.9 `
  --split-seed 42 `
  --mask-ratio 0.8 `
  --mask-strategy random `
  --mask-seed 0 `
  --diffusion-base-channels 64 `
  --sample-stride 1 `
  --sampling-method ddpm `
  --cache-postprocess linear_interp_blend `
  --interp-weight 0.95 `
  --output-input-blend 0.98 `
  --batch-size 64 `
  --eval-fraction 1.0 `
  --max-plots 0 `
  --output-dir .\results\eval\el_test
```

### Evaluate Train Set

```powershell
python -m scripts.eval.evaluate_pidr `
  --data-path .\data\el.mat `
  --diffusion-ckpt .\checkpoints\final\el\pidr_diffusion_el.pth `
  --refiner-ckpt .\checkpoints\final\el\pidr_refiner_el.pth `
  --mode train `
  --split-ratio 0.9 `
  --split-seed 42 `
  --mask-ratio 0.8 `
  --mask-strategy random `
  --mask-seed 12345 `
  --diffusion-base-channels 64 `
  --sample-stride 1 `
  --sampling-method ddpm `
  --cache-postprocess linear_interp_blend `
  --interp-weight 0.95 `
  --output-input-blend 0.98 `
  --batch-size 64 `
  --eval-fraction 1.0 `
  --max-plots 0 `
  --output-dir .\results\eval\el_train
```

## Field/FD Model

### Metadata

- Train dataset: `.\data\field.mat`
- Test/generation input example: `.\data\test\fd.mat`
- Diffusion checkpoint: `.\checkpoints\final\fd\pidr_diffusion_fd.pth`
- Refiner checkpoint: `.\checkpoints\final\fd\pidr_refiner_fd.pth`
- Refiner format: plain `state_dict`
- Split ratio: `0.9`
- Split seed: `42`
- Mask ratio: `0.8`
- Mask strategy: `random`
- Train mask seed: `12345`
- Eval mask seed: `0`
- Diffusion base channels: `32`
- Sampling method: `ddpm`
- Sample stride: `1`
- Cache postprocess: `none`
- Interp weight: `0.0`
- Output input blend: `0.0`
- Train SSIM, Gain/Real/Imag/min: `0.963648 / 0.957603 / 0.955509 / 0.955509`
- Test SSIM, Gain/Real/Imag/min: `0.962584 / 0.955812 / 0.953180 / 0.953180`

### Generate Channel Maps

```powershell
python -m scripts.ma.generate_channel_maps `
  --input .\data\test\fd.mat `
  --output .\data\test\fd_pidr_field_model_batch.mat `
  --heatmap-dir .\data\test\fd_pidr_field_model_heatmaps `
  --metrics .\data\test\fd_pidr_field_model_metrics.csv `
  --diffusion-ckpt .\checkpoints\final\fd\pidr_diffusion_fd.pth `
  --refiner-ckpt .\checkpoints\final\fd\pidr_refiner_fd.pth `
  --norm-stats-mat .\data\field.mat `
  --mask-ratio 0.8 `
  --mask-strategy random `
  --mask-seed 0 `
  --diffusion-base-channels 32 `
  --sample-stride 1 `
  --sampling-method ddpm `
  --cache-postprocess none `
  --interp-weight 0 `
  --output-input-blend 0 `
  --batch-size 16 `
  --plot-channel 0 `
  --max-plots 50
```

### Evaluate Test Set

```powershell
python -m scripts.eval.evaluate_pidr `
  --data-path .\data\field.mat `
  --diffusion-ckpt .\checkpoints\final\fd\pidr_diffusion_fd.pth `
  --refiner-ckpt .\checkpoints\final\fd\pidr_refiner_fd.pth `
  --mode test `
  --split-ratio 0.9 `
  --split-seed 42 `
  --mask-ratio 0.8 `
  --mask-strategy random `
  --mask-seed 0 `
  --diffusion-base-channels 32 `
  --sample-stride 1 `
  --sampling-method ddpm `
  --cache-postprocess none `
  --interp-weight 0 `
  --output-input-blend 0 `
  --batch-size 64 `
  --eval-fraction 1.0 `
  --max-plots 0 `
  --output-dir .\results\eval\fd_test
```

### Evaluate Train Set

```powershell
python -m scripts.eval.evaluate_pidr `
  --data-path .\data\field.mat `
  --diffusion-ckpt .\checkpoints\final\fd\pidr_diffusion_fd.pth `
  --refiner-ckpt .\checkpoints\final\fd\pidr_refiner_fd.pth `
  --mode train `
  --split-ratio 0.9 `
  --split-seed 42 `
  --mask-ratio 0.8 `
  --mask-strategy random `
  --mask-seed 12345 `
  --diffusion-base-channels 32 `
  --sample-stride 1 `
  --sampling-method ddpm `
  --cache-postprocess none `
  --interp-weight 0 `
  --output-input-blend 0 `
  --batch-size 64 `
  --eval-fraction 1.0 `
  --max-plots 0 `
  --output-dir .\results\eval\fd_train
```

## SR Model

### Metadata

- Train dataset: `.\data\sr.mat`
- Test/generation input example: `.\data\test\sr.mat`
- Diffusion checkpoint: `.\checkpoints\final\sr\pidr_diffusion_sr.pth`
- Refiner checkpoint: `.\checkpoints\final\sr\pidr_refiner_sr.pth`
- Refiner format: plain `state_dict`
- Split ratio: `0.9`
- Split seed: `42`
- Mask ratio: `0.8`
- Mask strategy: `random`
- Train mask seed: `12345`
- Eval mask seed: `0`
- Diffusion base channels: `64`
- Sampling method: `ddim_x0`
- Sample stride: `20`
- Cache postprocess: `none`
- Interp weight: `0.0`
- Output input blend: `0.7`
- Train SSIM, Gain/Real/Imag/min: `0.956162 / 0.958574 / 0.961929 / 0.956162`
- Test SSIM, Gain/Real/Imag/min: `0.953883 / 0.956870 / 0.959928 / 0.953883`

### Generate Channel Maps

```powershell
python -m scripts.ma.generate_channel_maps `
  --input .\data\test\sr.mat `
  --output .\data\test\sr_pidr_sr_model_batch.mat `
  --heatmap-dir .\data\test\sr_pidr_sr_model_heatmaps `
  --metrics .\data\test\sr_pidr_sr_model_metrics.csv `
  --diffusion-ckpt .\checkpoints\final\sr\pidr_diffusion_sr.pth `
  --refiner-ckpt .\checkpoints\final\sr\pidr_refiner_sr.pth `
  --norm-stats-mat .\data\sr.mat `
  --mask-ratio 0.8 `
  --mask-strategy random `
  --mask-seed 0 `
  --diffusion-base-channels 64 `
  --sample-stride 20 `
  --sampling-method ddim_x0 `
  --cache-postprocess none `
  --interp-weight 0 `
  --output-input-blend 0.7 `
  --batch-size 16 `
  --plot-channel 0 `
  --max-plots 50
```

### Evaluate Test Set

```powershell
python -m scripts.eval.evaluate_pidr `
  --data-path .\data\sr.mat `
  --diffusion-ckpt .\checkpoints\final\sr\pidr_diffusion_sr.pth `
  --refiner-ckpt .\checkpoints\final\sr\pidr_refiner_sr.pth `
  --mode test `
  --split-ratio 0.9 `
  --split-seed 42 `
  --mask-ratio 0.8 `
  --mask-strategy random `
  --mask-seed 0 `
  --diffusion-base-channels 64 `
  --sample-stride 20 `
  --sampling-method ddim_x0 `
  --cache-postprocess none `
  --interp-weight 0 `
  --output-input-blend 0.7 `
  --batch-size 64 `
  --eval-fraction 1.0 `
  --max-plots 0 `
  --output-dir .\results\eval\sr_test
```

### Evaluate Train Set

```powershell
python -m scripts.eval.evaluate_pidr `
  --data-path .\data\sr.mat `
  --diffusion-ckpt .\checkpoints\final\sr\pidr_diffusion_sr.pth `
  --refiner-ckpt .\checkpoints\final\sr\pidr_refiner_sr.pth `
  --mode train `
  --split-ratio 0.9 `
  --split-seed 42 `
  --mask-ratio 0.8 `
  --mask-strategy random `
  --mask-seed 12345 `
  --diffusion-base-channels 64 `
  --sample-stride 20 `
  --sampling-method ddim_x0 `
  --cache-postprocess none `
  --interp-weight 0 `
  --output-input-blend 0.7 `
  --batch-size 64 `
  --eval-fraction 1.0 `
  --max-plots 0 `
  --output-dir .\results\eval\sr_train
```

## Multi-User Movable Antenna Position Optimization

This task uses `.\data\test\sr.mat` as the ground-truth multi-user channel map.
The file contains three samples, and each sample is treated as one user.
The PIDR-generated multi-user channel map is `.\data\test\sr_pidr_batch.mat`.

The optimization objective is the sum spectral efficiency:

```text
sum_k log2(1 + SINR_k)
```

The unit is `bps/Hz`; no bandwidth multiplier is used.
The plot output is saved as PDF and shows SINR, not SNR.

```powershell
python -m scripts.ma.ma_multi_user `
  --gt-path .\data\test\sr.mat `
  --pidr-path .\data\test\sr_pidr_batch.mat `
  --n-antennas 8 `
  --total-power 1.0 `
  --noise-power-dbm -70 `
  --min-distance-ports 6 `
  --candidate-pool-size 256 `
  --output-dir .\results\ma_multi_user\sr
```

Outputs:

```text
.\results\ma_multi_user\sr\ma_multiuser_optimization_results.csv
.\results\ma_multi_user\sr\ma_multiuser_comparison_*.pdf
```

For the FD and EL panels, swap `--gt-path`/`--pidr-path` for `.\data\test\fd.mat` /
`.\data\test\fd_pidr_field_model_batch.mat` and `.\data\test\el.mat` / `.\data\test\el_pidr_batch.mat`.

## Single-User Movable Antenna Position Optimization

```powershell
python -m scripts.ma.ma_single_user `
  --gt-path .\data\test\sr.mat `
  --pidr-path .\data\test\sr_pidr_batch.mat `
  --user-index 2 `
  --n-antennas 8 `
  --total-power 1 `
  --noise-power-dbm -70 `
  --min-distance-ports 6 `
  --seed 42 `
  --output-dir .\results\ma_single_user_sr
```
