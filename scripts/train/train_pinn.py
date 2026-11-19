"""Training script for the physics-informed CNN (PINN) CSI reconstructor."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from pidr.models.pinn_refiner import PINNCSIReconstructor
from pidr.config import Config
from pidr.data import CSIGridDataset


def to_physical(x_norm: torch.Tensor, g_min: torch.Tensor, g_max: torch.Tensor) -> torch.Tensor:
    """Map normalized [-1, 1] predictions back to physical range."""
    return ((x_norm + 1.0) / 2.0) * (g_max - g_min) + g_min


def missing_region_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    missing = 1.0 - mask
    denom = missing.sum().clamp_min(1.0)
    return (((prediction - target) ** 2) * missing).sum() / denom


def power_consistency_l1(prediction: torch.Tensor, g_min: torch.Tensor, g_max: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Enforce channel consistency in physical space:
    CSI_Gain_dB ~= 10*log10(|h|) = 5*log10(Re^2 + Im^2).
    """
    pred_phys = to_physical(prediction, g_min, g_max)
    p_db = pred_phys[:, 0:1]
    h_re = pred_phys[:, 1:2]
    h_im = pred_phys[:, 2:3]
    implied_p_db = 5.0 * torch.log10(h_re * h_re + h_im * h_im + 1e-15)

    missing = 1.0 - mask
    denom = missing.sum().clamp_min(1.0)
    return (torch.abs(p_db - implied_p_db) * missing).sum() / denom


def laplacian_2d(x: torch.Tensor) -> torch.Tensor:
    """Discrete Laplacian for each channel."""
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=x.shape[1])


def helmholtz_residual_l1(
    prediction: torch.Tensor,
    g_min: torch.Tensor,
    g_max: torch.Tensor,
    wave_number: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    PINN residual:
    ∇^2(Re) + k^2 Re = 0
    ∇^2(Im) + k^2 Im = 0
    """
    pred_phys = to_physical(prediction, g_min, g_max)
    h_re = pred_phys[:, 1:2]
    h_im = pred_phys[:, 2:3]

    lap_re = laplacian_2d(h_re)
    lap_im = laplacian_2d(h_im)

    k2 = wave_number * wave_number
    res_re = lap_re + k2 * h_re
    res_im = lap_im + k2 * h_im

    missing = 1.0 - mask
    denom = missing.sum().clamp_min(1.0)
    return ((torch.abs(res_re) + torch.abs(res_im)) * missing).sum() / denom


def propagation_param_reg(pathloss_exp: torch.Tensor, wave_number: torch.Tensor) -> torch.Tensor:
    """Softly keep learned propagation parameters in a plausible range."""
    reg = torch.zeros((), dtype=pathloss_exp.dtype, device=pathloss_exp.device)
    reg = reg + F.relu(1.2 - pathloss_exp) ** 2 + F.relu(pathloss_exp - 6.0) ** 2
    reg = reg + F.relu(0.1 - wave_number) ** 2 + F.relu(wave_number - 40.0) ** 2
    return reg


def ramp_weight(epoch: int, warmup_epochs: int, max_value: float) -> float:
    if warmup_epochs <= 0:
        return max_value
    alpha = min(1.0, float(epoch + 1) / float(warmup_epochs))
    return float(max_value) * alpha


def train_cnn_pinn():
    device = Config.DEVICE
    train_dataset = CSIGridDataset(Config.DATA_PATH, mode="train", split_ratio=Config.DATA_SPLIT_RATIO)
    test_dataset = CSIGridDataset(
        Config.DATA_PATH,
        mode="test",
        split_ratio=Config.DATA_SPLIT_RATIO,
        mask_seed=Config.EVAL_MASK_SEED,
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = PINNCSIReconstructor().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.PINN.WEIGHT_DECAY,
    )

    g_min, g_max = train_dataset.get_normalization_tensors(device)

    save_path = Config.PINN.SAVE_PATH
    os.makedirs(Config.PATHS.CHECKPOINT_DIR, exist_ok=True)

    # Physics terms are ramped up to avoid hurting early reconstruction learning.
    warmup_epochs = max(Config.PINN.MIN_WARMUP_EPOCHS, Config.NUM_EPOCHS // 5)
    max_lambda_power = Config.PINN.MAX_LAMBDA_POWER
    max_lambda_pde = Config.PINN.MAX_LAMBDA_PDE
    max_lambda_param = Config.PINN.MAX_LAMBDA_PARAM

    print(
        f"Start PINN-CNN training. train_samples={len(train_dataset)}, test_samples={len(test_dataset)}, "
        f"warmup_epochs={warmup_epochs}"
    )

    best_val_missing_mse = float("inf")

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        running_total = 0.0
        running_data = 0.0
        running_power = 0.0
        running_pde = 0.0

        w_power = ramp_weight(epoch, warmup_epochs, max_lambda_power)
        w_pde = ramp_weight(epoch, warmup_epochs, max_lambda_pde)
        w_param = ramp_weight(epoch, warmup_epochs, max_lambda_param)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.NUM_EPOCHS} [PINN-Train]")
        for batch in pbar:
            gt = batch["gt_csi"].to(device)
            sparse = batch["sparse_csi"].to(device)
            mask = batch["mask"].to(device)

            pred, aux = model(sparse, mask, return_aux=True)
            # Keep observed positions exact.
            pred = pred * (1.0 - mask) + sparse * mask

            loss_data = missing_region_mse(pred, gt, mask)
            loss_power = power_consistency_l1(pred, g_min, g_max, mask)
            loss_pde = helmholtz_residual_l1(pred, g_min, g_max, aux["wave_number"], mask)
            loss_param = propagation_param_reg(aux["pathloss_exp"], aux["wave_number"])

            loss = loss_data + w_power * loss_power + w_pde * loss_pde + w_param * loss_param

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
            optimizer.step()

            running_total += float(loss.item())
            running_data += float(loss_data.item())
            running_power += float(loss_power.item())
            running_pde += float(loss_pde.item())

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.5f}",
                    "data": f"{loss_data.item():.5f}",
                    "pow": f"{loss_power.item():.5f}",
                    "pde": f"{loss_pde.item():.5f}",
                }
            )

        avg_total = running_total / max(1, len(train_loader))
        avg_data = running_data / max(1, len(train_loader))
        avg_power = running_power / max(1, len(train_loader))
        avg_pde = running_pde / max(1, len(train_loader))

        model.eval()
        val_missing_mse = 0.0
        with torch.no_grad():
            for batch in test_loader:
                gt = batch["gt_csi"].to(device)
                sparse = batch["sparse_csi"].to(device)
                mask = batch["mask"].to(device)

                pred = model(sparse, mask)
                pred = pred * (1.0 - mask) + sparse * mask
                val_missing_mse += float(missing_region_mse(pred, gt, mask).item())

        avg_val_missing_mse = val_missing_mse / max(1, len(test_loader))
        pathloss = float(model.prior.get_pathloss_exponent().item())
        wave_num = float(model.prior.get_wave_number().item())

        print(
            f"Epoch {epoch + 1}: train_total={avg_total:.6f}, train_data={avg_data:.6f}, "
            f"train_power={avg_power:.6f}, train_pde={avg_pde:.6f}, "
            f"val_missing_mse={avg_val_missing_mse:.6f}, "
            f"w_power={w_power:.2e}, w_pde={w_pde:.2e}, w_param={w_param:.2e}, "
            f"pathloss_n={pathloss:.3f}, wave_k={wave_num:.3f}"
        )

        if avg_val_missing_mse < best_val_missing_mse:
            best_val_missing_mse = avg_val_missing_mse
            torch.save(model.state_dict(), save_path)
            print(f"New best PINN model saved to {save_path} (val_missing_mse={best_val_missing_mse:.6f})")

    print(f"Training complete. Best validation missing MSE: {best_val_missing_mse:.6f}")


if __name__ == "__main__":
    train_cnn_pinn()
