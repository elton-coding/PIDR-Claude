"""扩散模型训练入口。"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pidr.config import Config
from pidr.data import CSIGridDataset
from pidr.diffusion import DiffusionManager
from pidr.models.diffusion_unet import CSIDiffusionPhysicsModel

# python -m scripts.train.train_diffusion
def masked_mse(prediction, target, mask, missing_weight):
    """对缺失区域施加更高权重，使模型更关注真正需要补全的部分。"""
    missing_mask = 1.0 - mask
    weights = mask + missing_mask * missing_weight
    return ((prediction - target) ** 2 * weights).mean()


def masked_l1(prediction, target, mask):
    """仅在缺失区域统计 x0 重建误差。"""
    missing_mask = 1.0 - mask
    denominator = missing_mask.sum().clamp_min(1.0)
    return (torch.abs(prediction - target) * missing_mask).sum() / denominator


def masked_region_mse(prediction, target, mask):
    """用缺失区域的重建 MSE 作为模型优劣判断指标。"""
    missing_mask = 1.0 - mask
    denominator = missing_mask.sum().clamp_min(1.0)
    return (((prediction - target) ** 2) * missing_mask).sum() / denominator


def train():
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

    model = CSIDiffusionPhysicsModel().to(device)
    diffusion = DiffusionManager(device, train_dataset.global_min, train_dataset.global_max)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    mse = nn.MSELoss()

    g_min, g_max = train_dataset.get_normalization_tensors(device)
    best_test_mse = float("inf")

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss_epoch = 0.0
        train_recon_epoch = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")

        for data in pbar:
            gt = data["gt_csi"].to(device)
            sparse = data["sparse_csi"].to(device)
            mask = data["mask"].to(device)

            t = torch.randint(1, Config.TIMESTEPS, (gt.shape[0],), device=device)
            x_t, noise = diffusion.noise_images(gt, t)
            pred_noise = model(x_t, t, sparse, mask)
            x0_est = diffusion.predict_x0(x_t, t, pred_noise)

            loss_noise = masked_mse(pred_noise, noise, mask, Config.MISSING_REGION_WEIGHT)
            loss_x0 = masked_l1(x0_est, gt, mask)

            x0_phys = ((x0_est + 1.0) / 2.0) * (g_max - g_min) + g_min
            actual_p_db = x0_phys[:, 0]
            target_p_db = 5.0 * torch.log10(x0_phys[:, 1] ** 2 + x0_phys[:, 2] ** 2 + 1e-15)
            loss_phys = mse(actual_p_db, target_p_db)

            total_loss = loss_noise #+ Config.LAMBDA_X0 * loss_x0 #+ Config.LAMBDA_PHYS * loss_phys

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
            optimizer.step()

            train_loss_epoch += total_loss.item()
            train_recon_epoch += masked_region_mse(x0_est.detach(), gt, mask).item()
            pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "x0": f"{loss_x0.item():.4f}",
                    "phys": f"{loss_phys.item():.4f}",
                }
            )

        avg_train_loss = train_loss_epoch / len(train_loader)
        avg_train_mse = train_recon_epoch / len(train_loader)

        model.eval()
        test_mse_epoch = 0.0
        with torch.no_grad():
            for data in test_loader:
                gt = data["gt_csi"].to(device)
                sparse = data["sparse_csi"].to(device)
                mask = data["mask"].to(device)

                t = torch.randint(1, Config.TIMESTEPS, (gt.shape[0],), device=device)
                x_t, noise = diffusion.noise_images(gt, t)
                pred_noise = model(x_t, t, sparse, mask)
                x0_est = diffusion.predict_x0(x_t, t, pred_noise)
                test_mse_epoch += masked_region_mse(x0_est, gt, mask).item()

        avg_test_mse = test_mse_epoch / len(test_loader)
        print(
            f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, "
            f"train_missing_mse={avg_train_mse:.6f}, test_missing_mse={avg_test_mse:.6f}"
        )

        if avg_test_mse < best_test_mse:
            best_test_mse = avg_test_mse
            torch.save(model.state_dict(), Config.SAVE_PATH)
            print(f"New best model saved to {Config.SAVE_PATH} (test_missing_mse={best_test_mse:.6f})")


if __name__ == "__main__":
    train()
