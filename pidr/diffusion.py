"""扩散过程管理器。"""

import math

import torch
import tqdm

from pidr.config import Config


class DiffusionManager:
    """封装前向加噪、反向采样与物理投影逻辑。"""

    def __init__(self, device, global_min=None, global_max=None):
        self.device = device
        self.timesteps = Config.TIMESTEPS
        self.beta = self._build_beta_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.global_min = None
        self.global_max = None
        if global_min is not None and global_max is not None:
            self.set_normalization_stats(global_min, global_max)

    def _build_beta_schedule(self):
        """根据配置构造噪声调度表。"""
        if Config.NOISE_SCHEDULE == "linear":
            return torch.linspace(Config.BETA_START, Config.BETA_END, self.timesteps)
        if Config.NOISE_SCHEDULE == "cosine":
            steps = self.timesteps + 1
            time_axis = torch.linspace(0, self.timesteps, steps)
            alphas_cumprod = torch.cos(
                ((time_axis / self.timesteps) + 0.008) / 1.008 * math.pi * 0.5
            ) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return betas.clamp(1e-5, 0.999)
        raise ValueError(f"Unsupported noise schedule: {Config.NOISE_SCHEDULE}")

    def set_normalization_stats(self, global_min, global_max):
        """记录当前数据集的归一化范围，用于物理投影和反归一化。"""
        self.global_min = torch.as_tensor(global_min, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.global_max = torch.as_tensor(global_max, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def noise_images(self, x, t, noise=None):
        """按照时间步 t 对输入图进行前向加噪。"""
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x) if noise is None else noise
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def predict_x0(self, x_t, t, predicted_noise):
        """根据当前噪声预测值反推出原始样本 x0。"""
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        x0_est = (x_t - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
        return x0_est.clamp(-1.0, 1.0)

    def project_to_physics(self, x_norm):
        """将归一化张量投影回满足功率约束的物理空间。"""
        if self.global_min is None or self.global_max is None:
            raise RuntimeError("Normalization stats must be set before calling project_to_physics().")

        x_phys = ((x_norm + 1.0) / 2.0) * (self.global_max - self.global_min) + self.global_min
        h_re, h_im = x_phys[:, 1:2], x_phys[:, 2:3]
        # CSI_Gain in this project is 10*log10(|h|), not 10*log10(|h|^2).
        x_phys[:, 0:1] = 5.0 * torch.log10(h_re ** 2 + h_im ** 2 + 1e-15)
        return ((x_phys - self.global_min) / (self.global_max - self.global_min + 1e-20)) * 2.0 - 1.0

    def sample(self, model, sparse_csi, mask):
        """执行条件扩散采样，并持续做数据一致性与物理投影约束。"""
        model.eval()
        batch_size = sparse_csi.shape[0]
        map_height, map_width = sparse_csi.shape[-2:]
        x = torch.randn((batch_size, 3, map_height, map_width), device=self.device)

        for step in tqdm.tqdm(reversed(range(1, self.timesteps)), desc="Physics-guided sampling"):
            t = (torch.ones(batch_size, device=self.device) * step).long()
            with torch.no_grad():
                predicted_noise = model(x, t, sparse_csi, mask)

                if step % Config.PHYSICS_GUIDANCE_INTERVAL == 0:
                    x0_est = self.predict_x0(x, t, predicted_noise)
                    x0_est = self.project_to_physics(x0_est)
                    x0_est = x0_est * (1 - mask) + sparse_csi * mask
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    predicted_noise = (x - torch.sqrt(alpha_hat) * x0_est) / torch.sqrt(1 - alpha_hat)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                noise = torch.randn_like(x) if step > 1 else 0
                x = 1 / torch.sqrt(alpha) * (
                    x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
                ) + torch.sqrt(beta) * noise

                x = x * (1 - mask) + sparse_csi * mask

        return x.clamp(-1.0, 1.0)
