"""主扩散模型：条件残差 U-Net + 瓶颈自注意力。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pidr.config import Config


def _group_count(channels):
    """为 GroupNorm 选择一个可整除的分组数。"""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalPositionEmbeddings(nn.Module):
    """将离散时间步编码成连续的正余弦向量。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        factor = math.log(10000) / max(half_dim - 1, 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -factor)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class ResidualBlock(nn.Module):
    """带时间条件注入的残差卷积块。"""

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time_emb):
        residual = self.skip(x)
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + residual


class SpatialSelfAttention(nn.Module):
    """在瓶颈层显式建模全局空间相关性。"""

    def __init__(self, channels, heads):
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        h = self.norm(x).reshape(batch, channels, height * width).permute(0, 2, 1)
        attn_out, _ = self.attn(h, h, h)
        h = h + attn_out
        h = h + self.ff(h)
        return h.permute(0, 2, 1).reshape(batch, channels, height, width)


class Downsample(nn.Module):
    """把空间分辨率减半，同时提升通道数。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.proj(x)


class Upsample(nn.Module):
    """把空间分辨率恢复到上一层。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, target_hw):
        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return self.proj(x)


class OutputHead(nn.Module):
    """输出头，负责将特征映射回噪声预测。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(_group_count(in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class CSIDiffusionPhysicsModel(nn.Module):
    """用于 CSI 缺失区域重建的条件扩散主干网络。"""

    def __init__(self):
        super().__init__()
        base_channels = Config.BASE_CHANNELS
        time_emb_dim = Config.TIME_EMB_DIM
        input_channels = Config.IMG_CHANNELS * 2 + 1

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.input_proj = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)

        self.enc1 = ResidualBlock(base_channels, base_channels, time_emb_dim, Config.MODEL_DROPOUT)
        self.down1 = Downsample(base_channels, base_channels * 2)

        self.enc2 = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim, Config.MODEL_DROPOUT)
        self.down2 = Downsample(base_channels * 2, base_channels * 4)

        self.mid1 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, Config.MODEL_DROPOUT)
        self.mid_attn = SpatialSelfAttention(base_channels * 4, Config.ATTENTION_HEADS)
        self.mid2 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, Config.MODEL_DROPOUT)

        self.up2 = Upsample(base_channels * 4, base_channels * 2)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim, Config.MODEL_DROPOUT)

        self.up1 = Upsample(base_channels * 2, base_channels)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim, Config.MODEL_DROPOUT)

        self.output_head = OutputHead(base_channels, Config.IMG_CHANNELS)

    def forward(self, x_t, t, sparse_csi, mask):
        """前向传播时同时输入当前噪声图、条件稀疏图和观测掩码。"""
        time_emb = self.time_mlp(t)
        x = torch.cat([x_t, sparse_csi, mask], dim=1)

        x = self.input_proj(x)
        skip1 = self.enc1(x, time_emb)

        x = self.down1(skip1)
        skip2 = self.enc2(x, time_emb)

        x = self.down2(skip2)
        x = self.mid1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, time_emb)

        x = self.up2(x, skip2.shape[-2:])
        x = torch.cat([x, skip2], dim=1)
        x = self.dec2(x, time_emb)

        x = self.up1(x, skip1.shape[-2:])
        x = torch.cat([x, skip1], dim=1)
        x = self.dec1(x, time_emb)

        return self.output_head(x)
