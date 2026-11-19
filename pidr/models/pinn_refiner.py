"""PINN 风格的 CSI 热力图重建网络。

该文件实现了一个“数据驱动 CNN + 可学习传播先验”的混合模型，目标是在稀疏观测条件下重建完整 CSI 场。
核心思想是：
1. 用轻量 CNN 学习局部/全局空间补全能力；
2. 用传播先验分支生成与物理传播相关的辅助特征（幅度与相位）；
3. 将稀疏输入、掩码、先验特征、坐标特征联合输入主干网络，提高在缺失区域的重建稳定性。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pidr.models.field_response import FieldResponseLMMSE


class ConvBNAct(nn.Module):
    """基础卷积块：Conv2d + BatchNorm2d + ReLU。

    设计目的：
    1. `Conv2d` 负责提取局部空间结构；
    2. `BatchNorm2d` 稳定训练分布、加速收敛；
    3. `ReLU` 引入非线性表达能力。

    该模块在编码器、瓶颈层、解码器中重复使用，保持结构统一。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PropagationPrior(nn.Module):
    """可学习传播先验分支。

    该分支不直接输出最终重建结果，而是根据可学习的传播参数生成“先验特征图”，
    作为主干 CNN 的辅助输入。它模拟了一个粗粒度传播模型，主要由以下参数决定：
    1. 发射机归一化坐标 `(tx_x, tx_y)`；
    2. 路径损耗指数 `n`（控制幅度随距离衰减）；
    3. 波数 `k`（控制相位随距离变化）。

    先验输出三通道：
    - 幅度先验 `amp_prior`
    - 相位实部先验 `re_prior`
    - 相位虚部先验 `im_prior`
    """

    def __init__(self):
        super().__init__()
        # 通过 raw 参数 + 约束激活函数 的方式学习物理参数，避免直接学习时数值越界。
        self.tx_x_raw = nn.Parameter(torch.tensor(0.0))
        self.tx_y_raw = nn.Parameter(torch.tensor(0.0))
        self.pathloss_exp_raw = nn.Parameter(torch.tensor(0.0))
        self.wave_number_raw = nn.Parameter(torch.tensor(1.5))

        # 幅度和相位的仿射校正项，用于提高先验分支的拟合灵活性。
        self.amp_scale = nn.Parameter(torch.tensor(1.0))
        self.amp_bias = nn.Parameter(torch.tensor(0.0))
        self.phase_bias = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _build_coords(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """构建归一化二维坐标网格。

        返回形状为 `[B, 2, H, W]`，其中：
        - 第 0 通道是 x 坐标（横向）；
        - 第 1 通道是 y 坐标（纵向）。
        """
        yy = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype).view(1, 1, height, 1)
        xx = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype).view(1, 1, 1, width)
        yy = yy.expand(batch_size, -1, -1, width)
        xx = xx.expand(batch_size, -1, height, -1)
        return torch.cat([xx, yy], dim=1)

    def get_pathloss_exponent(self) -> torch.Tensor:
        """返回路径损耗指数 n。

        用 `softplus + 1.0` 保证 n 始终为正且不至于过小，减少训练早期数值不稳定。
        """
        return F.softplus(self.pathloss_exp_raw) + 1.0

    def get_wave_number(self) -> torch.Tensor:
        """返回波数 k。

        用 `softplus + 1e-3` 保证 k 为正，避免后续相位构造时出现退化。
        """
        return F.softplus(self.wave_number_raw) + 1e-3

    def forward(self, sparse_csi: torch.Tensor):
        """根据输入尺寸生成传播先验特征。

        参数:
        - `sparse_csi`: 形状 `[B, 3, H, W]` 的稀疏 CSI 输入，仅用于获取 batch 和空间尺寸。

        返回:
        - `prior`: 形状 `[B, 3, H, W]`，依次为幅度先验、相位实部先验、相位虚部先验；
        - `coords`: 形状 `[B, 2, H, W]` 的归一化坐标。
        """
        batch, _channels, height, width = sparse_csi.shape
        coords = self._build_coords(batch, height, width, sparse_csi.device, sparse_csi.dtype)

        # 将 raw 坐标通过 sigmoid 约束到 [0, 1]，与归一化网格坐标保持一致。
        tx_x = torch.sigmoid(self.tx_x_raw)
        tx_y = torch.sigmoid(self.tx_y_raw)

        # 计算每个网格点到发射机位置的欧氏距离，加入小常数防止 0 距离数值问题。
        dx = coords[:, 0:1] - tx_x
        dy = coords[:, 1:2] - tx_y
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)

        n = self.get_pathloss_exponent()
        k = self.get_wave_number()

        # 幅度先验：近似路径损耗形式，距离越远幅度通常越弱。
        amp_prior = self.amp_scale * (-10.0 * n * torch.log10(dist + 1e-3)) + self.amp_bias

        # 相位先验：用径向传播相位 k*dist 构造余弦/正弦两通道。
        phase = k * dist + self.phase_bias
        re_prior = torch.cos(phase)
        im_prior = torch.sin(phase)

        prior = torch.cat([amp_prior, re_prior, im_prior], dim=1)
        return prior, coords


class PINNCSIReconstructor(nn.Module):
    """PINN 风格 CSI 重建网络主体。

    网络输入由四部分拼接而成：
    1. 稀疏 CSI（3 通道）；
    2. 观测掩码（1 通道）；
    3. 传播先验特征（3 通道）；
    4. 坐标特征（2 通道）。

    总输入通道数 = 3 + 1 + 3 + 2 = 9。
    整体结构是“轻量 U-Net 风格”：
    - 编码器提取浅层特征；
    - 下采样 + 瓶颈捕获更大感受野信息；
    - 上采样并与跳连特征融合恢复细节；
    - 最终输出 3 通道重建结果，并用 `tanh` 约束在 [-1, 1]。
    """

    PRIORS = ("radial", "field_response", "none")
    RESIDUALS = ("none", "coarse", "physics")

    def __init__(
        self,
        base_channels: int = 32,
        prior: str = "radial",
        use_coords: bool = True,
        residual: str = "none",
        height: int = 32,
        width: int = 32,
        physics_kwargs: Optional[dict] = None,
    ):
        """初始化网络结构。

        参数:
        - `base_channels`: 主干基础通道数，控制模型容量与计算量。
        - `prior`: 物理驱动输入分支。
            * "radial": 原论文的全局可学习径向模板（PropagationPrior，3 通道）；
            * "field_response": 平面波场响应 LMMSE（FieldResponseLMMSE，4 通道，逐图由测量值估计）；
            * "none": 不使用物理分支（消融）。
        - `use_coords`: 是否输入归一化坐标（2 通道）。
        - `residual`: 输出形式。"none" 直接输出；"coarse" 在输入估计上加残差；
          "physics" 在物理估计（field_response）上加残差。
        - `height`/`width`/`physics_kwargs`: field_response 分支的网格尺寸与校准参数。

        默认参数与原论文模型一致，已有权重的 state_dict 不受影响。
        """
        super().__init__()
        if prior not in self.PRIORS:
            raise ValueError(f"prior must be one of {self.PRIORS}, got {prior!r}")
        if residual not in self.RESIDUALS:
            raise ValueError(f"residual must be one of {self.RESIDUALS}, got {residual!r}")
        if residual == "physics" and prior != "field_response":
            raise ValueError("residual='physics' needs prior='field_response'")
        self.prior_type = prior
        self.use_prior = prior != "none"
        self.use_coords = use_coords
        self.residual = residual
        if prior == "radial":
            self.prior = PropagationPrior()
            prior_channels = 3
        elif prior == "field_response":
            self.prior = FieldResponseLMMSE(height, width, **(physics_kwargs or {}))
            prior_channels = FieldResponseLMMSE.out_channels
        else:
            prior_channels = 0
        if residual != "none":
            self.register_buffer("residual_mode", torch.tensor(self.RESIDUALS.index(residual)))

        in_channels = 3 + 1 + prior_channels + (2 if use_coords else 0)

        # 编码端：先进行两层同分辨率特征提取。
        self.enc = nn.Sequential(
            ConvBNAct(in_channels, base_channels),
            ConvBNAct(base_channels, base_channels),
        )
        # 下采样：分辨率减半、通道翻倍。
        self.down = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        # 瓶颈层：在低分辨率上继续提炼特征，扩大有效感受野。
        self.bottleneck = nn.Sequential(
            ConvBNAct(base_channels * 2, base_channels * 2),
            ConvBNAct(base_channels * 2, base_channels * 2),
        )
        # 上采样后先做通道投影，便于与 skip 特征拼接。
        self.up_proj = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        # 解码端：融合上采样特征与跳连特征，恢复空间细节。
        self.dec = nn.Sequential(
            ConvBNAct(base_channels * 2, base_channels),
            ConvBNAct(base_channels, base_channels),
        )

        # 输出 3 通道（对应 CSI 三个分量）。
        self.final_conv = nn.Conv2d(base_channels, 3, kernel_size=1)
        if residual != "none":
            # residual output: start exactly at the base estimate
            nn.init.zeros_(self.final_conv.weight)
            nn.init.zeros_(self.final_conv.bias)
        self.tanh = nn.Tanh()

    def physics_features(self, sparse_csi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """field_response 分支的输出（无可训练参数，训练时可预先计算并缓存）。"""
        if self.prior_type != "field_response":
            raise RuntimeError("physics_features needs prior='field_response'")
        return self.prior(sparse_csi, mask)

    def forward(
        self,
        sparse_csi: torch.Tensor,
        mask: torch.Tensor,
        return_aux: bool = False,
        physics: Optional[torch.Tensor] = None,
    ):
        """前向传播。

        参数:
        - `sparse_csi`: 输入估计（PIDR 中为扩散重建，测量位置已恢复为测量值），形状 `[B, 3, H, W]`；
        - `mask`: 观测掩码，形状 `[B, 1, H, W]`；
        - `return_aux`: 为 True 时额外返回径向先验的中间量（仅 prior="radial"）；
        - `physics`: 预先计算的 field_response 特征 `[B, 4, H, W]`，为 None 时现场计算。
        """
        batch, _channels, height, width = sparse_csi.shape
        coords = PropagationPrior._build_coords(batch, height, width, sparse_csi.device, sparse_csi.dtype)
        prior_feat = None
        if self.prior_type == "radial":
            prior_feat, coords = self.prior(sparse_csi)
        elif self.prior_type == "field_response":
            prior_feat = physics if physics is not None else self.prior(sparse_csi, mask)
        features = [sparse_csi, mask]
        if prior_feat is not None:
            features.append(prior_feat)
        if self.use_coords:
            features.append(coords)
        x = torch.cat(features, dim=1)

        # 编码 + 下采样 + 瓶颈。
        skip = self.enc(x)
        x = self.down(skip)
        x = self.bottleneck(x)

        # 双线性上采样回到原分辨率，再与 skip 连接恢复细节。
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up_proj(x)
        x = torch.cat([x, skip], dim=1)
        x = self.dec(x)
        # 用 tanh 将输出限制在归一化范围 [-1, 1]。
        if self.residual == "none":
            pred = self.tanh(self.final_conv(x))
        else:
            base = sparse_csi if self.residual == "coarse" else prior_feat[:, :3]
            pred = (base + self.final_conv(x)).clamp(-1.0, 1.0)

        if not return_aux:
            return pred

        # 训练时可用的辅助量：先验特征、坐标、关键传播参数。
        if self.prior_type != "radial":
            raise RuntimeError("return_aux needs the radial propagation prior (prior='radial').")
        aux = {
            "prior_feat": prior_feat,
            "coords": coords,
            "pathloss_exp": self.prior.get_pathloss_exponent(),
            "wave_number": self.prior.get_wave_number(),
        }
        return pred, aux


def build_refiner_from_state_dict(state_dict: dict, base_channels: Optional[int] = None) -> PINNCSIReconstructor:
    """Construct the refiner variant that matches a saved state_dict (old and new checkpoints)."""
    if base_channels is None:
        base_channels = int(state_dict["enc.0.block.0.weight"].shape[0])
    in_ch = int(state_dict["enc.0.block.0.weight"].shape[1])
    if "prior.tx_x_raw" in state_dict:
        prior, prior_ch = "radial", 3
    elif "prior.global_psd" in state_dict:
        prior, prior_ch = "field_response", FieldResponseLMMSE.out_channels
    else:
        prior, prior_ch = "none", 0
    use_coords = in_ch - 4 - prior_ch == 2
    residual = "none"
    if "residual_mode" in state_dict:
        residual = PINNCSIReconstructor.RESIDUALS[int(state_dict["residual_mode"])]
    kwargs = {}
    if prior == "field_response":
        psd = state_dict["prior.global_psd"]
        kwargs = dict(height=psd.shape[0] // 2, width=psd.shape[1] // 2)
    model = PINNCSIReconstructor(base_channels, prior=prior, use_coords=use_coords, residual=residual, **kwargs)
    model.load_state_dict(state_dict)
    return model


def load_refiner(path: str, device="cpu") -> PINNCSIReconstructor:
    """Load any refiner checkpoint (radial prior, field-response prior or no prior)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    return build_refiner_from_state_dict(state).to(device).eval()
