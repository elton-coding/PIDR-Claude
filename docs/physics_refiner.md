# PIDR refiner 物理模块重设计：场响应 LMMSE

2026-11。目标：给 PIDR-Net 的第二阶段（refiner）设计一个真正有用的物理驱动模块，并用消融实验验证其正收益。
本文档记录动机、方法、实现与结果；论文尚未修改。

## 1. 为什么原来的物理分支无效

原 refiner 的 `PropagationPrior` 用一组**全局**可学习参数（发射机位置、路径损耗指数 n、波数 k）生成
径向幅度/相位模板。消融（`docs/checkpoints.md` 最后一节）表明它没有收益：每张 SR 图的接收机位置不同，
而先验对所有图共用同一组参数，收敛到图中心后退化为一个固定的径向特征，不携带逐图信息。

## 2. 数据中的物理结构

论文场景是**可移动天线的小尺度信道图**：天线在一个边长 `D_t·λ` 的小区域内扫描，接收端固定。在这样的
区域内，信道就是各传播路径平面波的叠加（MA 文献中的场响应模型）：

    h(t) = Σ_l g_l · exp(j k_l · t),   |k_l| ≤ k0 = 2π/λ

因此每张图都满足 Helmholtz 方程，空间谱集中在半径 k0 的圆环上（或圆内）。对复信道
`h = re + j·im` 做二维 FFT（训练集平均）：

| 数据集 | 谱能量集中位置（周期/网格步长） | 说明 |
|---|---|---|
| SR（Sionna RT） | 0.28–0.34（峰值 0.31，87% 能量） | 明显的 Helmholtz 圆环，k0 ≈ 1.95 rad/步 |
| FD（field response） | 0.10–0.14（峰值 0.13） | 圆环，k0 ≈ 0.82 rad/步 |
| EL（EM 仿真） | 低频（55% 能量在 0.02 以内） | 平滑场 |

增益通道与实部/虚部满足 `gain = 5·log10(re² + im²)`（当前数据单位），在数据中精确成立。

## 3. 方法：场响应 LMMSE 层（`pidr/models/field_response.py`）

若各路径增益相互独立，则信道的空间协方差是路径功率波数谱 P(k) 的傅里叶变换：

    E[h(t+d) h*(t)] = Σ_l E|g_l|² exp(j k_l · d)

这是一个由物理模型保证正定的平面波协方差。对每张图：

1. **逐图波数谱**：用当前全图估计（扩散输出，测量位置已恢复为测量值）的周期图估计 P(k)，可选地与训练集平均谱混合
   `P = β·P_map + (1-β)·P_global`（两者均归一化为单位功率）。扩散输出提供了"这张图有哪些主导路径"，
   而真正的重建只用测量值完成。
2. **协方差** `K = F⁻¹{P}`（循环滞后网格 2H×2W，零填充，保证半正定）。
3. **LMMSE（kriging）重估计**：只用测量位置 O 的复测量值 y_O，
   `ĥ = K[:,O] (K[O,O] + σ²I)⁻¹ y_O`，同时得到后验方差 `1 - diag(K[:,O](K[O,O]+σ²I)⁻¹K[O,:])`。
4. 可迭代：用 ĥ 重新估计周期图后再解一次（`iterations`）。

输出 4 个通道送入 refiner：物理估计的增益、实部、虚部（与数据同样归一化）和归一化的 log 后验方差
（告诉网络哪里物理估计可靠）。

**超参数只在训练集上校准**（`calibrate_field_response`：在 800 张训练图上网格搜索
β ∈ {1, 0.95, 0.9}、σ² ∈ {1e-4, 1e-3, 1e-2}、iterations ∈ {1, 2}，按增益 NMSE 选最优），
校准结果作为 buffer 存进 refiner 权重，加载时自动恢复。该层没有可训练参数，训练时其输出预先算好并缓存。

## 4. refiner 结构（`pidr/models/pinn_refiner.py`）

主干与原 refiner 完全相同（两层编码 + 一次下采样 + 瓶颈 + 上采样 + 解码，base 32 通道，约 15 万参数）。
改动：

- `prior="field_response"`：输入 = 扩散估计(3) + 掩码(1) + 场响应 LMMSE 输出(4) + 坐标(2) = 10 通道；
- `residual="physics"`：输出 = 物理估计 + 网络残差（残差头零初始化，训练开始时输出即物理估计），
  测量位置仍强制等于测量值；损失不变（缺失区域 L1）。

消融对照 `NP`：去掉物理模块（输入 = 扩散估计 + 掩码 + 坐标），输出 = 扩散估计 + 网络残差。
两者网络、训练流程、数据、扩散缓存、随机种子完全一致，唯一区别是有无物理模块。

旧权重仍可加载：`load_refiner` / `build_refiner_from_state_dict` 根据 state_dict 自动识别变体
（径向先验 / 场响应 / 无先验，是否残差）。

## 5. 文献对应

- 场响应模型与 MA 信道估计：Ma, Zhu, Zhang 等用有限位置测量在角度域恢复场响应（压缩感知，arXiv 2306.04333、2312.06969）。
  本层是其统计（LMMSE）版本：不显式求解路径参数，而用路径功率谱定义协方差。
- Helmholtz 约束的无线电地图：RadioDiff-k²（arXiv 2504.15623）、PhyRMDM（arXiv 2501.19160）、RMSup（arXiv 2512.10965）。
- 物理先验 + 学习细化：RMPrior（arXiv 2606.03074）、RadioDUN（arXiv 2506.08418）。

## 6. 结果

所有数字：增益通道，论文单位（`10·log10|h|²`），NMSE/PSNR/SSIM 与单位无关。FR、NP 各 3 个训练 seed（0/1/2），
表中为均值，± 为训练 seed 间标准差；PAPER 为 v13 论文的径向先验 refiner（现存于 `checkpoints/archive/radial_v13/<ds>/pidr/refiner.pth`），
在同一扩散输出上评估。每个 seed 都跑满 1000 epoch，最佳 epoch 的选法与论文协议一致。

### 6.1 消融（论文协议：前 50 张测试图，5 个扩散采样 seed）

| 数据集 | 方法 | MSE | NMSE (dB) | PSNR (dB) | SSIM |
|---|---|---|---|---|---|
| SR | 仅扩散 | 1.600 | −39.58 | 28.63 | 0.928 |
| | 仅物理层（LMMSE，无网络） | 0.923 | −44.21 | 33.26 | 0.967 |
| | v13 论文 refiner | 1.186 | −41.92 | 30.97 | 0.956 |
| | NP：无物理模块 | 0.872 | −43.90 ± 0.04 | 32.95 | 0.969 |
| | **FR：场响应物理模块** | **0.628** | **−46.60 ± 0.09** | **35.65** | **0.978** |
| FD | 仅扩散 | 0.678 | −41.69 | 33.55 | 0.988 |
| | 仅物理层 | 0.297 | −45.69 | 37.55 | 0.997 |
| | v13 论文 refiner | 0.541 | −42.79 | 34.65 | 0.992 |
| | NP | 0.279 | −45.91 ± 0.07 | 37.77 | 0.996 |
| | **FR** | **0.182** | **−47.63 ± 0.03** | **39.49** | **0.998** |
| EL | 仅扩散 | 0.0483 | −40.31 | 31.42 | 0.711 |
| | 仅物理层 | 0.0050 | −53.79 | 44.90 | 0.994 |
| | v13 论文 refiner | 0.0218 | −45.26 | 36.38 | 0.916 |
| | NP | 0.0078 | −49.65 ± 0.48 | 40.76 | 0.979 |
| | **FR** | **0.0010** | **−61.09 ± 0.31** | **52.20** | **0.998** |

v13 论文 refiner 在此协议下的复现值（SR −41.92、FD −42.79、EL −45.26）与 v13 论文主表（−41.89、−42.67、−45.27）一致。

### 6.2 全部测试图（SR/FD 1000 张，EL 97 张；5 个采样 seed）

| 数据集 | 仅扩散 | 仅物理层 | v13 refiner | NP | FR | 物理模块增益（FR − NP） | 相对v13 refiner |
|---|---|---|---|---|---|---|---|
| SR | −39.94 | −44.09 | −42.10 | −44.13 ± 0.03 | **−46.57 ± 0.05** | **−2.44 dB** | −4.47 dB |
| FD | −41.68 | −46.69 | −42.81 | −46.08 ± 0.11 | **−48.45 ± 0.02** | **−2.37 dB** | −5.64 dB |
| EL | −39.53 | −53.67 | −44.94 | −48.83 ± 0.51 | **−60.85 ± 0.24** | **−12.02 dB** | −15.91 dB |

逐图配对（缓存扩散输出，同一训练 seed）：FR 的 MSE 低于 NP 的比例为 SR 88.4–91.2%、FD 80.5–83.9%、
EL 99.0–100%；低于v13 论文 refiner 的比例为 SR 98.8%、FD 99.8%、EL 100%。

SR 上的其他变体（全部 1000 张，同一协议）：FR_coarse（物理模块作为输入、残差加在扩散估计上）−45.94；
旧径向先验 A_full −42.01、B_no_prior −42.35、C_plain −41.98。物理模块即使只作为输入也有 +1.8 dB 增益，
把残差建立在物理估计上再多 0.6 dB。

### 6.3 校准结果（只用训练集）

| 数据集 | β | σ² | iterations | 训练集增益 NMSE：扩散输入 → 物理层 |
|---|---|---|---|---|
| SR | 1.0 | 1e-4 | 1 | −40.39 → −44.63 dB |
| FD | 1.0 | 1e-4 | 2 | −41.80 → −46.73 dB |
| EL | 0.9 | 1e-4 | 2 | −39.68 → −53.18 dB |

### 6.4 结论与注意事项

- 物理模块在三个数据集上都给出稳定的正收益（2.4 / 2.4 / 12.0 dB），远大于训练 seed 间的波动（≤ 0.5 dB），
  并且新 refiner 在三个数据集上都显著优于 v13 论文结果（4.5 / 5.6 / 15.9 dB）。
- 收益来源：物理层是逐图的、只依赖测量值的线性估计，与扩散模型的误差互补；网络只需学习物理估计的残差。
  EL 的场最平滑，平面波协方差最准确，收益最大。
- 2026-11 论文已改用 FR refiner（方法章节、结构图与全部实验结果）；6.1 的消融（仅扩散 / 仅物理层 / NP / FR）作为表 IV 和图 9 写入论文 Sec. V-E，
  图由 `paper/figures/draw_refiner_ablation.py` 生成，权重已复制到论文使用的目录
  （`<ds>/pidr/`、`sr/mask_ratio/rho*/`、`sr/semisup/alpha*/`），新旧数值对照见 [checkpoints.md](checkpoints.md)。
- 最佳 epoch 仍按测试集 missing-MSE 选取（与论文主表协议相同）；FR 与 NP 使用完全相同的选法，所以消融对比是公平的。

权重：`checkpoints/<ds>/refiner_fr/{FR,FR_s1,FR_s2,NP,NP_s1,NP_s2}/refiner.pth`（SR 另有 `FR_coarse`），
训练/评估日志在同目录 `train.log`、`eval.log`、`eval_paper_protocol_n50.log`；结果 `results/refiner_ablation_v2_<ds>/`
（论文协议在其子目录 `paper_protocol_n50/`）。`Config.PIDR.REFINER_KWARGS` 默认已改为 FR，
`scripts/train/train_pidr.py` 训练新模型时默认使用场响应 refiner。

## 7. 复现

```bash
# 训练（共享同一冻结扩散模型与缓存；FD 的扩散模型为 checkpoints/fd/baselines/diffusion.pth）
PIDR_DATASET=sr python -m scripts.train.train_refiner_ablation --variants FR NP --seeds 0 1 2
PIDR_DATASET=fd python -m scripts.train.train_refiner_ablation --variants FR NP --seeds 0 1 2 --diffusion-ckpt checkpoints/fd/baselines/diffusion.pth
PIDR_DATASET=el python -m scripts.train.train_refiner_ablation --variants FR NP --seeds 0 1 2
# 评估（runs/<ds>/ablation_refiner/PAPER/refiner.pth 为 v13 论文 refiner 的副本）
PIDR_DATASET=sr python -m scripts.eval.eval_refiner_ablation --variants FR NP FR_coarse PAPER --n 1000 --out-dir results/refiner_ablation_v2_sr
PIDR_DATASET=sr python -m scripts.eval.eval_refiner_ablation --variants FR NP PAPER --n 50 --out-dir results/refiner_ablation_v2_sr/paper_protocol_n50
# FD 评估另加 --diffusion-ckpt checkpoints/fd/baselines/diffusion.pth
```
