# PIDR-Net：基于扩散与物理信息细化的可移动天线 CSI 图重建

本仓库是论文 *PIDR-Net*（`paper/pidr_net.pdf`）的代码与论文工程。

**任务**：在可移动天线（MA）系统中，仅测得网格上少量位置的 CSI（缺失率默认 80%），
重建完整的三通道 CSI 图（增益 / 实部 / 虚部）。方法分两级：

1. **条件扩散阶段**：残差 U-Net + 瓶颈自注意力，以稀疏 CSI 与掩码为条件做反向扩散，
   每步回填观测值，并周期性地由实部/虚部重算增益（物理投影），得到粗重建；
2. **PINN 细化阶段**：结合可学习传播先验（传播中心、路径损耗指数、波数）与坐标特征，
   对粗重建做局部修正，最后回填观测值。

重建得到的 CSI 图进一步用于 MA 单用户 / 多用户天线位置优化。

## 目录结构

```text
.
├── pidr/                         # 核心库
│   ├── config.py                 # 全局配置（路径、数据、扩散、训练、各实验的 benchmark 配置）
│   ├── data.py                   # .mat 数据加载、归一化、掩码生成（CSIGridDataset）
│   ├── diffusion.py              # 噪声调度与物理投影（DiffusionManager）
│   └── models/
│       ├── diffusion_unet.py     # 条件扩散 U-Net（CSIDiffusionPhysicsModel）
│       └── pinn_refiner.py       # PINN 细化网络（PINNCSIReconstructor）
├── scripts/
│   ├── train/
│   │   ├── train_diffusion.py            # 单独训练扩散模型（Diffusion 基线）
│   │   ├── train_pinn.py                 # 单独训练 PINN（PINN 基线）
│   │   ├── train_pidr.py                 # PIDR-Net 两级训练：扩散 → 缓存重建 → 细化
│   │   ├── train_pidr_semisupervised.py  # 部分测量下的半监督训练
│   │   └── train_pidr_ssim_target.py     # 在已训扩散模型上训练细化器直至达到 SSIM 目标（最终模型）
│   ├── eval/
│   │   ├── benchmark_methods.py          # Diffusion / PINN / PIDR-Net 对比
│   │   ├── benchmark_mask_strategies.py  # 采样策略对比（含插值基线）
│   │   ├── benchmark_mask_ratios.py      # 不同缺失率
│   │   ├── benchmark_semisupervised.py   # 半监督模型评估
│   │   └── evaluate_pidr.py              # 通用的训练/测试集评估
│   └── ma/
│       ├── generate_channel_maps.py      # 用 PIDR-Net 批量重建测试 CSI 图
│       ├── ma_single_user.py             # 单用户 MA 位置优化
│       └── ma_multi_user.py              # 多用户 MA 位置优化（MRT，和速率）
├── results/                      # 论文中各实验对应的结果（仅跟踪 CSV 与 MA 图）
├── paper/                        # 论文 LaTeX 工程
│   ├── pidr_net.tex / pidr_net.pdf / references.bib
│   ├── figures/                  # 论文用图及绘图脚本 draw_*.py
│   └── slides/                   # 可编辑的示意图 PPT
├── docs/
│   ├── run_commands.md           # 最终模型的元数据与完整运行命令
│   ├── checkpoints.md            # 权重目录规范、论文实验与权重对照、复核发现的问题
│   └── conditional_diffusion_algorithm_notes.md
├── data/                         # 数据集（不入 git）
├── checkpoints/                  # 模型权重（不入 git），按 <数据集>/<实验>/<设置>/ 组织，见 MANIFEST.csv
└── runs/                         # 训练输出（不入 git），每次训练一个 <数据集>/<时间戳>/ 目录
```

## 环境

```bash
conda activate ma_diffusion
```

```bash
pip install -r requirements.txt
```

所有脚本都从**仓库根目录**以模块方式运行（配置中的 `./data`、`./checkpoints` 等均为相对根目录的路径）：

```bash
python -m scripts.train.train_pidr
```

## 数据与权重

`data/`、`checkpoints/`、`cache/`、`runs/` 体积较大，已在 `.gitignore` 中排除，需要单独拷贝。

| 数据集 | 文件 | 论文中的名称 |
|---|---|---|
| SR | `data/sr.mat` | Sionna RT dataset（主数据集） |
| FD | `data/field.mat` | Field response dataset |
| EL | `data/el.mat` | EM simulation dataset |

`data/test/` 中是 MA 实验用的 3 用户测试图及其 PIDR 重建结果（`*_pidr_*batch.mat`）。
权重按 `checkpoints/<数据集>/<实验>/<设置>/<阶段>.pth` 组织，每个文件的来源与用途登记在
`checkpoints/MANIFEST.csv` 中，规范和论文对照见 [docs/checkpoints.md](docs/checkpoints.md)。
训练脚本只写入 `runs/<数据集>/<时间戳>/`，不会覆盖 `checkpoints/` 中的权重；扩散重建缓存也放在该目录下。

## 论文实验与代码对照

| 论文章节 | 脚本 | 结果目录 |
|---|---|---|
| Results on Sionna RT / Cross dataset comparison | `scripts/eval/benchmark_methods.py`（测试集） | `results/method_comparison_{sr,fd,el}/` |
| Impact of different sampling strategies | `scripts/eval/benchmark_mask_strategies.py` | `results/mask_strategies_sr/` |
| Impact of different masking ratios | `scripts/eval/benchmark_mask_ratios.py` | `results/mask_ratios_sr/` |
| Impact of training measurement budget (α=0.1–0.6) | `scripts/train/train_pidr_semisupervised.py` + `scripts/eval/benchmark_semisupervised.py` | `results/semisupervised_sr/alpha_*/` |
| MA position optimization, single user (L=4, 8) | `scripts/ma/generate_channel_maps.py` + `scripts/ma/ma_single_user.py` | `results/ma_single_user_sr/` |
| MA position optimization, multi user (SR/FD/EL) | `scripts/ma/generate_channel_maps.py` + `scripts/ma/ma_multi_user.py` | `results/ma_multi_user/{sr,fd,el}/` |
| Refiner physics-module ablation (not in paper) | `scripts/train/train_refiner_ablation.py` + `scripts/eval/eval_refiner_ablation.py` | `results/refiner_ablation_v2_{sr,fd,el}/` |

说明：主表已于 2026-11 按论文所述协议（测试集）重新评估，旧结果及其问题见 docs/checkpoints.md。
各实验对应的权重及复核结论（包括 α=0.5 用错权重、FD 主表等问题）见 [docs/checkpoints.md](docs/checkpoints.md)。

refiner 物理模块重设计（场响应 LMMSE，未写入论文；新训练默认使用）见 [docs/physics_refiner.md](docs/physics_refiner.md)。

最终模型（用于 MA 实验）的完整推理/评估命令见 [docs/run_commands.md](docs/run_commands.md)。

## 数据集预设（`PIDR_DATASET`）

与数据集相关的设置（数据文件、网格尺寸、训练超参数、默认权重组合）由环境变量
`PIDR_DATASET` 在导入 `pidr/config.py` 时选择，取值 `sr` / `fd` / `el`（默认 `el`）：

```powershell
$env:PIDR_DATASET = "sr"; python -m scripts.train.train_pidr
```

```bash
PIDR_DATASET=sr python -m scripts.train.train_pidr
```

| 预设 | 数据 | 网格 | 训练样本 | lr / epochs | 默认权重（`checkpoints/` 下：基线 / PIDR-Net） |
|---|---|---|---|---|---|
| `sr` | `data/sr.mat` | 32×32 | 9000 | 2e-4 / 1000 | `sr/baselines/` / `sr/pidr/` |
| `fd` | `data/field.mat` | 32×32 | 9000 | 2e-4 / 1000 | `fd/baselines/` / `fd/pidr/` |
| `el` | `data/el.mat` | 21×41 | 864 | 2e-5 / 30000 | `el/baselines/` / `el/pidr/` |

`sr`、`fd` 使用论文超参数表；`el` 保留最后一次 EL 实验的设置（与整理前的默认值完全相同）。
带有独立命令行参数的脚本（`evaluate_pidr`、`generate_channel_maps`、`train_pidr_ssim_target`）
以命令行参数为准，见 [docs/run_commands.md](docs/run_commands.md)。

## 复现论文的训练设置

论文中的超参数（Table: Main experimental hyperparameters）对应 **SR 数据集**（即 `PIDR_DATASET=sr`）：

| 项目 | 值 |
|---|---|
| 数据 | `./data/sr.mat`，32×32，3 通道 |
| 划分 | 0.9/0.1，`split_seed=42`，训练样本上限 9000 |
| 掩码 | 缺失率 0.8，random；细化器训练掩码种子 12345，评估掩码种子 0 |
| 扩散 | T=1000，cosine 调度，每 25 步物理投影，base channels 64，time emb 256，dropout 0.1，4 头注意力 |
| 训练 | batch 32，AdamW，lr 2e-4，1000 epochs，缺失区域权重 3.0 |
| PINN | base channels 32 |

> **注意**：默认预设是 `el`，复现 SR 结果前务必设置 `PIDR_DATASET=sr`。
> `checkpoints/fd/final/diffusion.pth` 是 32 通道模型（base channels 32），不在 `fd` 预设中，
> 只能通过 `--diffusion-base-channels 32` 在最终模型命令里使用。
>
> 另外，2026-06 数据集重新导出后 `CSI_Gain` 从 `10*log10|h|²` 变为 `10*log10|h|`，
> `diffusion.py` 的增益投影相应从 `10*log10(re²+im²)` 改为 `5*log10(re²+im²)`，两者等价，旧权重可直接使用。
> 论文按 `10*log10|h|²` 定义增益（式 (2)），因此用当前数据评估得到的 MSE 需乘以 4 才能与论文比较；
> NMSE、PSNR、SSIM 不受影响。

## 编译论文

```bash
cd paper && latexmk -pdf pidr_net.tex
```
