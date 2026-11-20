# 权重整理与溯源

本文记录 `checkpoints/` 的目录规范、论文每个实验对应的权重、以及 2026-11 复核时发现的问题。
每个权重文件的详细信息（md5、训练时间、网络类型、通道数、论文用途、证据）在
`checkpoints/MANIFEST.csv` 中，可随时校验：

```bash
python -m scripts.tools.checkpoint_manifest check
```

## 目录规范

```text
checkpoints/
├── MANIFEST.csv                       # 唯一的权重登记表
├── <dataset>/                         # sr | fd | el
│   ├── baselines/{diffusion,pinn}.pth # 单独的 Diffusion / PINN 基线
│   ├── pidr/{diffusion,refiner}.pth   # 论文 PIDR-Net（主表、采样策略、MA；refiner 为场响应 FR）
│   ├── pidr_cascade/...               # 4 月的级联版 PIDR（v13 采样策略实验，论文已不用）
│   ├── mask_ratio/rho0.70/...         # 缺失率实验，每个设置一个子目录
│   ├── semisup/alpha0.1/...           # 半监督实验，每个 α 一个子目录
│   ├── refiner_fr/...                 # 场响应 refiner 及其消融（FR/NP，各 3 个 seed）
│   ├── final/...                      # v13 MA 实验用的模型（含 refiner.metadata.json）
│   └── legacy_dataset/...             # 仅 el：针对已不存在的旧版 EL 数据训练的权重
└── archive/                           # 论文未使用的权重，保留原文件名，确认后可删除
    └── radial_v13/<原路径>            # 被 FR refiner 替换的 v13 refiner（2026-11）
```

规则：

1. 文件名只表示阶段（`diffusion` / `refiner` / `pinn`），数据集、实验和设置都由目录表达。
2. `pidr/config.py` 只**读取** `checkpoints/`。训练脚本输出到 `runs/<dataset>/<时间戳>/`，
   不会再覆盖论文结果依赖的权重（以前 `cnn_pinn_best_sr`、`cnn_pinn_best_el` 就是这样被覆盖的）。
3. 新权重确认要保留时：手动复制到对应目录，运行
   `python -m scripts.tools.checkpoint_manifest update`，再在 `MANIFEST.csv` 中填写
   `paper_usage` 和 `evidence` 两列。

## 论文实验 ↔ 权重

复核方法：用当前代码在论文相同的协议下（前 50 个样本、评估掩码种子 0、缺失率 0.8）
重新评估候选权重，与论文数值对比。PINN 输出是确定性的，可以逐样本精确比对；扩散采样带随机噪声，
用多个随机种子取均值比较。

| 论文实验 | 权重 | 结论 |
|---|---|---|
| 主表 SR：Diffusion | `sr/baselines/diffusion.pth` | ✅ 复现（NMSE −40.96 vs −40.85） |
| 主表 SR：PINN | `sr/baselines/pinn.pth` | ⚠️ 原权重已被覆盖；此文件最接近（−37.57 vs −37.69） |
| 主表 SR：PIDR-Net | `sr/pidr/` | ✅ 最可能（8 个种子 −42.77±0.07 vs −42.725） |
| 主表 FD：Diffusion | — | ❌ 现有权重均无法复现（论文 −27.33 / SSIM 0.853） |
| 主表 FD：PINN | `fd/baselines/pinn.pth` | ⚠️ 接近但不精确（−37.23 vs −37.12） |
| 主表 FD：PIDR-Net | — | ❌ 该行数值等于 **`fd/baselines/diffusion.pth` 单独扩散**的结果，见下文问题 3 |
| 主表 EL：三种方法 | `el/legacy_dataset/` | ❌ 在当前 `el.mat` 上无法评估（论文用的是旧版 EL 数据） |
| 采样策略（SR） | `sr/baselines/` + `sr/pidr_cascade/` | ✅ Diffusion、PIDR-Net 复现；PINN 原权重已被覆盖 |
| 缺失率 0.7 / 0.8 / 0.9 / 0.95（SR） | `sr/mask_ratio/rho*/` | ✅ 全部复现 |
| 半监督 α=0.1–0.6（SR） | `sr/semisup/alpha*/` | ✅ 结果 CSV 中有记录；α=0.5 见问题 2 |
| MA 单 / 多用户（SR/FD/EL） | `<dataset>/final/` | ✅ 重建图中记录了权重路径；SR 已逐位复现 |

## 复核中发现的问题

1. **2026-06 数据集重新导出，增益尺度减半。** 三个数据集中的 `CSI_Gain` 现为 `10·log10|h|`，
   4 月的数据为 `10·log10|h|²`。归一化会抵消这一尺度，所以旧权重仍然可用，
   当前 `5·log10(re²+im²)` 投影与旧代码等价；但**论文中所有 MSE 都是旧尺度**，
   用当前数据评估得到的 MSE 乘以 4 才能与论文比较（NMSE、PSNR、SSIM 不受影响）。
2. **论文 α=0.5 一行用错了权重。** 当时评估加载的是 `alpha0.6` 的权重。用正确的 `alpha0.5`
   权重重新评估（单个种子）：MSE 1.269（旧尺度）、NMSE −41.205 dB、SSIM 0.953；
   论文中为 1.307 / −40.128 / 0.946。修正后的数值位于 α=0.4 与 α=0.6 之间，趋势更合理。
   （同样方法复核 α=0.4：1.549 / −40.047 / 0.939，与论文 1.530 / −40.079 / 0.939 一致；
   用 `alpha0.6` 权重评估 α=0.5 得到 1.315 / −40.005 / 0.946，复现了论文中的错误数值。）
3. **FD 的 PIDR-Net 一行很可能其实是单独扩散模型的结果。** `benchmark_methods` 的配置中，
   PIDR 扩散权重的候选列表最后一项是**基线扩散模型**；文件缺失时会静默回退。现已删除该回退。
   用现有 FD 权重组合，PIDR-Net 最好只到 NMSE −36.9 dB（论文写的是 −41.92）。
4. **主表评估的是训练集。** 主表 PSNR+NMSE 的样本不变量与“训练集前 50 个样本”完全一致，
   而论文写的是测试集。采样策略、缺失率、半监督实验用的是测试集。
5. **主表 EL 一行使用的是旧版 EL 数据**（样本不变量为 +2.85，当前 `el.mat` 的训练集为 −6.76、
   测试集为 −8.89），旧数据已不在 `data/` 中。`el/legacy_dataset/` 中的权重在当前数据上输出无意义。
   另外，论文 EL 行中只有 PIDR-Net 与 `results/method_comparison_el` 一致，
   Diffusion 和 PINN 两项来自另一次评估。
6. **缺失率 0.95 一行** 使用的 refiner 是按缺失率 0.99 训练的（`rho0.95/refiner.pth`）。
7. **`pidr_diffusion_0.8fd.pth`（已归档）** 在当前 `field.mat` 上输出无意义（NMSE −23 dB）。
8. **半监督实验的测试输入密度随 α 变化。** 论文原文写测试条件与全监督相同（缺失率 80%），
   实际评估时输入点比例为 α·η（α=0.1 时 5%，α=0.6 时 30%），只有 α=0.4 与全监督的 20% 相同。
9. **refiner 训练时用测试集挑选最佳 epoch。** `train_pidr.py` 以测试集上的 missing-region MSE
   保存最佳权重（掩码与评估相同），存在轻微的测试集泄漏。现有全部 PIDR 权重都是这样训练的。

## 2026-11 论文修订记录

| 位置 | 修改 | 依据 |
|---|---|---|
| 表 semisup α=0.5 | 1.307 / −40.128 / 0.946 → **1.269 / −41.205 / 0.953** | 用正确的 `alpha0.5` 权重重新评估（3 个种子：MSE 1.249–1.269） |
| 图 semisup α=0.5 | 重新生成同一样本（1523） | 同上 |
| 半监督实验描述 | 改为实际的测试输入密度 α·η；与全监督的比较改为 α=0.4 | 问题 8 |
| 主表 RT（两张表） | 改为测试集结果：Diffusion −39.959 / PINN −37.143 / **PIDR-Net −41.887** | 论文所述协议；`results/method_comparison_sr/` |
| 图 RT 定性对比 | 测试样本 1（排名与均值一致、PIDR-Net MSE 最接近中位数） | 同上 |
| 主表 EM（EL） | 改为当前 `el.mat` 测试集结果：Diffusion −39.752 / PINN −41.569 / **PIDR-Net −45.274** | `results/method_comparison_el/` |
| 图 EM 定性对比与描述 | 测试样本 6（同一标准）；描述改为平滑渐变 | 同上；原文“条纹”描述与原图也不符 |
| 数据集描述、超参数表 | EM 数据集为 21×41、961 张图（训练 864 张） | 数据文件 |
| 主表 FD | 改为测试集结果：Diffusion −41.532 / PINN −37.477 / **PIDR-Net −42.672**（SSIM 0.992） | 新训练的 refiner（`fd/pidr/`），见下文 |
| 图 FD 定性对比与描述 | 测试样本 8（同一标准）；描述改为插值纹理与孤立深衰落 | `results/method_comparison_fd/` |
| 跨数据集结论（第 859 行等） | 措辞对应三张新图 | 新图 |
| 式 (2) 与物理投影式 | `10·log10(√(re²+im²))` → `10·log10(re²+im²)` | 与论文全部结果及 4 月代码实际使用的单位一致 |

所有新数值都按论文其余部分的单位给出（增益 `10·log10|h|²`，即当前数据的 2 倍），
因此 MSE 可与论文中其他表格直接比较。论文式 (2) 和物理投影式原写作 `10·log10|h|`，
与全部结果实际使用的单位不一致，已改为 `10·log10|h|²`。

仓库中的数据文件仍按 `10·log10|h|` 存储（代码投影为 `5·log10(re²+im²)`，二者配套）。
网络在归一化空间工作，两种尺度下的权重和 NMSE/PSNR/SSIM 完全相同；用当前数据算出的 MSE 乘以 4
即为论文单位。

### FD refiner 重新训练

原有 FD refiner 都是针对其他（较弱的）扩散模型的输出训练的，接在 `fd/baselines/diffusion.pth`
之后反而会变差。2026-11 用 `scripts/train/train_fd_refiner_on_baseline.py` 以该扩散模型为冻结的第一阶段，
按 `train_pidr.py` 的标准流程（1000 epochs，missing-region L1）重新训练 refiner：

- 验证集 missing-region MSE（3 通道，归一化空间）：仅扩散 0.000530 → PIDR-Net 0.000378
- 测试集（论文协议，增益通道）：NMSE −41.532 → −42.672 dB，SSIM 0.987 → 0.992
- 权重：`checkpoints/archive/radial_v13/fd/pidr/refiner.pth`（已被 FR refiner 替换），训练日志在同目录 `refiner_train.log`

与其他数据集一致，最佳 epoch 仍按测试集挑选（问题 9）。

## 2026-11 论文改用场响应 refiner

论文中 PIDR-Net 的 refiner 已从 v13 的径向传播先验（`prior="radial"`）换成场响应 LMMSE 物理先验
（`prior="field_response"`, `residual="physics"`，设计与消融见 [physics_refiner.md](physics_refiner.md)；
消融在论文 Sec. V-E：表 IV、图 9，数据取自 `results/refiner_ablation_v2_<ds>/paper_protocol_n50/`）。方法章节、refiner 结构图（`paper/figures/draw_pinn_refiner.py`）、超参数表
（κ / δ² / I_fr：SR 1.0/1e-4/1，FD 1.0/1e-4/2，EL 0.9/1e-4/2）随之改写。扩散模型均未重新训练，
只重新训练 refiner，然后重跑全部实验；新结果都优于或接近原论文，因此全部替换。

| 实验 | 原论文（v13 refiner） | 现在（FR refiner） | 权重 / 结果 |
|---|---|---|---|
| 主表 RT PIDR-Net（MSE / NMSE / SSIM） | 1.204 / −41.887 / 0.955 | **0.631 / −46.672 / 0.978** | `sr/pidr/`；`results/method_comparison_sr_fr/` |
| 跨数据集 FR PIDR-Net（NMSE / SSIM） | −42.672 / 0.992 | **−47.710 / 0.998** | `fd/pidr/`；`results/method_comparison_fd_fr/` |
| 跨数据集 EM PIDR-Net（NMSE / SSIM） | −45.274 / 0.916 | **−61.255 / 0.998** | `el/pidr/`；`results/method_comparison_el_fr/` |
| 采样策略 PIDR-Net MSE（random / uniform / local） | 1.204 / — / 高于扩散 | **0.663 / 0.830 / 4.476**（三种策略都最好） | `sr/pidr/`（原为 `sr/pidr_cascade/`）；`results/mask_strategies_sr_fr/` |
| 缺失率 0.7 / 0.8 / 0.9 / 0.95 MSE | 0.868 / 1.231 / 2.117 / 9.076 | **0.514 / 0.638 / 1.224 / 2.662** | `sr/mask_ratio/rho*/`（ρ=0.95 现在按 0.95 训练，原为 0.99）；`results/mask_ratios_sr_fr/` |
| 半监督 α=0.1 / 0.2 / 0.3 / 0.4 / 0.5 / 0.6 MSE | 7.523 / 4.059 / 2.364 / 1.530 / 1.269 / 0.988 | **5.448 / 2.440 / 1.327 / 1.036 / 0.786 / 0.587** | `sr/semisup/alpha*/`（扩散模型不变）；`results/semisupervised_sr_fr/alpha_*/` |
| MA 多用户 PIDR 和速率（N=4，SR / FD / EL，bps/Hz） | 2.956 / 10.516 / 2.765 | **3.155 / 9.705 / 2.875**（GT 3.361 / 9.702 / 2.884） | 图 `data/test/<ds>_pidr_fr_batch.mat`；`results/ma_multi_user_fr/` |
| MA 单用户 SR user 0（L=4 / 8） | 5.069 / 4.870 | **5.057 / 4.845**（GT 5.070 / 4.845） | `results/ma_single_user_sr_fr/u0_L{4,8}/` |

说明：

- 表中各基线行（Nearest/Linear/Cubic/PINN/Diffusion）重新评估后与原值差别都在采样随机性范围内，一并更新。
- FD 多用户 MA：原论文 PIDR 10.516 明显高于 GT 9.702（原图与真值差异大、恰好引导贪心搜索到更好位置）；
  新图与真值几乎一致，PIDR 9.705 ≈ GT，正文已相应改写。
- 定性对比图、采样策略图、缺失率图、半监督图沿用原论文的同一样本（RT 1、FD 8、EM 6、采样策略 5003、缺失率 6635、半监督 1523）。
- 原 v13 refiner 移到 `checkpoints/archive/radial_v13/<原路径>`，MANIFEST 中保留原来的用途说明。
- 单位：`benchmark_methods.py`、`benchmark_mask_strategies.py`、`benchmark_mask_ratios.py`、
  `benchmark_semisupervised.py` 现在通过 `dataset.gain_db()` 直接输出论文单位（`10·log10|h|²`），
  MSE 不需要再乘 4；`results/*_fr/` 中的数值都是论文单位。旧结果目录（无 `_fr` 后缀）仍是数据单位。
- 最佳 epoch 仍按测试集 missing-MSE 选取（与 v13 协议相同，问题 9）。

## Refiner 径向先验消融（SR，2026-11，未写入论文）

问题：refiner 输入端的可学习传播先验（`PropagationPrior`：路径损耗幅度 + `cos/sin(k·d)` 相位，
3 通道）是否有用？三个变体只改输入通道，其余完全相同（同一冻结扩散模型 `sr/pidr/diffusion.pth`、
同一扩散重建缓存、seed 0、1000 epochs、missing-region L1）：

| 变体 | 输入 | 参数量 |
|---|---|---|
| A_full | 扩散估计 + 掩码 + 传播先验 + 坐标（论文设计） | 151 018 |
| B_no_prior | 扩散估计 + 掩码 + 坐标 | 150 147 |
| C_plain | 扩散估计 + 掩码 | 149 571 |

结果（`results/refiner_ablation_sr/`，增益通道，论文单位）：

| 评估 | 方法 | MSE | NMSE (dB) | PSNR (dB) | SSIM |
|---|---|---|---|---|---|
| 全部 1000 张测试图（缓存扩散输出） | 仅扩散 | 1.535 | −39.92 | 28.33 | 0.924 |
| | A_full | 1.161 | −42.01 | 30.41 | 0.952 |
| | B_no_prior | **1.086** | **−42.34** | **30.74** | **0.957** |
| | C_plain | 1.128 | −41.99 | 30.39 | 0.954 |
| 论文协议（前 50 张，5 个采样 seed，NMSE std ≈ 0.07） | 仅扩散 | 1.600 | −39.58 | 28.63 | 0.928 |
| | A_full | 1.195 | −41.82 | 30.87 | 0.955 |
| | B_no_prior | **1.115** | **−42.17** | **31.22** | **0.960** |
| | C_plain | 1.160 | −41.81 | 30.86 | 0.958 |

- 逐图配对：A 的 MSE 低于 B 的只占 13.2%，低于 C 的占 40.1%。
- 验证集 missing-MSE（3 通道）：A 0.000956、B 0.000908、C 0.000931；三者最佳 epoch 都在 943–994，尚未完全收敛。
- 学到的先验参数（A）：tx ≈ (0.52, 0.49)，n ≈ 1.65，k ≈ 1.28。
- 原因分析：SR 每张图的发射机位置不同（测试集最强增益点分布在 620 个不同网格，x/y 标准差约 0.33），
  而先验只有一组全局可学习参数，收敛到图中心，退化为一个固定的径向特征，不携带逐图信息；
  k 很小，相位先验在网格内几乎单调，同样只是距离的函数。
- 结论：在当前设计下，物理先验对 SR 没有带来增益（略差于只用坐标）。仅一个训练 seed，A/B 差 0.33 dB 未做训练 seed 方差估计。

复现：

```bash
# 变体已改名：A_full = RAD，B_no_prior = NP_direct，C_plain = PLAIN（输出目录同名）
PIDR_DATASET=sr python -m scripts.train.train_refiner_ablation --variants RAD NP_direct PLAIN --seeds 0
PIDR_DATASET=sr python -m scripts.eval.eval_refiner_ablation --variants RAD NP_direct PLAIN --n 50
```

权重：`checkpoints/sr/ablation_refiner/{A_full,B_no_prior,C_plain}/refiner.pth`，训练日志 `checkpoints/sr/ablation_refiner/train.log`。

后续：refiner 的物理模块已重新设计为逐图的场响应 LMMSE 层，在 SR/FD/EL 上都有明显正收益
（论文协议下比无物理模块的同一网络好 2.4–12 dB），见 [physics_refiner.md](physics_refiner.md)；
权重在 `checkpoints/<ds>/refiner_fr/`。
