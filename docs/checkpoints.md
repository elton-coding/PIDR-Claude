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
│   ├── pidr/{diffusion,refiner}.pth   # 该数据集主表中的 PIDR-Net
│   ├── pidr_cascade/...               # 4 月的级联版 PIDR（采样策略实验）
│   ├── mask_ratio/rho0.70/...         # 缺失率实验，每个设置一个子目录
│   ├── semisup/alpha0.1/...           # 半监督实验，每个 α 一个子目录
│   ├── final/...                      # MA 实验用的最终模型（含 refiner.metadata.json）
│   └── legacy_dataset/...             # 仅 el：针对已不存在的旧版 EL 数据训练的权重
└── archive/                           # 论文未使用的权重，保留原文件名，确认后可删除
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
