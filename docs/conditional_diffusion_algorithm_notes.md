# 条件扩散模型算法流程说明

本文档整理了本次对论文中条件扩散模型部分的主要修改内容，并根据当前项目代码说明条件扩散阶段的实际算法流程。

## 一、本次主要修改内容

本次主要修改的是论文中 `Conditional Diffusion Stage` 相关内容，使其更符合当前代码实现，并使算法逻辑更加清晰。

主要调整如下：

1. 将条件扩散模型部分重新组织为四个小部分：
   - `Forward noising process`
   - `Conditional denoising network`
   - `Physics-guided reverse denoising`
   - `Diffusion training objective`

2. 保留并突出前向加噪公式。前向过程只描述如何从完整归一化 CSI map `\widetilde{\mat{X}}` 得到带噪声样本 `\mat{x}_t`。

3. 将 denoising network 的输入关系讲清楚。当前代码中，去噪网络输入为当前噪声样本 `\mat{x}_t`、时间步 `t`、归一化稀疏观测 `\widetilde{\mat{Y}}` 和观测掩码 `\mat{M}`，输出为预测噪声 `\widehat{\vect{\epsilon}}_{\theta}`。

4. 重新梳理反向去噪流程。论文中明确说明反向采样从初始高斯样本 `\mat{x}_{T-1}` 开始，时间步为 `t=T-1,T-2,\ldots,1`，与代码中的 `reversed(range(1, self.timesteps))` 一致。

5. 新增了 physics guidance 后重新计算噪声的公式，用于对应代码中 `project_to_physics()` 后重新计算 `predicted_noise` 的步骤。

6. 明确 measurement consistency 和 physics projection 是 sampling-time operations，不属于 diffusion training loss。

7. 将 diffusion training objective 移到反向采样描述之后，单独说明训练阶段只进行 weighted masked noise prediction loss。

8. 同步修改 Algorithm 2 中的符号，将中间 clean estimate 记为 `\widehat{\mat{x}}_{0,t}`，避免和最终输出 `\mat{x}_0` 混淆。

## 二、当前代码对应的条件扩散模型流程

当前代码中条件扩散模型主要由三部分组成：前向加噪、反向采样和训练目标。

### 1. 前向加噪过程

给定完整归一化 CSI map `\widetilde{\mat{X}}`，在训练阶段随机采样时间步 `t`，并向 clean map 中加入高斯噪声：

```latex
\mat{x}_t
=
\sqrt{\bar{\alpha}_t}\,\widetilde{\mat{X}}
+
\sqrt{1-\bar{\alpha}_t}\,\vect{\epsilon},
\qquad
\vect{\epsilon}\sim\mathcal{N}(\mat{0},\mat{I}).
```

代码中对应的是：

```python
t = torch.randint(1, Config.TIMESTEPS, (gt.shape[0],), device=device)
x_t, noise = diffusion.noise_images(gt, t)
```

由于 `torch.randint(1, Config.TIMESTEPS, ...)` 的上界不包含 `Config.TIMESTEPS`，所以实际训练时间步为：

```text
t = 1, 2, ..., T-1
```

其中 `T = Config.TIMESTEPS`。

### 2. 条件去噪网络

去噪网络记为：

```latex
\mathcal{D}_{\theta}
```

其输入包括：

```text
当前噪声样本 x_t
时间步 t
归一化稀疏观测 \widetilde{Y}
观测掩码 M
```

输出为当前时间步的噪声预测：

```latex
\widehat{\vect{\epsilon}}_{\theta}
=
\mathcal{D}_{\theta}
(\mat{x}_t,t,\widetilde{\mat{Y}},\mat{M}).
```

代码中对应的是：

```python
predicted_noise = model(x, t, sparse_csi, mask)
```

网络内部将 `x_t`、`sparse_csi` 和 `mask` 在通道维度拼接，因此输入通道数为：

```text
3 + 3 + 1 = 7
```

其中 3 个通道分别对应 gain、real 和 imaginary。

### 3. 反向去噪采样过程

采样阶段从一个随机高斯噪声样本开始：

```latex
\mat{x}_{T-1}\sim\mathcal{N}(\mat{0},\mat{I})
```

代码中对应的是：

```python
x = torch.randn((batch_size, 3, map_height, map_width), device=self.device)
```

反向采样的时间步为：

```text
T-1, T-2, ..., 1
```

代码中对应的是：

```python
for step in reversed(range(1, self.timesteps)):
```

在每一个时间步，首先使用去噪网络预测噪声：

```python
predicted_noise = model(x, t, sparse_csi, mask)
```

然后根据预测噪声得到当前 clean estimate：

```latex
\widehat{\mat{x}}_{0,t}
=
\frac{
\mat{x}_t-\sqrt{1-\bar{\alpha}_t}\widehat{\vect{\epsilon}}_{\theta}
}
{\sqrt{\bar{\alpha}_t}}.
```

代码中对应的是：

```python
x0_est = diffusion.predict_x0(x, t, predicted_noise)
```

### 4. Physics guidance

在每隔固定反向步数时，代码会对 `x0_est` 进行 physics projection：

```python
if step % Config.PHYSICS_GUIDANCE_INTERVAL == 0:
    x0_est = diffusion.predict_x0(x, t, predicted_noise)
    x0_est = diffusion.project_to_physics(x0_est)
```

该操作的作用是重新根据 real 和 imaginary 通道计算 gain 通道，使三个通道之间保持一致。

论文中对应的 gain-channel update 为：

```latex
\widehat{X}^{\mathrm g}_{0,t}(u,v)
=
10\log_{10}
\left(
\sqrt{
\big(\widehat{X}^{\mathrm{re}}_{0,t}(u,v)\big)^2
+
\big(\widehat{X}^{\mathrm{im}}_{0,t}(u,v)\big)^2
}
\right).
```

需要注意的是，physics projection 不是在 denoising network 输入之前修正 `x_t`，而是在网络预测噪声之后，作用于由预测噪声得到的 clean estimate `\widehat{\mat{x}}_{0,t}`。

### 5. Measurement consistency

为了保证已观测位置不被扩散模型改写，代码中使用 observation consistency：

```python
x0_est = x0_est * (1 - mask) + sparse_csi * mask
```

论文中对应的投影算子为：

```latex
\Pi_{\mathrm{obs}}(\mat{Z})
=
(1-\mat{M})\od\mat{Z}
+
\mat{M}\od\widetilde{\mat{Y}}.
```

它的含义是：

```text
未观测位置：保留模型估计值
已观测位置：替换为真实测量值
```

### 6. 重新计算噪声

代码中在 physics projection 和 observation consistency 后，会重新计算用于 reverse update 的噪声：

```python
predicted_noise = (x - torch.sqrt(alpha_hat) * x0_est) / torch.sqrt(1 - alpha_hat)
```

论文中新增公式对应为：

```latex
\widetilde{\vect{\epsilon}}_{\theta}
=
\frac{
\mat{x}_t
-
\sqrt{\bar{\alpha}_t}
\Pi_{\mathrm{obs}}(\widehat{\mat{x}}^{\mathrm{phy}}_{0,t})
}
{\sqrt{1-\bar{\alpha}_t}}.
```

如果当前时间步不是 physics guidance step，则：

```latex
\widetilde{\vect{\epsilon}}_{\theta}
=
\widehat{\vect{\epsilon}}_{\theta}.
```

### 7. Reverse update

随后根据更新后的噪声估计执行 DDPM-style reverse update：

```latex
\boldsymbol{\mu}_{\theta}(\mat{x}_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
\mat{x}_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\widetilde{\vect{\epsilon}}_{\theta}
\right),
```

```latex
\mat{x}_{t-1}
=
\Pi_{\mathrm{obs}}
\left(
\boldsymbol{\mu}_{\theta}(\mat{x}_t,t)
+
\sigma_t\vect{z}
\right).
```

代码中对应的是：

```python
x = 1 / torch.sqrt(alpha) * (
    x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
) + torch.sqrt(beta) * noise

x = x * (1 - mask) + sparse_csi * mask
```

也就是说，每次 reverse update 后，代码都会再次对 `x_{t-1}` 做 observed-value reinsertion。

### 8. 扩散模型训练目标

训练阶段并不直接优化重建图像，而是训练网络预测加入的噪声。

为了让模型更关注缺失区域，论文中使用 spatially weighted noise prediction loss：

```latex
\omega(u,v)
=
M(u,v)
+
\lambda_{\mathrm{miss}}
\big(1-M(u,v)\big),
```

```latex
\mathcal{L}_{\mathrm{diff}}
=
\frac{1}{3HW}
\sum_{c=1}^{3}
\sum_{(u,v)\in\Omega}
\omega(u,v)
\big(
\widehat{\epsilon}_{\theta,c}(u,v)
-
\epsilon_c(u,v)
\big)^2.
```

代码中对应的是：

```python
loss = masked_noise_mse(pred_noise, noise, mask, Config.MISSING_REGION_WEIGHT)
```

需要强调的是：

```text
physics projection 和 observed-value reinsertion 只用于采样阶段，
不属于 diffusion training loss。
```

## 三、简化流程总结

当前条件扩散模型的实际流程可以概括为：

```text
训练阶段:
clean CSI map -> forward noising -> x_t
x_t, t, sparse CSI, mask -> U-Net -> predicted noise
predicted noise vs true noise -> weighted masked diffusion loss

采样阶段:
initial Gaussian noise x_{T-1}
for t = T-1, ..., 1:
    x_t, t, sparse CSI, mask -> U-Net -> predicted noise
    if physics guidance step:
        predicted noise -> clean estimate x_{0,t}
        physics projection on gain channel
        observed-value reinsertion on clean estimate
        recompute predicted noise
    DDPM reverse update -> x_{t-1}
    observed-value reinsertion on x_{t-1}
return x_0 as diffusion coarse reconstruction
```

## 四、需要注意的符号点

1. 当前代码实际使用的时间步是 `1` 到 `T-1`，不是 `0` 到 `T`。

2. 采样起点在论文中写作 `\mat{x}_{T-1}`，以对应代码中的 `reversed(range(1, self.timesteps))`。

3. `\widehat{\mat{x}}_{0,t}` 表示第 `t` 个反向时间步由当前噪声样本估计出的 clean map，不等同于最终输出 `\mat{x}_0`。

4. `\Pi_{\mathrm{obs}}(\cdot)` 表示观测一致性投影，即已测量位置使用真实观测值，未测量位置保留模型估计值。

5. physics projection 作用于 clean estimate `\widehat{\mat{x}}_{0,t}`，不是直接作用于进入 U-Net 之前的 `\mat{x}_t`。
