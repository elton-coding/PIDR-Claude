"""Field-response (plane-wave) LMMSE layer: the physics module of the PIDR refiner.

Physical model
--------------
Within the small movement region of a movable antenna, the channel is the superposition of the
plane waves of the propagation paths (field-response model of movable-antenna channels):

    h(t) = sum_l g_l exp(j k_l . t),        |k_l| <= k0 = 2*pi/lambda,

so every map satisfies the Helmholtz equation and its spatial spectrum lies on / inside the circle
|k| = k0 (on the SR data, 87 % of the energy sits in a narrow ring at 0.28-0.34 cycles per grid
step). With independent path gains, the spatial covariance is the Fourier transform of the path
power wavenumber spectrum P(k):

    E[h(t + d) h*(t)] = sum_l E|g_l|^2 exp(j k_l . d).

Layer
-----
For every map:
1. P(k) is estimated from the current full-map estimate (the diffusion output) as its periodogram,
   which identifies the dominant paths of this map, and is mixed with the dataset-level wavenumber
   spectrum (both normalised to unit power): P = beta * P_map + (1 - beta) * P_global.
2. The covariance K = F^-1{P} is a valid (positive semi-definite) plane-wave covariance.
3. The map is re-estimated from the measured positions only, by linear MMSE (kriging):
       h_hat = K[:, O] (K[O, O] + sigma^2 I)^-1 y_O,
   together with the posterior variance, which tells the network where the physics estimate is
   reliable.

The output (gain, real, imaginary in the normalised data space, plus a normalised log posterior
variance) is fed to the refiner as physics-driven input channels.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

GAIN_FLOOR_EPS = 1e-30


class FieldResponseLMMSE(nn.Module):
    """Per-map plane-wave LMMSE re-estimation from the measured positions.

    Args:
        height, width: grid size.
        norm_min, norm_max: per-channel (gain, real, imag) normalisation bounds of the dataset, used to
            map the normalised [-1, 1] tensors back to complex channel values.
        beta: weight of the per-map spectrum (1 = only the map's own periodogram).
        nugget: diagonal loading of K[O, O], relative to the mean prior power.
        global_psd: optional dataset-level wavenumber spectrum on the (2H, 2W) FFT grid.
    """

    out_channels = 4

    def __init__(
        self,
        height: int,
        width: int,
        norm_min: Sequence[float] = (0.0, 0.0, 0.0),
        norm_max: Sequence[float] = (1.0, 1.0, 1.0),
        beta: float = 0.95,
        nugget: float = 1e-3,
        global_psd: Optional[torch.Tensor] = None,
        iterations: int = 1,
    ):
        super().__init__()
        self.height, self.width = height, width
        # stored as buffers so that a saved refiner restores its calibrated physics layer
        self.register_buffer("iterations", torch.tensor(int(iterations)))
        self.register_buffer("beta", torch.tensor(float(beta)))
        self.register_buffer("nugget", torch.tensor(float(nugget)))
        self.register_buffer("norm_min", torch.as_tensor(norm_min, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("norm_max", torch.as_tensor(norm_max, dtype=torch.float32).view(1, 3, 1, 1))
        if global_psd is None:
            global_psd = torch.ones(2 * height, 2 * width)
        global_psd = torch.as_tensor(global_psd, dtype=torch.float32)
        self.register_buffer("global_psd", global_psd / global_psd.sum())

        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        self.register_buffer("pos_y", yy.flatten(), persistent=False)
        self.register_buffer("pos_x", xx.flatten(), persistent=False)

    # ------------------------------------------------------------------ helpers
    def _to_physical(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) / 2.0 * (self.norm_max - self.norm_min) + self.norm_min

    def _gain_to_norm(self, gain: torch.Tensor) -> torch.Tensor:
        lo, hi = self.norm_min[:, 0:1], self.norm_max[:, 0:1]
        return ((gain - lo) / (hi - lo)).clamp(0.0, 1.0) * 2.0 - 1.0

    def _reim_to_norm(self, v: torch.Tensor, ch: int) -> torch.Tensor:
        lo, hi = self.norm_min[:, ch:ch + 1], self.norm_max[:, ch:ch + 1]
        return ((v - lo) / (hi - lo)) * 2.0 - 1.0

    def lag_covariance(self, h: torch.Tensor) -> torch.Tensor:
        """Unit-power plane-wave covariance on the circular lag grid, (B, 2H, 2W) complex."""
        H, W = self.height, self.width
        spec = torch.fft.fft2(h, s=(2 * H, 2 * W)).abs() ** 2
        spec = spec / spec.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
        beta = self.beta.to(spec.dtype)
        psd = beta * spec + (1.0 - beta) * self.global_psd.to(spec.dtype)
        return torch.fft.ifft2(psd) * (4 * H * W)  # zero lag = 1

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward(self, estimate: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            estimate: (B, 3, H, W) normalised full-map estimate whose measured entries equal the
                measurements (e.g. the diffusion output after restoring measured values).
            mask: (B, 1, H, W) measurement mask.
        Returns:
            (B, 4, H, W): gain, real, imag of the LMMSE estimate (normalised as the data) and the
            log posterior variance (scaled to roughly [-1, 1]).
        """
        B, _, H, W = estimate.shape
        phys = self._to_physical(estimate.float()).double()
        scale = float(self.norm_max[0, 1].abs().clamp_min(1e-30))
        h = torch.complex(phys[:, 1], phys[:, 2]) / scale          # (B, H, W)
        m = mask[:, 0].flatten(1) > 0.5                              # (B, HW)
        counts = m.sum(1)
        n_obs = int(counts.max())
        # Observed indices, padded; padded slots are neutralised below.
        order = torch.argsort((~m).to(torch.int8), dim=1, stable=True)
        o_idx = order[:, :n_obs]                                     # (B, n_obs)
        valid = torch.arange(n_obs, device=m.device)[None] < counts[:, None]
        vf = valid.to(torch.float64)

        py, px = self.pos_y, self.pos_x
        oy, ox = py[o_idx], px[o_idx]                                # (B, n_obs)
        dy = (py[None, :, None] - oy[:, None, :]) % (2 * H)          # (B, HW, n_obs)
        dx = (px[None, :, None] - ox[:, None, :]) % (2 * W)
        b_idx = torch.arange(B, device=m.device)[:, None, None]
        eye = torch.eye(n_obs, device=m.device, dtype=torch.complex128)
        load = self.nugget.double() * vf[:, :, None] * eye + (1.0 - vf)[:, :, None] * eye
        y = torch.gather(h.flatten(1), 1, o_idx) * vf                 # (B, n_obs)

        current = h
        for _ in range(int(self.iterations)):
            R = self.lag_covariance(current)                         # (B, 2H, 2W)
            K_ao = R[b_idx, dy, dx] * vf[:, None, :]                 # (B, HW, n_obs)
            K_oo = torch.gather(K_ao, 1, o_idx[:, :, None].expand(-1, -1, n_obs))
            K_oo = K_oo * (vf[:, :, None] * vf[:, None, :]) + load
            L = torch.linalg.cholesky(K_oo)
            z = torch.cholesky_solve(y[..., None], L)                 # (B, n_obs, 1)
            h_hat = (K_ao @ z)[..., 0]                                # (B, HW)
            current = torch.where(m, h.flatten(1), h_hat).view(B, H, W)

        # posterior variance: 1 - diag(K_ao K_oo^-1 K_ao^H)
        V = torch.linalg.solve_triangular(L, K_ao.conj().transpose(1, 2), upper=False)  # (B, n_obs, HW)
        var = (1.0 - (V.abs() ** 2).sum(1)).clamp_min(1e-6)            # (B, HW)

        h_hat = h_hat.view(B, H, W) * scale
        re, im = h_hat.real, h_hat.imag
        gain = 5.0 * torch.log10(re * re + im * im + GAIN_FLOOR_EPS)
        out_gain = self._gain_to_norm(gain[:, None].float())
        out_re = self._reim_to_norm(re[:, None].float(), 1)
        out_im = self._reim_to_norm(im[:, None].float(), 2)
        logvar = (torch.log10(var.view(B, 1, H, W)) / 3.0 + 1.0).float().clamp(-1.0, 1.0)  # 1e-6..1 -> -1..1
        return torch.cat([out_gain, out_re, out_im, logvar], dim=1)


def estimate_global_psd(h: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Dataset-level wavenumber spectrum (unit power) from complex training maps (N, H, W)."""
    spec = torch.fft.fft2(h, s=(2 * height, 2 * width)).abs() ** 2
    spec = spec / spec.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    psd = spec.mean(0)
    return (psd / psd.sum()).float()


def wavenumber_ring(psd: torch.Tensor) -> dict:
    """Summary of a (2H, 2W) spectrum: peak radial frequency (cycles/grid step) and energy fraction."""
    Hh, Ww = psd.shape
    fy = torch.fft.fftfreq(Hh).view(-1, 1)
    fx = torch.fft.fftfreq(Ww).view(1, -1)
    r = torch.sqrt(fy ** 2 + fx ** 2)
    bins = torch.arange(0, 0.72, 0.02)
    hist = torch.stack([psd[(r >= a) & (r < a + 0.02)].sum() for a in bins])
    k = int(hist.argmax())
    return {"peak_cycles_per_step": float(bins[k] + 0.01), "k0_rad_per_step": float(2 * math.pi * (bins[k] + 0.01))}


def configure_field_response(
    layer: FieldResponseLMMSE,
    norm_min: Sequence[float],
    norm_max: Sequence[float],
    global_psd: Optional[torch.Tensor] = None,
    beta: Optional[float] = None,
    nugget: Optional[float] = None,
    iterations: Optional[int] = None,
) -> None:
    """Write dataset statistics / calibrated hyper-parameters into the layer's buffers."""
    dev = layer.norm_min.device
    layer.norm_min.copy_(torch.as_tensor(norm_min, dtype=torch.float32, device=dev).view(1, 3, 1, 1))
    layer.norm_max.copy_(torch.as_tensor(norm_max, dtype=torch.float32, device=dev).view(1, 3, 1, 1))
    if global_psd is not None:
        layer.global_psd.copy_((global_psd / global_psd.sum()).to(dev, torch.float32))
    if beta is not None:
        layer.beta.fill_(float(beta))
    if nugget is not None:
        layer.nugget.fill_(float(nugget))
    if iterations is not None:
        layer.iterations.fill_(int(iterations))


@torch.no_grad()
def apply_field_response(layer: FieldResponseLMMSE, estimates: torch.Tensor, masks: torch.Tensor, batch: int = 64) -> torch.Tensor:
    """Run the layer over a whole tensor of maps (CPU in, CPU out)."""
    dev = layer.norm_min.device
    outs = [layer(estimates[s:s + batch].to(dev), masks[s:s + batch].to(dev)).cpu() for s in range(0, estimates.shape[0], batch)]
    return torch.cat(outs)


def _gain_nmse_db(layer: FieldResponseLMMSE, pred_norm_gain: torch.Tensor, gt_norm_gain: torch.Tensor) -> float:
    lo, hi = float(layer.norm_min[0, 0, 0, 0]), float(layer.norm_max[0, 0, 0, 0])
    g = (gt_norm_gain.double() + 1) / 2 * (hi - lo) + lo
    p = (pred_norm_gain.double() + 1) / 2 * (hi - lo) + lo
    err = ((g - p) ** 2).flatten(1).sum(1) / (g ** 2).flatten(1).sum(1).clamp_min(1e-30)
    return float((10 * torch.log10(err.clamp_min(1e-30))).mean())


@torch.no_grad()
def calibrate_field_response(
    layer: FieldResponseLMMSE,
    estimates: torch.Tensor,
    masks: torch.Tensor,
    gt: torch.Tensor,
    betas: Sequence[float] = (1.0, 0.95, 0.9),
    nuggets: Sequence[float] = (1e-4, 1e-3, 1e-2),
    iterations: Sequence[int] = (1, 2),
    max_maps: int = 800,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Grid-search beta / nugget / iterations on (a subset of) the TRAINING maps by gain NMSE.

    `estimates`, `masks`, `gt` are normalised training tensors. The best setting is written into the
    layer and returned.
    """
    n = estimates.shape[0]
    if n > max_maps:
        g = torch.Generator().manual_seed(seed)
        sel = torch.randperm(n, generator=g)[:max_maps].sort().values
        estimates, masks, gt = estimates[sel], masks[sel], gt[sel]
    nmin, nmax = layer.norm_min.flatten().tolist(), layer.norm_max.flatten().tolist()
    best = None
    for it in iterations:
        for beta in betas:
            for nug in nuggets:
                configure_field_response(layer, nmin, nmax, beta=beta, nugget=nug, iterations=it)
                out = apply_field_response(layer, estimates, masks)
                gain = torch.where(masks[:, 0] > 0.5, gt[:, 0], out[:, 0])
                score = _gain_nmse_db(layer, gain, gt[:, 0])
                if verbose:
                    print(f"[FR-calib] iterations={it} beta={beta} nugget={nug:g}: gain NMSE {score:.3f} dB", flush=True)
                if best is None or score < best["gain_nmse_db"]:
                    best = {"beta": beta, "nugget": nug, "iterations": it, "gain_nmse_db": score}
    configure_field_response(layer, nmin, nmax, beta=best["beta"], nugget=best["nugget"], iterations=best["iterations"])
    best["input_gain_nmse_db"] = _gain_nmse_db(layer, estimates[:, 0], gt[:, 0])
    return best
