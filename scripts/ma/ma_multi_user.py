"""
Multi-User Movable Antenna Position Optimization: PIDR vs Ground Truth.

This script extends scripts/ma/ma_single_user.py from a multi-antenna
single-user setting to a multi-antenna multi-user downlink setting.

Scenario:
  - One transmitter has N movable antennas.
  - K users share the same selected antenna positions.
  - User channels are read from a multi-sample channel-map .mat file.
  - Inter-user interference is modeled through MRT precoding.
  - The optimization objective is sum spectral efficiency:
        sum_k log2(1 + SINR_k), in bps/Hz

Compared approaches:
  1. GT-based selection: positions optimized on ground-truth channels.
  2. PIDR-based selection: positions optimized on PIDR-estimated channels,
     then evaluated on ground-truth channels.
  3. Fixed random selection: random positions satisfying spacing constraint.

Outputs:
  - Console summary.
  - CSV with per-user SINR/spectral-efficiency and per-antenna positions.
  - PDF figures showing selected positions and per-user SINR comparison.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# The .mat maps store the gain as 10*log10|h|; the paper (eq. (2)) and all other figures use
# 10*log10|h|^2. Plots show the paper's units (same constant as pidr.data.GAIN_TO_PAPER_UNITS).
GAIN_TO_PAPER_UNITS = 2.0


NOISE_POWER_DBM = -70
NOISE_POWER_LINEAR = 10 ** ((NOISE_POWER_DBM - 30) / 10)


def _extract_field_case_insensitive(mat_data: Dict[str, np.ndarray], field_name: str) -> np.ndarray | None:
    if field_name in mat_data:
        return np.asarray(mat_data[field_name])
    lower_map = {str(key).lower(): key for key in mat_data.keys()}
    key = lower_map.get(field_name.lower())
    if key is None:
        return None
    return np.asarray(mat_data[key])


def _to_user_first(array: np.ndarray, field_name: str) -> np.ndarray:
    """Convert a 2D/3D CSI field to [K, H, W]."""
    array = np.squeeze(np.asarray(array))
    if array.ndim == 2:
        return array[None, :, :].astype(np.float64)
    if array.ndim != 3:
        raise ValueError(f"{field_name} must be 2D or 3D, got shape {array.shape}")

    shape = tuple(int(dim) for dim in array.shape)
    if shape[0] == shape[1]:
        return np.moveaxis(array, -1, 0).astype(np.float64)
    if shape[1] == shape[2]:
        return array.astype(np.float64)

    spatial = sorted(shape, reverse=True)[:2]
    h, w = sorted(spatial)
    if shape[0] == h and shape[1] == w:
        return np.moveaxis(array, -1, 0).astype(np.float64)
    if shape[1] == h and shape[2] == w:
        return array.astype(np.float64)
    if shape[0] == h and shape[2] == w:
        return np.moveaxis(array, 1, 0).astype(np.float64)

    raise ValueError(f"Cannot infer user/spatial axes for {field_name} with shape {array.shape}")


def load_multiuser_channel(mat_path: str, max_users: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load gain maps and complex user channels.

    Returns:
        gain_db: [K, H, W] CSI_Gain in dB.
        complex_channel: [K, H, W] complex channel. Uses CSI_Real/CSI_Imag
            when available; otherwise uses gain-derived nonnegative amplitudes.
    """
    data = loadmat(mat_path)
    gain_raw = _extract_field_case_insensitive(data, "CSI_Gain")
    if gain_raw is None:
        available = ", ".join(sorted(k for k in data.keys() if not k.startswith("__")))
        raise KeyError(f"CSI_Gain not found in {mat_path}. Available fields: {available}")

    gain_db = _to_user_first(gain_raw, "CSI_Gain")

    real_raw = _extract_field_case_insensitive(data, "CSI_Real")
    imag_raw = _extract_field_case_insensitive(data, "CSI_Imag")
    if real_raw is not None and imag_raw is not None:
        real = _to_user_first(real_raw, "CSI_Real")
        imag = _to_user_first(imag_raw, "CSI_Imag")
        if real.shape != gain_db.shape or imag.shape != gain_db.shape:
            raise ValueError(
                f"CSI field shape mismatch in {mat_path}: "
                f"Gain={gain_db.shape}, Real={real.shape}, Imag={imag.shape}"
            )
        channel = real + 1j * imag
    else:
        # CSI_Gain_dB = 10 * log10(|h|), following the original script.
        channel = 10.0 ** (gain_db / 10.0)

    if max_users is not None:
        if max_users <= 0:
            raise ValueError(f"max_users must be positive, got {max_users}")
        gain_db = gain_db[:max_users]
        channel = channel[:max_users]

    if gain_db.shape[0] < 2:
        print("Warning: only one user loaded; SINR reduces to SNR because there is no inter-user interference.")
    return gain_db, channel


def load_both_multiuser_channels(
    gt_path: str,
    pidr_path: str,
    max_users: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt_gain, gt_channel = load_multiuser_channel(gt_path, max_users=max_users)
    pidr_gain, pidr_channel = load_multiuser_channel(pidr_path, max_users=max_users)
    if gt_gain.shape != pidr_gain.shape:
        raise ValueError(f"Shape mismatch: GT {gt_gain.shape} vs PIDR {pidr_gain.shape}")
    return gt_gain, gt_channel, pidr_gain, pidr_channel


def positions_are_valid(selected: List[Tuple[int, int]], candidate: Tuple[int, int], min_distance_ports: float) -> bool:
    return all(np.hypot(candidate[0] - r, candidate[1] - c) >= min_distance_ports for r, c in selected)


def compute_multiuser_sinr_and_rate(
    channel_maps: np.ndarray,
    positions: np.ndarray,
    total_power: float = 1.0,
    noise_power: float = NOISE_POWER_LINEAR,
    precoding: str = "mrt",
    precoder_channel_maps: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Compute per-user SINR and spectral efficiency for selected antenna positions.

    Args:
        channel_maps: Complex channel maps [K, H, W] used for rate/SINR evaluation.
        positions: Antenna positions [N, 2].
        total_power: Total transmit power in Watts.
        noise_power: Normalized noise power in Watts.
        precoding: Currently supports "mrt".
        precoder_channel_maps: Optional complex channel maps [K, H, W] used to
            build MRT beam vectors. If omitted, uses channel_maps. This lets the
            PIDR case use PIDR-estimated channels for w while evaluating SINR on
            ground-truth channels.

    Returns:
        sinr: [K] linear SINR values.
        rates_bpshz: [K] user spectral efficiencies in bps/Hz.
        sum_rate_bpshz: Sum spectral efficiency in bps/Hz.
        signal_powers: [K] received signal powers.
        interference_powers: [K] inter-user interference powers.
    """
    if len(positions) == 0:
        raise ValueError("At least one antenna position is required.")
    if precoding.lower() != "mrt":
        raise ValueError(f"Unsupported precoding: {precoding}")
    if precoder_channel_maps is None:
        precoder_channel_maps = channel_maps
    if precoder_channel_maps.shape != channel_maps.shape:
        raise ValueError(
            f"Precoder/evaluation channel shape mismatch: "
            f"{precoder_channel_maps.shape} vs {channel_maps.shape}"
        )

    rows = positions[:, 0]
    cols = positions[:, 1]
    h_eval = channel_maps[:, rows, cols].astype(np.complex128)  # true/evaluation h, [K, N]
    h_precoder = precoder_channel_maps[:, rows, cols].astype(np.complex128)  # CSI used for w, [K, N]
    num_users = h_eval.shape[0]
    user_power = total_power / max(num_users, 1)

    norms = np.linalg.norm(h_precoder, axis=1, keepdims=True)
    beams = np.conj(h_precoder) / np.maximum(norms, 1e-30)  # MRT beam per user, [K, N]
    effective = h_eval @ beams.T  # [receiver user, beam user]
    power_matrix = user_power * np.abs(effective) ** 2

    signal_powers = np.diag(power_matrix)
    interference_powers = power_matrix.sum(axis=1) - signal_powers
    sinr = signal_powers / (interference_powers + noise_power + 1e-30)
    rates_bpshz = np.log2(1.0 + sinr)
    return sinr, rates_bpshz, float(rates_bpshz.sum()), signal_powers, interference_powers


def greedy_select_positions_sum_rate(
    channel_maps: np.ndarray,
    n_antennas: int,
    min_distance_ports: float,
    total_power: float,
    noise_power: float,
    candidate_pool_size: int | None = 256,
) -> np.ndarray:
    """Greedy antenna placement that maximizes multi-user sum rate."""
    _, height, width = channel_maps.shape
    selected: List[Tuple[int, int]] = []
    all_positions = [(r, c) for r in range(height) for c in range(width)]

    aggregate_power = np.mean(np.abs(channel_maps) ** 2, axis=0)
    ranked = sorted(all_positions, key=lambda pos: aggregate_power[pos[0], pos[1]], reverse=True)
    if candidate_pool_size is not None and candidate_pool_size > 0:
        ranked = ranked[: min(candidate_pool_size, len(ranked))]

    for _ in range(n_antennas):
        best_pos = None
        best_sum_rate = -np.inf
        for candidate in ranked:
            if candidate in selected:
                continue
            if not positions_are_valid(selected, candidate, min_distance_ports):
                continue
            trial = np.asarray(selected + [candidate], dtype=np.int32)
            _sinr, _rates, sum_rate, _sig, _int = compute_multiuser_sinr_and_rate(
                channel_maps, trial, total_power=total_power, noise_power=noise_power
            )
            if sum_rate > best_sum_rate:
                best_sum_rate = sum_rate
                best_pos = candidate

        if best_pos is None:
            break
        selected.append(best_pos)

    result = np.asarray(selected, dtype=np.int32)
    if len(result) < n_antennas:
        print(
            f"Warning: only {len(result)}/{n_antennas} positions could be selected "
            f"with min_distance_ports={min_distance_ports}."
        )
    return result


def select_fixed_positions(
    map_shape: Tuple[int, int],
    n_antennas: int,
    min_distance_ports: float,
    rng: np.random.Generator,
) -> np.ndarray:
    selected: List[Tuple[int, int]] = []
    candidates = [(r, c) for r in range(map_shape[0]) for c in range(map_shape[1])]
    rng.shuffle(candidates)
    for candidate in candidates:
        if positions_are_valid(selected, candidate, min_distance_ports):
            selected.append(candidate)
            if len(selected) == n_antennas:
                break
    result = np.asarray(selected, dtype=np.int32)
    if len(result) < n_antennas:
        print(
            f"Warning: only {len(result)}/{n_antennas} fixed positions could be selected "
            f"with min_distance_ports={min_distance_ports}."
        )
    return result


def db10(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(values, 1e-30))


def plot_multiuser_comparison(
    gt_gain: np.ndarray,
    pidr_gain: np.ndarray,
    gt_positions: np.ndarray,
    pidr_positions: np.ndarray,
    fixed_positions: np.ndarray,
    gt_sinr: np.ndarray,
    pidr_sinr: np.ndarray,
    fixed_sinr: np.ndarray,
    gt_sum_rate: float,
    pidr_sum_rate: float,
    fixed_sum_rate: float,
    save_path: str,
) -> None:
    """Save PDF comparison figure with selected positions and SINR bars."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_bar = fig.add_subplot(gs[1, :])

    gt_aggregate = np.mean(gt_gain, axis=0) * GAIN_TO_PAPER_UNITS
    pidr_aggregate = np.mean(pidr_gain, axis=0) * GAIN_TO_PAPER_UNITS
    vmin = min(float(gt_aggregate.min()), float(pidr_aggregate.min()))
    vmax = max(float(gt_aggregate.max()), float(pidr_aggregate.max()))

    panels = [
        ("GT-Based Optimization", gt_aggregate, gt_positions, gt_sum_rate, "o"),
        ("PIDR-Based Optimization", pidr_aggregate, pidr_positions, pidr_sum_rate, "s"),
        ("Fixed Random Antennas", gt_aggregate, fixed_positions, fixed_sum_rate, "^"),
    ]
    for ax, (title, image, positions, sum_rate, marker) in zip(axes, panels):
        im = ax.imshow(image, cmap="jet", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"{title}\nSum Rate={sum_rate:.3f} bps/Hz", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if len(positions) > 0:
            ax.scatter(
                positions[:, 1],
                positions[:, 0],
                marker=marker,
                s=120,
                edgecolors="white",
                facecolors="none",
                linewidths=2,
            )
            for i, (r, c) in enumerate(positions):
                ax.annotate(str(i), (c, r), color="white", fontsize=8, ha="center", va="center", fontweight="bold")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    users = np.arange(len(gt_sinr))
    width = 0.25
    ax_bar.bar(users - width, db10(gt_sinr), width, label="GT")
    ax_bar.bar(users, db10(pidr_sinr), width, label="PIDR")
    ax_bar.bar(users + width, db10(fixed_sinr), width, label="Fixed")
    ax_bar.set_title("Per-User SINR Comparison")
    ax_bar.set_xlabel("User Index")
    ax_bar.set_ylabel("SINR (dB)")
    ax_bar.set_xticks(users)
    ax_bar.legend()
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)


    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"PDF comparison figure saved to: {save_path}")


def save_results_csv(
    csv_path: str,
    methods: Dict[str, Dict[str, object]],
    total_power: float,
    noise_power: float,
    min_distance_ports: float,
) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Configuration"])
        writer.writerow(["Total_power_W", total_power])
        writer.writerow(["Normalized_noise_power_W", noise_power])
        writer.writerow(["Min_distance_ports", min_distance_ports])
        writer.writerow(["Evaluation_channel_for_all_methods", "GT"])
        writer.writerow(["PIDR_positions_and_MRT_beams_from", "PIDR"])
        writer.writerow([])

        writer.writerow(["Summary"])
        writer.writerow(["Method", "Sum_Rate_bps_per_Hz", "Mean_SINR_dB", "Min_SINR_dB"])
        for name, item in methods.items():
            sinr = item["sinr"]
            writer.writerow([name, f"{item['sum_rate']:.6f}", f"{db10(sinr).mean():.6f}", f"{db10(sinr).min():.6f}"])
        writer.writerow([])

        writer.writerow(["Per_User"])
        writer.writerow(["Method", "User", "SINR_linear", "SINR_dB", "Rate_bps_per_Hz", "Signal_Power_W", "Interference_Power_W"])
        for name, item in methods.items():
            for user_idx, (sinr, rate, signal, interference) in enumerate(
                zip(item["sinr"], item["rates"], item["signal"], item["interference"])
            ):
                writer.writerow([
                    name,
                    user_idx,
                    f"{sinr:.12e}",
                    f"{10.0 * np.log10(max(float(sinr), 1e-30)):.6f}",
                    f"{rate:.6f}",
                    f"{signal:.12e}",
                    f"{interference:.12e}",
                ])
        writer.writerow([])

        writer.writerow(["Antenna_Positions"])
        writer.writerow(["Method", "Antenna", "Row", "Col"])
        for name, item in methods.items():
            for antenna_idx, (row, col) in enumerate(item["positions"]):
                writer.writerow([name, antenna_idx, int(row), int(col)])
    print(f"Detailed CSV saved to: {csv_path}")


def run_optimization(
    gt_path: str,
    pidr_path: str,
    n_antennas: int,
    total_power: float,
    output_dir: str,
    min_distance_ports: float = 6.0,
    noise_power: float = NOISE_POWER_LINEAR,
    seed: int = 54,
    max_users: int | None = None,
    candidate_pool_size: int | None = 256,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 78)
    print("Multi-User Movable Antenna Position Optimization")
    print("=" * 78)
    print(f"  GT map                 : {gt_path}")
    print(f"  PIDR map               : {pidr_path}")
    print(f"  N antennas             : {n_antennas}")
    print(f"  Total power            : {total_power} W")
    print(f"  Normalized noise power : {noise_power:.3e} W")
    print(f"  Min antenna spacing    : {min_distance_ports} grid units")
    print(f"  Candidate pool size    : {candidate_pool_size}")
    print(f"  Random seed            : {seed}")
    print(f"  Output dir             : {output_dir}")
    print("-" * 78)

    gt_gain, gt_channel, pidr_gain, pidr_channel = load_both_multiuser_channels(gt_path, pidr_path, max_users=max_users)
    num_users, height, width = gt_gain.shape
    print(f"  Users                  : {num_users}")
    print(f"  Map size               : {height} x {width}")
    print()

    rng = np.random.default_rng(seed)
    print(">>> Selecting antenna positions by multi-user sum-rate objective ...")
    gt_positions = greedy_select_positions_sum_rate(
        gt_channel,
        n_antennas,
        min_distance_ports,
        total_power,
        noise_power,
        candidate_pool_size=candidate_pool_size,
    )
    pidr_positions = greedy_select_positions_sum_rate(
        pidr_channel,
        n_antennas,
        min_distance_ports,
        total_power,
        noise_power,
        candidate_pool_size=candidate_pool_size,
    )
    fixed_positions = select_fixed_positions((height, width), n_antennas, min_distance_ports, rng)
    print(f"  GT-based selected      : {len(gt_positions)}")
    print(f"  PIDR-based selected    : {len(pidr_positions)}")
    print(f"  Fixed selected         : {len(fixed_positions)}")
    print()
    print(">>> Evaluating SINR/rate on ground-truth channels ...")
    print("  GT    : positions from GT,   MRT beams from GT")
    print("  PIDR  : positions from PIDR, MRT beams from PIDR, evaluated on GT")
    print("  Fixed : random positions,    MRT beams from GT")
    print()

    methods: Dict[str, Dict[str, object]] = {}
    evaluation_plan = [
        ("GT", gt_positions, gt_channel),
        ("PIDR", pidr_positions, pidr_channel),
        ("Fixed", fixed_positions, gt_channel),
    ]
    for name, positions, precoder_channel in evaluation_plan:
        sinr, rates, sum_rate, signal, interference = compute_multiuser_sinr_and_rate(
            gt_channel,
            positions,
            total_power=total_power,
            noise_power=noise_power,
            precoder_channel_maps=precoder_channel,
        )
        methods[name] = {
            "positions": positions,
            "sinr": sinr,
            "rates": rates,
            "sum_rate": sum_rate,
            "signal": signal,
            "interference": interference,
        }

    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"{'Method':<10} {'Sum Rate (bps/Hz)':>18} {'Mean SINR (dB)':>16} {'Min SINR (dB)':>16}")
    for name, item in methods.items():
        sinr_db = db10(item["sinr"])
        print(f"{name:<10} {item['sum_rate']:>18.6f} {sinr_db.mean():>16.6f} {sinr_db.min():>16.6f}")

    print("\nPer-user SINR and spectral efficiency:")
    for name, item in methods.items():
        print(f"\n--- {name} Selection, evaluated on GT channel ---")
        for user_idx, (sinr, rate) in enumerate(zip(item["sinr"], item["rates"])):
            print(
                f"  User {user_idx:02d}: SINR={10*np.log10(max(float(sinr), 1e-30)):8.3f} dB, "
                f"Rate={rate:9.6f} bps/Hz"
            )

    csv_path = os.path.join(output_dir, "ma_multiuser_optimization_results.csv")
    save_results_csv(csv_path, methods, total_power, noise_power, min_distance_ports)

    plot_path = os.path.join(output_dir, f"ma_multiuser_comparison_n{n_antennas}_k{num_users}.pdf")
    plot_multiuser_comparison(
        gt_gain,
        pidr_gain,
        gt_positions,
        pidr_positions,
        fixed_positions,
        methods["GT"]["sinr"],
        methods["PIDR"]["sinr"],
        methods["Fixed"]["sinr"],
        methods["GT"]["sum_rate"],
        methods["PIDR"]["sum_rate"],
        methods["Fixed"]["sum_rate"],
        plot_path,
    )

    print("\n" + "=" * 78)
    print("Multi-user benchmark complete.")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-user MA position optimization: maximize sum spectral efficiency with SINR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gt-path", type=str, default="./data/test/sr.mat", help="Ground-truth multi-user channel map .mat")
    parser.add_argument("--pidr-path", type=str, default="./data/test/sr_pidr_batch.mat", help="PIDR-estimated multi-user channel map .mat")
    parser.add_argument("--n-antennas", type=int, default=4, help="Number of movable antennas")
    parser.add_argument("--total-power", type=float, default=1.0, help="Total transmit power in Watts")
    parser.add_argument(
        "--noise-power-dbm",
        type=float,
        default=NOISE_POWER_DBM,
        help="Normalized noise power in dBm. The rate is computed directly as log2(1+SINR) in bps/Hz.",
    )
    parser.add_argument("--output-dir", type=str, default="results/ma_multi_user/new", help="Output directory")
    parser.add_argument("--min-distance-ports", type=float, default=6.0, help="Minimum antenna spacing in grid units")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fixed baseline")
    parser.add_argument("--max-users", type=int, default=None, help="Limit number of users loaded from the .mat file")
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=256,
        help="Top aggregate-gain grid points considered by greedy search; use <=0 for all grid points.",
    )
    args = parser.parse_args()

    noise_power_linear = 10 ** ((args.noise_power_dbm - 30) / 10)
    candidate_pool_size = None if args.candidate_pool_size <= 0 else args.candidate_pool_size

    run_optimization(
        gt_path=args.gt_path,
        pidr_path=args.pidr_path,
        n_antennas=args.n_antennas,
        total_power=args.total_power,
        output_dir=args.output_dir,
        min_distance_ports=args.min_distance_ports,
        noise_power=noise_power_linear,
        seed=args.seed,
        max_users=args.max_users,
        candidate_pool_size=candidate_pool_size,
    )


if __name__ == "__main__":
    main()
