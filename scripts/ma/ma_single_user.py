"""
MA Position Optimization: PIDR Estimated vs Ground Truth Channel Map

Problem:
  - Single-user scenario with N movable antennas at the transmitter.
  - Each antenna selects a port (grid point) on the 32×32 channel map.
  - Antennas must be at least M ports apart (default M=4) to enforce
    the half-wavelength spacing constraint.
  - Goal: maximize the user's received power spectral density and spectral efficiency.
  - Constraint: transmit power spectral density P_total is fixed; power is
    allocated via channel-gain-weighted water-filling.

Three approaches compared:
  1. GT-based selection: antenna positions chosen using the ground-truth channel map.
  2. PIDR-based selection: antenna positions chosen using the PIDR-estimated channel map,
     but actual received PSD/rate computed using the true channel gains at those positions.
  3. Fixed (random) selection: antenna positions chosen randomly (satisfying the M-spacing
     constraint), serving as a baseline without any optimization.

Outputs:
  - Console summary of selected positions, received PSD, and rate in bps/Hz.
  - CSV file with per-antenna details.
  - Visualization: three-panel comparison heatmaps with selected antenna positions marked.
"""

from __future__ import annotations

import os
import argparse
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# The .mat maps store the gain as 10*log10|h|; the paper (eq. (2)) and all other figures use
# 10*log10|h|^2. Plots show the paper's units (same constant as pidr.data.GAIN_TO_PAPER_UNITS).
GAIN_TO_PAPER_UNITS = 2.0

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NOISE_PSD_DBM_PER_HZ = -70   # noise power spectral density N0 in dBm/Hz
NOISE_PSD_LINEAR = 10 ** ((NOISE_PSD_DBM_PER_HZ - 30) / 10)  # W/Hz

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _to_user_first(array: np.ndarray, field: str, mat_path: str) -> np.ndarray:
    """Convert a 2D/3D CSI field to [num_users, H, W]."""
    array = np.squeeze(np.asarray(array, dtype=np.float64))
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim != 3:
        raise ValueError(f"Field '{field}' in {mat_path} must be 2D or 3D, got shape {array.shape}")

    shape = tuple(int(dim) for dim in array.shape)
    if shape[0] == shape[1]:
        return np.moveaxis(array, -1, 0)
    if shape[1] == shape[2]:
        return array

    spatial = sorted(shape, reverse=True)[:2]
    height, width = sorted(spatial)
    if shape[0] == height and shape[1] == width:
        return np.moveaxis(array, -1, 0)
    if shape[1] == height and shape[2] == width:
        return array
    if shape[0] == height and shape[2] == width:
        return np.moveaxis(array, 1, 0)
    raise ValueError(f"Cannot infer user/spatial axes for field '{field}' in {mat_path}; shape={array.shape}")


def load_channel_maps(mat_path: str, field: str = "CSI_Gain") -> np.ndarray:
    """Load one or more channel gain maps from a .mat file.

    Returns shape [num_users, H, W] in dB scale. A 2D map is treated as a
    single-user map.
    """
    data = loadmat(mat_path)
    if field not in data:
        raise KeyError(f"Field '{field}' not found in {mat_path}. Available: {list(data.keys())}")
    return _to_user_first(data[field], field, mat_path)


def load_both_maps(
    gt_path: str,
    pidr_path: str,
    field: str = "CSI_Gain",
    user_index: int | None = None,
    user_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load one user from ground-truth and PIDR-estimated channel gain maps."""
    gt_maps = load_channel_maps(gt_path, field)
    pidr_maps = load_channel_maps(pidr_path, field)
    if gt_maps.shape != pidr_maps.shape:
        raise ValueError(
            f"Shape mismatch: GT {gt_maps.shape} vs PIDR {pidr_maps.shape}"
        )
    num_users = gt_maps.shape[0]
    if user_index is None:
        rng = np.random.default_rng(user_seed)
        user_index = int(rng.integers(0, num_users))
    if not (0 <= user_index < num_users):
        raise ValueError(f"user_index must be in [0, {num_users - 1}], got {user_index}")
    return gt_maps[user_index], pidr_maps[user_index], int(user_index)


# ---------------------------------------------------------------------------
# Antenna selection
# ---------------------------------------------------------------------------

def select_antenna_positions(
    gain_map: np.ndarray,
    n_antennas: int,
    min_distance_ports: float = 4.0,
) -> np.ndarray:
    """Greedy selection of N antenna positions to maximize sum channel gain.

    The algorithm iteratively picks the grid point with the highest gain,
    then excludes a neighbourhood of `min_distance_ports` around it to enforce
    the minimum antenna spacing constraint (antennas must be at least
    M ports apart).

    Args:
        gain_map: 2D array of channel gains (in linear scale, not dB).
        n_antennas: Number of antennas to place.
        min_distance_ports: Minimum Euclidean distance between antennas
                            (in grid units, default 4).

    Returns:
        Array of shape (n_antennas, 2) with (row, col) indices.
    """
    H, W = gain_map.shape
    selected: List[Tuple[int, int]] = []
    available = np.ones((H, W), dtype=bool)

    for _ in range(n_antennas):
        # Find the best available position
        available_gain = np.where(available, gain_map, -np.inf)
        best_idx = np.argmax(available_gain)
        best_pos = np.unravel_index(best_idx, (H, W))
        selected.append(best_pos)

        # Exclude neighbourhood around the selected position
        rows, cols = np.where(available)
        for r, c in zip(rows, cols):
            dist = np.sqrt((r - best_pos[0]) ** 2 + (c - best_pos[1]) ** 2)
            if dist < min_distance_ports:
                available[r, c] = False

        # If no positions remain, break early
        if not available.any():
            break

    result = np.array(selected, dtype=np.int32)
    if len(result) < n_antennas:
        print(
            f"Warning: only {len(result)}/{n_antennas} positions could be selected "
            f"with min_distance_ports={min_distance_ports}."
        )
    return result


def select_fixed_positions(
    map_shape: Tuple[int, int],
    n_antennas: int,
    min_distance_ports: float = 4.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomly select N antenna positions satisfying the M-spacing constraint.

    This simulates a fixed (non-optimized) antenna placement as a baseline
    for comparison against GT-based and PIDR-based optimized selections.

    Args:
        map_shape: (H, W) dimensions of the channel map.
        n_antennas: Number of antennas to place.
        min_distance_ports: Minimum Euclidean distance between antennas
                            (in grid units, default 4).
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Array of shape (n_antennas, 2) with (row, col) indices.
    """
    if rng is None:
        rng = np.random.default_rng()
    H, W = map_shape
    selected: List[Tuple[int, int]] = []
    available = np.ones((H, W), dtype=bool)

    for _ in range(n_antennas):
        candidates = np.argwhere(available)
        if len(candidates) == 0:
            break
        # Pick a random available position
        idx = rng.integers(0, len(candidates))
        best_pos = tuple(candidates[idx])
        selected.append(best_pos)

        # Exclude neighbourhood around the selected position
        rows, cols = np.where(available)
        for r, c in zip(rows, cols):
            dist = np.sqrt((r - best_pos[0]) ** 2 + (c - best_pos[1]) ** 2)
            if dist < min_distance_ports:
                available[r, c] = False

    result = np.array(selected, dtype=np.int32)
    if len(result) < n_antennas:
        print(
            f"Warning: only {len(result)}/{n_antennas} fixed positions could be selected "
            f"with min_distance_ports={min_distance_ports}."
        )
    return result


# ---------------------------------------------------------------------------
# Power allocation & rate calculation
# ---------------------------------------------------------------------------

def compute_received_power_and_rate(
    gain_map_db: np.ndarray,
    positions: np.ndarray,
    total_power: float = 1.0,
    noise_psd: float = NOISE_PSD_LINEAR,
) -> Tuple[float, float, np.ndarray]:
    """Compute received PSD and achievable rate for selected antenna positions.

    CSI_Gain represents the channel magnitude in dB:
        CSI_Gain_dB = 10 * log10(|h|),  where |h| = sqrt(Real^2 + Imag^2)
    Therefore:
        |h|   = 10^(CSI_Gain_dB / 10)   (channel magnitude, linear)
        |h|^2 = 10^(CSI_Gain_dB / 5)    (channel power, linear)

    For each antenna i:
        S_rx_i = S_tx_i * |h_i|^2
    Total received PSD:
        S_rx_total = sum_i S_rx_i  (incoherent sum of per-antenna received PSDs)

    Power is allocated proportionally to channel power |h|^2 (water-filling-like).

    Args:
        gain_map_db: 2D array of CSI_Gain in dB (10*log10(|h|)).
        positions: Array of shape (N, 2) with (row, col) indices.
        total_power: Total transmit power spectral density (linear scale, W/Hz).
        noise_psd: Noise power spectral density N0 (linear scale, W/Hz).

    Returns:
        received_power: Total received power spectral density (W/Hz).
        rate_bpshz: Achievable spectral efficiency in bps/Hz.
        power_weights: Per-antenna transmit power allocation weights.
    """
    n_antennas = len(positions)

    # CSI_Gain_dB = 10 * log10(|h|)
    # |h|   = 10^(CSI_Gain_dB / 10)
    # |h|^2 = (10^(CSI_Gain_dB / 10))^2 
    channel_power_linear = 10.0 ** (gain_map_db / 5 )
    
    # Extract channel power |h|^2 at selected positions
    channel_powers = np.array([channel_power_linear[r, c] for r, c in positions])

    # Power allocation proportional to channel power (water-filling)
    power_weights = channel_powers / (channel_powers.sum() + 1e-30)
    tx_powers = total_power * power_weights  # per-antenna transmit PSD

    # Per-antenna received PSD, then sum (incoherent combining)
    #   S_rx_i = S_tx_i * |h_i|^2
    #   S_rx_total = sum_i S_rx_i
    rx_powers = tx_powers * channel_powers
    received_power = rx_powers.sum()

    # Spectral efficiency: R = log2(1 + SNR), where SNR is PSD ratio S_rx/N0.
    snr = received_power / (noise_psd + 1e-30)
    rate_bpshz = np.log2(1.0 + snr)

    return received_power, rate_bpshz, power_weights


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_three_way_comparison(
    gt_map: np.ndarray,
    pidr_map: np.ndarray,
    gt_positions: np.ndarray,
    pidr_positions: np.ndarray,
    fixed_positions: np.ndarray,
    gt_power: float,
    pidr_power: float,
    fixed_power: float,
    gt_rate: float,
    pidr_rate: float,
    fixed_rate: float,
    save_path: str,
    n_antennas: int,
) -> None:
    """Generate a 3-panel comparison figure with antenna positions marked."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    gt_map = gt_map * GAIN_TO_PAPER_UNITS
    pidr_map = pidr_map * GAIN_TO_PAPER_UNITS
    vmin = min(gt_map.min(), pidr_map.min())
    vmax = max(gt_map.max(), pidr_map.max())

    titles = [
        f"GT-Based Optimization\n"
        f"S_rx={gt_power:.4e} W/Hz, Rate={gt_rate:.3f} bps/Hz",
        f"PIDR-Based Optimization\n"
        f"S_rx={pidr_power:.4e} W/Hz, Rate={pidr_rate:.3f} bps/Hz",
        f"Fixed (Random) Antennas\n"
        f"S_rx={fixed_power:.4e} W/Hz, Rate={fixed_rate:.3f} bps/Hz",
    ]

    maps = [gt_map, pidr_map, gt_map]
    positions = [gt_positions, pidr_positions, fixed_positions]
    markers = ["o", "s", "^"]

    for ax, title, cmap_data, pos, marker in zip(
        axes, titles, maps, positions, markers
    ):
        im = ax.imshow(cmap_data, cmap="jet", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(title, fontsize=10, pad=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Mark selected antenna positions
        if len(pos) > 0:
            ax.scatter(
                pos[:, 1], pos[:, 0],
                marker=marker, s=120, edgecolors="white",
                facecolors="none", linewidths=2,
                label=f"{len(pos)} antennas",
            )
            # Annotate with antenna index
            for i, (r, c) in enumerate(pos):
                ax.annotate(
                    str(i), (c, r), color="white", fontsize=8,
                    ha="center", va="center", fontweight="bold",
                )
        ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Three-way comparison figure saved to: {save_path}")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_optimization(
    gt_path: str,
    pidr_path: str,
    n_antennas: int,
    total_power: float,
    output_dir: str,
    min_distance_ports: float = 10.0,
    noise_psd: float = NOISE_PSD_LINEAR,
    seed: int = 42,
    user_index: int | None = None,
    user_seed: int = 42,
) -> None:
    """Run the full MA position optimization benchmark with three-way comparison."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert noise power spectral density to dBm/Hz for display.
    noise_psd_dbm_per_hz = 10 * np.log10(noise_psd) + 30 if noise_psd > 0 else float('nan')

    # 1. Load channel maps
    print("=" * 70)
    print("Movable Antenna Position Optimization Benchmark")
    print("=" * 70)
    print(f"  N antennas              : {n_antennas}")
    print(f"  Tx power PSD             : {total_power} W/Hz")
    print(f"  Noise PSD N0             : {noise_psd_dbm_per_hz:.1f} dBm/Hz ({noise_psd:.2e} W/Hz)")
    print(f"  Min antenna spacing (M)  : {min_distance_ports} grid units")
    print(f"  Random seed              : {seed}")
    print(f"  Requested user index     : {'random' if user_index is None else user_index}")
    print(f"  User selection seed      : {user_seed}")
    print(f"  GT map                   : {gt_path}")
    print(f"  PIDR map                 : {pidr_path}")
    print(f"  Output dir               : {output_dir}")
    print("-" * 70)

    gt_map_db, pidr_map_db, selected_user_index = load_both_maps(
        gt_path,
        pidr_path,
        user_index=user_index,
        user_seed=user_seed,
    )
    H, W = gt_map_db.shape
    print(f"  Selected user index : {selected_user_index}")
    print(f"  Map size            : {H} x {W}")
    print()

    # 2. Select antenna positions
    print(">>> Selecting antenna positions ...")
    rng = np.random.default_rng(seed)
    gt_positions = select_antenna_positions(gt_map_db, n_antennas, min_distance_ports)
    pidr_positions = select_antenna_positions(pidr_map_db, n_antennas, min_distance_ports)
    fixed_positions = select_fixed_positions(
        (H, W), n_antennas, min_distance_ports, rng=rng
    )
    print(f"  GT-based  : {len(gt_positions)} positions selected")
    print(f"  PIDR-based: {len(pidr_positions)} positions selected")
    print(f"  Fixed      : {len(fixed_positions)} positions selected (random)")
    print()

    # 3. Compute received PSD and rate
    #    IMPORTANT: For PIDR-based selection, the actual received PSD/rate
    #    is computed using the TRUE channel gains at the PIDR-selected positions.
    #    Fixed antennas also use GT gains for fair comparison.
    print(">>> Computing received PSD and achievable rate ...")
    gt_power, gt_rate, gt_weights = compute_received_power_and_rate(
        gt_map_db, gt_positions, total_power, noise_psd
    )
    pidr_power, pidr_rate, pidr_weights = compute_received_power_and_rate(
        gt_map_db, pidr_positions, total_power, noise_psd  # <-- uses GT gains at PIDR positions
    )
    fixed_power, fixed_rate, fixed_weights = compute_received_power_and_rate(
        gt_map_db, fixed_positions, total_power, noise_psd  # <-- uses GT gains at fixed positions
    )
    print()

    # 4. Print results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n--- Ground-Truth-Based Selection ---")
    for i, (r, c) in enumerate(gt_positions):
        gain_db = gt_map_db[r, c]
        print(f"    Antenna {i}: ({r:2d}, {c:2d})  gain={gain_db:.2f} dB  "
              f"power_share={gt_weights[i]*100:.2f}%")
    print(f"  Total received PSD  : {gt_power:.6e} W/Hz")
    print(f"  Achievable rate     : {gt_rate:.6f} bps/Hz")

    print(f"\n--- PIDR-Based Selection (evaluated on GT map) ---")
    for i, (r, c) in enumerate(pidr_positions):
        gain_db = gt_map_db[r, c]
        print(f"    Antenna {i}: ({r:2d}, {c:2d})  gain={gain_db:.2f} dB  "
              f"power_share={pidr_weights[i]*100:.2f}%")
    print(f"  Total received PSD  : {pidr_power:.6e} W/Hz")
    print(f"  Achievable rate     : {pidr_rate:.6f} bps/Hz")

    print(f"\n--- Fixed (Random) Antennas (evaluated on GT map) ---")
    for i, (r, c) in enumerate(fixed_positions):
        gain_db = gt_map_db[r, c]
        print(f"    Antenna {i}: ({r:2d}, {c:2d})  gain={gain_db:.2f} dB  "
              f"power_share={fixed_weights[i]*100:.2f}%")
    print(f"  Total received PSD  : {fixed_power:.6e} W/Hz")
    print(f"  Achievable rate     : {fixed_rate:.6f} bps/Hz")

    # 5. Three-way comparison
    print(f"\n--- Three-Way Comparison ---")
    print(f"  {'Metric':<30} {'GT':<20} {'PIDR':<20} {'Fixed':<20}")
    print(f"  {'-'*28} {'-'*18} {'-'*18} {'-'*18}")
    print(f"  {'Received PSD (W/Hz)':<30} {gt_power:<20.6e} {pidr_power:<20.6e} {fixed_power:<20.6e}")
    print(f"  {'Rate (bps/Hz)':<30} {gt_rate:<20.6f} {pidr_rate:<20.6f} {fixed_rate:<20.6f}")

    gt_over_fixed_db = 10 * np.log10(gt_power / (fixed_power + 1e-30))
    pidr_over_fixed_db = 10 * np.log10(pidr_power / (fixed_power + 1e-30))
    print(f"\n  GT PSD gain over Fixed  : {gt_over_fixed_db:.2f} dB")
    print(f"  PIDR PSD gain over Fixed: {pidr_over_fixed_db:.2f} dB")
    print(f"  GT rate / Fixed rate      : {gt_rate/(fixed_rate+1e-30):.4f}x")
    print(f"  PIDR rate / Fixed rate    : {pidr_rate/(fixed_rate+1e-30):.4f}x")

    # 6. Position overlap analysis
    gt_set = set(tuple(p) for p in gt_positions)
    pidr_set = set(tuple(p) for p in pidr_positions)
    fixed_set = set(tuple(p) for p in fixed_positions)
    gt_pidr_overlap = gt_set & pidr_set
    print(f"\n  GT-PIDR position overlap: {len(gt_pidr_overlap)}/{n_antennas}")
    gt_fixed_overlap = gt_set & fixed_set
    print(f"  GT-Fixed position overlap: {len(gt_fixed_overlap)}/{n_antennas}")
    pidr_fixed_overlap = pidr_set & fixed_set
    print(f"  PIDR-Fixed position overlap: {len(pidr_fixed_overlap)}/{n_antennas}")

    # 7. Save detailed CSV
    csv_path = os.path.join(output_dir, "ma_optimization_results.csv")
    _save_csv(
        csv_path,
        gt_map_db, pidr_map_db,
        gt_positions, pidr_positions, fixed_positions,
        gt_weights, pidr_weights, fixed_weights,
        gt_power, pidr_power, fixed_power,
        gt_rate, pidr_rate, fixed_rate,
        n_antennas, total_power,
        noise_psd_dbm_per_hz,
        selected_user_index,
    )
    print(f"\nDetailed results saved to: {csv_path}")

    # 8. Plot three-way comparison
    plot_path = os.path.join(output_dir, f"ma_comparison_user{selected_user_index}_n{n_antennas}.pdf")
    plot_three_way_comparison(
        gt_map_db, pidr_map_db,
        gt_positions, pidr_positions, fixed_positions,
        gt_power, pidr_power, fixed_power,
        gt_rate, pidr_rate, fixed_rate,
        plot_path,
        n_antennas,
    )

    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)


def _save_csv(
    csv_path: str,
    gt_map: np.ndarray,
    pidr_map: np.ndarray,
    gt_positions: np.ndarray,
    pidr_positions: np.ndarray,
    fixed_positions: np.ndarray,
    gt_weights: np.ndarray,
    pidr_weights: np.ndarray,
    fixed_weights: np.ndarray,
    gt_power: float,
    pidr_power: float,
    fixed_power: float,
    gt_rate: float,
    pidr_rate: float,
    fixed_rate: float,
    n_antennas: int,
    total_power: float,
    noise_psd_dbm_per_hz: float = NOISE_PSD_DBM_PER_HZ,
    selected_user_index: int | None = None,
) -> None:
    """Save detailed per-antenna results to CSV (three-way comparison)."""
    import csv

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header: configuration
        writer.writerow(["Configuration"])
        writer.writerow(["N_antennas", n_antennas])
        writer.writerow(["Tx_power_PSD_W_per_Hz", total_power])
        writer.writerow(["Noise_PSD_N0_dBm_per_Hz", noise_psd_dbm_per_hz])
        if selected_user_index is not None:
            writer.writerow(["Selected_user_index", selected_user_index])
        writer.writerow([])

        # Per-antenna details
        writer.writerow([
            "Method", "Antenna_Index", "Row", "Col",
            "Gain_dB", "Power_Weight", "Tx_PSD_W_per_Hz",
        ])

        for i, (r, c) in enumerate(gt_positions):
            writer.writerow([
                "GT", i, r, c,
                f"{gt_map[r, c]:.4f}",
                f"{gt_weights[i]:.6f}",
                f"{total_power * gt_weights[i]:.6e}",
            ])

        for i, (r, c) in enumerate(pidr_positions):
            writer.writerow([
                "PIDR", i, r, c,
                f"{gt_map[r, c]:.4f}",  # actual GT gain at PIDR position
                f"{pidr_weights[i]:.6f}",
                f"{total_power * pidr_weights[i]:.6e}",
            ])

        for i, (r, c) in enumerate(fixed_positions):
            writer.writerow([
                "Fixed", i, r, c,
                f"{gt_map[r, c]:.4f}",  # actual GT gain at fixed position
                f"{fixed_weights[i]:.6f}",
                f"{total_power * fixed_weights[i]:.6e}",
            ])

        writer.writerow([])
        writer.writerow(["Summary"])
        writer.writerow(["Metric", "GT", "PIDR", "Fixed"])
        writer.writerow(["Total_Rx_PSD_W_per_Hz", f"{gt_power:.6e}", f"{pidr_power:.6e}", f"{fixed_power:.6e}"])
        writer.writerow(["Rate_bps_per_Hz", f"{gt_rate:.6f}", f"{pidr_rate:.6f}", f"{fixed_rate:.6f}"])
        writer.writerow([])
        writer.writerow(["Position_Overlap"])
        gt_set = set(tuple(p) for p in gt_positions)
        pidr_set = set(tuple(p) for p in pidr_positions)
        fixed_set = set(tuple(p) for p in fixed_positions)
        gt_pidr_overlap = gt_set & pidr_set
        gt_fixed_overlap = gt_set & fixed_set
        pidr_fixed_overlap = pidr_set & fixed_set
        writer.writerow(["GT-PIDR_Overlap_Count", len(gt_pidr_overlap)])
        writer.writerow(["GT-Fixed_Overlap_Count", len(gt_fixed_overlap)])
        writer.writerow(["PIDR-Fixed_Overlap_Count", len(pidr_fixed_overlap)])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MA Position Optimization: GT vs PIDR vs Fixed Antennas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gt-path", type=str,
        default="./data/test/sr.mat",
        help="Path to ground-truth channel map (.mat)",
    )
    parser.add_argument(
        "--pidr-path", type=str,
        default="./data/test/sr_pidr_batch.mat",
        help="Path to PIDR-estimated channel map (.mat)",
    )
    parser.add_argument(
        "--user-index", type=int, default=None,
        help="User/sample index to use from a multi-sample .mat file. If omitted, one user is selected randomly.",
    )
    parser.add_argument(
        "--user-seed", type=int, default=42,
        help="Random seed used when --user-index is omitted.",
    )
    parser.add_argument(
        "--n-antennas", type=int, default=4,
        help="Number of movable antennas to place",
    )
    parser.add_argument(
        "--total-power", type=float, default=1.0,
        help="Total transmit power spectral density in W/Hz for bps/Hz calculation",
    )
    parser.add_argument(
        "--noise-power-dbm", type=float, default=NOISE_PSD_DBM_PER_HZ,
        help="Noise power spectral density N0 in dBm/Hz. The reported rate is log2(1+SNR), in bps/Hz.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="results/ma_single_user/new",
        help="Output directory for results and figures",
    )
    parser.add_argument(
        "--min-distance-ports", type=float, default=6.0,
        help="Minimum Euclidean distance between antennas in grid units "
             "(antennas must be at least M ports apart, default M=4)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for fixed (random) antenna placement reproducibility",
    )
    args = parser.parse_args()

    # Convert noise power spectral density from dBm/Hz to W/Hz.
    noise_psd_linear = 10 ** ((args.noise_power_dbm - 30) / 10)

    run_optimization(
        gt_path=args.gt_path,
        pidr_path=args.pidr_path,
        n_antennas=args.n_antennas,
        total_power=args.total_power,
        output_dir=args.output_dir,
        min_distance_ports=args.min_distance_ports,
        noise_psd=noise_psd_linear,
        seed=args.seed,
        user_index=args.user_index,
        user_seed=args.user_seed,
    )


if __name__ == "__main__":
    main()
