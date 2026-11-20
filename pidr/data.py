"""MAT 数据集加载器，负责动态归一化与多种掩码生成。"""

import glob
import os

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from pidr.config import Config

# The data store the gain as 10*log10|h|; the paper (eq. (2)) reports 10*log10|h|^2.
GAIN_TO_PAPER_UNITS = 2.0

try:
    import h5py
except ImportError:
    h5py = None


class CSIGridDataset(Dataset):
    """读取 CSI `.mat` 数据，并生成训练所需的稀疏条件输入。"""

    def __init__(
        self,
        data_dir,
        mode="train",
        split_ratio=0.9,
        seed=42,
        gain_in_db=True,
        mask_strategy="random",
        mask_seed=None,
    ):
        self.data_dir = data_dir
        self.mode = mode
        self.split_ratio = split_ratio
        self.seed = seed
        self.gain_in_db = gain_in_db
        self.mask_strategy = mask_strategy
        self.mask_seed = mask_seed

        mat_files = self._resolve_mat_files(data_dir)
        raw_samples, valid_masks = self._load_all_files(mat_files)

        self.global_min, self.global_max = self._compute_normalization_stats(raw_samples)
        self.gt_data = self.normalize_samples(raw_samples)
        self.valid_masks = valid_masks.astype(np.float32)
        self.indices = self._build_split_indices(len(self.gt_data))
        if self.mode == "train" and Config.TRAIN_SAMPLE_COUNT is not None:
            limit = Config.TRAIN_SAMPLE_COUNT
            if not isinstance(limit, int):
                raise TypeError(f"Config.TRAIN_SAMPLE_COUNT must be an int or None, got {type(limit)}")
            if limit <= 0:
                raise ValueError(f"Config.TRAIN_SAMPLE_COUNT must be > 0, got {limit}")
            if limit > len(self.indices):
                raise ValueError(
                    f"Config.TRAIN_SAMPLE_COUNT={limit} exceeds train split size {len(self.indices)}."
                )
            self.indices = self.indices[:limit]

        print(
            f"[{self.mode.upper()}] loaded {len(self.indices)} / {len(self.gt_data)} samples "
            f"from {len(mat_files)} .mat file(s). "
            f"global_min={self.global_min.tolist()}, global_max={self.global_max.tolist()}"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        gt_csi = self.gt_data[sample_idx].copy()
        valid_mask = self.valid_masks[sample_idx, 0].copy()

        mask = self.generate_mask(valid_mask, strategy=self.mask_strategy, sample_idx=sample_idx)
        mask = mask[None, :, :]
        sparse_csi = gt_csi * mask

        return {
            "gt_csi": torch.from_numpy(gt_csi).float(),
            "sparse_csi": torch.from_numpy(sparse_csi).float(),
            "mask": torch.from_numpy(mask).float(),
        }

    def get_normalization_tensors(self, device):
        """把当前数据集的归一化上下界转换成 Torch 张量。"""
        global_min = torch.tensor(self.global_min, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        global_max = torch.tensor(self.global_max, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        return global_min, global_max

    def normalize_samples(self, samples):
        """按当前数据集统计量把样本线性归一化到 [-1, 1]。"""
        global_min = self.global_min.reshape(1, 3, 1, 1)
        global_max = self.global_max.reshape(1, 3, 1, 1)
        scale = np.maximum(global_max - global_min, 1e-12)

        safe_samples = np.where(np.isfinite(samples), samples, global_min)
        normalized = (safe_samples - global_min) / scale
        normalized = np.clip(normalized, 0.0, 1.0)
        normalized = normalized * 2.0 - 1.0
        return normalized.astype(np.float32)

    def inverse_normalize_np(self, normalized_map):
        """把归一化后的 numpy 张量还原到原始物理范围。"""
        normalized_map = np.asarray(normalized_map, dtype=np.float32)
        rescaled = (normalized_map + 1.0) / 2.0

        if normalized_map.ndim == 3:
            global_min = self.global_min.reshape(3, 1, 1)
            global_max = self.global_max.reshape(3, 1, 1)
        elif normalized_map.ndim == 4:
            global_min = self.global_min.reshape(1, 3, 1, 1)
            global_max = self.global_max.reshape(1, 3, 1, 1)
        else:
            raise ValueError(f"Expected 3D or 4D normalized map, got shape {normalized_map.shape}")

        return rescaled * (global_max - global_min) + global_min

    def inverse_normalize_channel(self, channel_data, channel_idx):
        """只对单个通道做反归一化，便于评估与绘图。"""
        c_min = self.global_min[channel_idx]
        c_max = self.global_max[channel_idx]
        rescaled = (np.asarray(channel_data, dtype=np.float32) + 1.0) / 2.0
        return rescaled * (c_max - c_min) + c_min

    def gain_db(self, channel_data):
        """Gain channel in the paper's units, 10*log10|h|^2 (the data store 10*log10|h|)."""
        return self.inverse_normalize_channel(channel_data, 0) * GAIN_TO_PAPER_UNITS

    def generate_mask(self, valid_mask, strategy="random", sample_idx=None, observed_ratio=None):
        """根据指定策略为单个样本生成观测掩码。"""
        observed_ratio = 1.0 - Config.MASK_RATIO if observed_ratio is None else observed_ratio
        valid_positions = np.argwhere(valid_mask > 0.5)
        if len(valid_positions) == 0:
            return np.zeros_like(valid_mask, dtype=np.float32)

        target_count = max(1, int(round(len(valid_positions) * observed_ratio)))
        rng = self._make_rng(sample_idx)

        if strategy == "random":
            selected = self._select_random_positions(valid_positions, target_count, rng)
        elif strategy == "uniform":
            selected = self._select_uniform_positions(valid_positions, target_count, valid_mask.shape, rng)
        elif strategy in {"extreme_uneven", "very_uneven"}:
            selected = self._select_extreme_uneven_positions(valid_positions, target_count, valid_mask.shape, rng)
        else:
            raise ValueError(f"Unsupported mask strategy: {strategy}")

        mask = np.zeros_like(valid_mask, dtype=np.float32)
        mask[selected[:, 0], selected[:, 1]] = 1.0
        return mask

    @staticmethod
    def _resolve_mat_files(data_dir):
        if os.path.isfile(data_dir):
            if not data_dir.lower().endswith(".mat"):
                raise ValueError(f"Expected a .mat file, got: {data_dir}")
            return [data_dir]

        if os.path.isdir(data_dir):
            mat_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
            if mat_files:
                return mat_files
            raise FileNotFoundError(f"No .mat files found in directory: {data_dir}")

        raise FileNotFoundError(f"Dataset path does not exist: {data_dir}")

    def _load_all_files(self, mat_files):
        """加载多个 `.mat` 文件并在样本维拼接。"""
        sample_blocks = []
        mask_blocks = []

        for mat_path in mat_files:
            samples, valid_mask = self._load_single_mat(mat_path)
            sample_blocks.append(samples)
            mask_blocks.append(valid_mask)

        return np.concatenate(sample_blocks, axis=0), np.concatenate(mask_blocks, axis=0)

    def _load_single_mat(self, mat_path):
        """读取单个 `.mat` 文件的三个 CSI 通道。"""
        mat_data = self._read_mat_file(mat_path)

        gain = self._to_sample_first(self._extract_field(mat_data, "CSI_Gain", mat_path), "CSI_Gain", mat_path)
        real = self._to_sample_first(self._extract_field(mat_data, "CSI_Real", mat_path), "CSI_Real", mat_path)
        imag = self._to_sample_first(self._extract_field(mat_data, "CSI_Imag", mat_path), "CSI_Imag", mat_path)

        if not (gain.shape == real.shape == imag.shape):
            raise ValueError(
                f"Field shapes do not match in {mat_path}: "
                f"gain={gain.shape}, real={real.shape}, imag={imag.shape}"
            )

        # if not self.gain_in_db:
        #     gain = 10.0 * np.log10(np.maximum(gain, 1e-12))

        samples = np.stack([gain, real, imag], axis=1).astype(np.float32)
        valid_mask = np.all(np.isfinite(samples), axis=1, keepdims=True).astype(np.float32)
        return samples, valid_mask

    @staticmethod
    def _read_mat_file(mat_path):
        try:
            return loadmat(mat_path)
        except NotImplementedError:
            if h5py is None:
                raise RuntimeError(f"Cannot read MATLAB v7.3 file without h5py: {mat_path}")

            with h5py.File(mat_path, "r") as mat_file:
                return {key: np.array(mat_file[key]) for key in mat_file.keys()}

    @staticmethod
    def _extract_field(mat_data, field_name, mat_path):
        if field_name in mat_data:
            return mat_data[field_name]

        lower_map = {str(key).lower(): key for key in mat_data.keys()}
        lower_name = field_name.lower()
        if lower_name in lower_map:
            return mat_data[lower_map[lower_name]]

        available_fields = ", ".join(sorted(str(key) for key in mat_data.keys()))
        raise KeyError(f"Missing field '{field_name}' in {mat_path}. Available fields: {available_fields}")

    @staticmethod
    def _to_sample_first(array, field_name, mat_path):
        array = np.asarray(array)
        array = np.squeeze(array)
        img_height = Config.IMG_HEIGHT
        img_width = Config.IMG_WIDTH

        if array.ndim == 2:
            if array.shape != (img_height, img_width):
                raise ValueError(
                    f"Field '{field_name}' in {mat_path} has invalid 2D shape {array.shape}; "
                    f"expected ({img_height}, {img_width})."
                )
            return array[None, :, :].astype(np.float32)

        if array.ndim != 3:
            raise ValueError(
                f"Field '{field_name}' in {mat_path} must be 2D or 3D, got shape {array.shape}."
            )

        if array.shape[0] == img_height and array.shape[1] == img_width:
            return np.moveaxis(array, -1, 0).astype(np.float32)

        if array.shape[1] == img_height and array.shape[2] == img_width:
            return array.astype(np.float32)

        if array.shape[0] == img_height and array.shape[2] == img_width:
            return np.moveaxis(array, 1, 0).astype(np.float32)

        raise ValueError(
            f"Field '{field_name}' in {mat_path} has unsupported shape {array.shape}. "
            f"Expected (H, W, N), (N, H, W) or (H, N, W) with "
            f"H={img_height}, W={img_width}."
        )

    @staticmethod
    def _compute_normalization_stats(samples):
        """统计三个通道各自的全局最小值和最大值。"""
        global_min = []
        global_max = []

        for channel_idx in range(samples.shape[1]):
            channel_values = samples[:, channel_idx]
            finite_values = channel_values[np.isfinite(channel_values)]
            if finite_values.size == 0:
                raise ValueError(f"Channel {channel_idx} does not contain any finite values.")

            channel_min = float(finite_values.min())
            channel_max = float(finite_values.max())

            if np.isclose(channel_min, channel_max):
                channel_max = channel_min + 1.0

            global_min.append(channel_min)
            global_max.append(channel_max)

        return np.asarray(global_min, dtype=np.float32), np.asarray(global_max, dtype=np.float32)

    def _build_split_indices(self, num_samples):
        if self.mode not in {"train", "test", "all"}:
            raise ValueError("mode must be one of: 'train', 'test', 'all'")

        indices = np.arange(num_samples)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)

        if self.mode == "all":
            return indices

        split_idx = int(num_samples * self.split_ratio)
        if self.mode == "train":
            return indices[:split_idx]
        return indices[split_idx:]

    def _make_rng(self, sample_idx):
        if self.mask_seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.mask_seed + int(sample_idx))

    @staticmethod
    def _select_random_positions(valid_positions, target_count, rng):
        indices = rng.choice(len(valid_positions), size=min(target_count, len(valid_positions)), replace=False)
        return valid_positions[indices]

    @staticmethod
    def _select_uniform_positions(valid_positions, target_count, mask_shape, rng):
        height, width = mask_shape
        aspect_ratio = max(width / max(height, 1), 1e-6)
        points_y = max(1, int(round(np.sqrt(target_count / aspect_ratio))))
        points_x = max(1, int(np.ceil(target_count / points_y)))
        ys = np.linspace(0, height - 1, points_y)
        xs = np.linspace(0, width - 1, points_x)

        grid_positions = np.array(
            [(int(round(y)), int(round(x))) for y in ys for x in xs],
            dtype=np.int32,
        )
        grid_positions = np.unique(grid_positions, axis=0)

        valid_set = {tuple(position) for position in valid_positions.tolist()}
        selected_list = [position for position in grid_positions.tolist() if tuple(position) in valid_set]

        if len(selected_list) > target_count:
            sample_indices = np.linspace(0, len(selected_list) - 1, target_count, dtype=int)
            return np.asarray(selected_list, dtype=np.int32)[sample_indices]

        if len(selected_list) < target_count:
            selected_set = {tuple(position) for position in selected_list}
            remaining = np.asarray(
                [position for position in valid_positions.tolist() if tuple(position) not in selected_set],
                dtype=np.int32,
            )
            fill_count = min(target_count - len(selected_list), len(remaining))
            if fill_count > 0:
                fill_indices = rng.choice(len(remaining), size=fill_count, replace=False)
                selected_list.extend(remaining[fill_indices].tolist())

        return np.asarray(selected_list, dtype=np.int32)

    @staticmethod
    def _select_extreme_uneven_positions(valid_positions, target_count, mask_shape, rng):
        height, width = mask_shape
        region_mask = (valid_positions[:, 0] < max(1, height // 2)) & (valid_positions[:, 1] < max(1, width // 2))
        concentrated_region = valid_positions[region_mask]
        remaining_region = valid_positions[~region_mask]

        concentrated_count = min(int(round(target_count * 0.9)), len(concentrated_region))
        selected_parts = []

        if concentrated_count > 0:
            concentrated_indices = rng.choice(
                len(concentrated_region), size=concentrated_count, replace=False
            )
            selected_parts.append(concentrated_region[concentrated_indices])

        remaining_count = target_count - concentrated_count
        if remaining_count > 0:
            if len(remaining_region) > 0:
                remaining_indices = rng.choice(
                    len(remaining_region), size=min(remaining_count, len(remaining_region)), replace=False
                )
                selected_parts.append(remaining_region[remaining_indices])
            else:
                fallback = CSIGridDataset._select_random_positions(valid_positions, target_count, rng)
                return fallback

        return np.concatenate(selected_parts, axis=0)
