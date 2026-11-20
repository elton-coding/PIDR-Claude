"""Project-wide configuration with grouped sections and compatibility aliases.

Dataset presets
---------------
The dataset-dependent settings (data file, grid size, training hyperparameters and
the default checkpoint set) are chosen with the ``PIDR_DATASET`` environment variable
when this module is imported:

    PowerShell:  $env:PIDR_DATASET = "sr"; python -m scripts.train.train_pidr
    bash:        PIDR_DATASET=sr python -m scripts.train.train_pidr

``sr`` (Sionna RT) and ``fd`` (field response) use the paper's hyperparameter table;
``el`` (EM simulation, default) keeps the settings of the last EL experiment.

Checkpoints
-----------
Curated weights live in ``checkpoints/<dataset>/<experiment>/[<setting>/]<stage>.pth``
and are described in ``checkpoints/MANIFEST.csv`` (see docs/checkpoints.md). They are
only ever *read* by this config. Training scripts write to a fresh run directory
``runs/<dataset>/<YYYYmmdd_HHMMSS>/`` so a new training run can never overwrite a
checkpoint that a paper result depends on; copy a run's output into ``checkpoints/``
by hand and register it with ``python -m scripts.tools.checkpoint_manifest update``.
"""

import datetime
import os

import torch


DATASET_PRESETS = {
    # Paper settings (Table "Main experimental hyperparameters").
    "sr": {
        "data_path": "./data/sr.mat",
        "img_height": 32,
        "img_width": 32,
        "train_sample_count": 9000,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "num_epochs": 1000,
        "force_rebuild_cache": False,
        # PIDR-Net (diffusion + field-response refiner) for every paper experiment at 80% missing.
        # pidr_cascade is the April cascade model used by the v13 sampling-strategy table.
        "pidr_dir": "pidr",
        "legacy_pidr_dir": "pidr_cascade",
    },
    # Same hyperparameters as SR; field.mat is also 32x32 with 10000 maps.
    # fd/pidr: refiner trained (2026-11) on the outputs of fd/baselines/diffusion.pth.
    "fd": {
        "data_path": "./data/field.mat",
        "img_height": 32,
        "img_width": 32,
        "train_sample_count": 9000,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "num_epochs": 1000,
        "force_rebuild_cache": False,
        "pidr_dir": "pidr",
        "legacy_pidr_dir": "pidr_cascade",
    },
    # EM simulation data: 21x41 grid, 961 maps (864 for training).
    "el": {
        "data_path": "./data/el.mat",
        "img_height": 21,
        "img_width": 41,
        "train_sample_count": 864,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "num_epochs": 30000,
        "force_rebuild_cache": True,
        "pidr_dir": "pidr",
        "legacy_pidr_dir": "pidr",
    },
}

DATASET = os.environ.get("PIDR_DATASET", "el").strip().lower()
if DATASET not in DATASET_PRESETS:
    raise ValueError(f"PIDR_DATASET must be one of {sorted(DATASET_PRESETS)}, got {DATASET!r}")
_PRESET = DATASET_PRESETS[DATASET]

CHECKPOINT_DIR = "./checkpoints"
# Output folder for this process's training run (created only when something is saved).
RUN_DIR = os.path.join("./runs", DATASET, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))


def ckpt(*parts):
    """Path of a curated checkpoint for the active dataset."""
    return os.path.join(CHECKPOINT_DIR, DATASET, *parts)


def run_path(*parts):
    """Path inside this process's training run directory."""
    return os.path.join(RUN_DIR, *parts)


class _PathsConfig:
    DATA_PATH = _PRESET["data_path"]
    LOG_DIR = run_path("tensorboard")

    CHECKPOINT_DIR = CHECKPOINT_DIR
    CACHE_DIR = "./cache"
    RUN_DIR = RUN_DIR

    # Curated checkpoints read by the benchmarks.
    DIFFUSION_CHECKPOINT = ckpt("baselines", "diffusion.pth")
    PINN_CHECKPOINT = ckpt("baselines", "pinn.pth")
    PIDR_DIFFUSION_CHECKPOINT = ckpt(_PRESET["pidr_dir"], "diffusion.pth")
    PIDR_REFINER_CHECKPOINT = ckpt(_PRESET["pidr_dir"], "refiner.pth")
    PIDR_CACHE_TRAIN = run_path("cache", "pidr_diffusion_recon_train.pt")
    PIDR_CACHE_TEST = run_path("cache", "pidr_diffusion_recon_test.pt")

    LEGACY_PIDR_DIFFUSION_CHECKPOINT = ckpt(_PRESET["legacy_pidr_dir"], "diffusion.pth")
    LEGACY_PIDR_REFINER_CHECKPOINT = ckpt(_PRESET["legacy_pidr_dir"], "refiner.pth")
    LEGACY_PIDR_CACHE_TRAIN = PIDR_CACHE_TRAIN
    LEGACY_PIDR_CACHE_TEST = PIDR_CACHE_TEST

    CASCADE_DIFFUSION_CHECKPOINT = PIDR_DIFFUSION_CHECKPOINT
    CASCADE_PINN_CHECKPOINT = PIDR_REFINER_CHECKPOINT
    CASCADE_CACHE_TRAIN = PIDR_CACHE_TRAIN
    CASCADE_CACHE_TEST = PIDR_CACHE_TEST


class _RuntimeConfig:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _DataConfig:
    IMG_HEIGHT = _PRESET["img_height"]
    IMG_WIDTH = _PRESET["img_width"]
    IMG_CHANNELS = 3

    DATA_SPLIT_RATIO = 0.90
    TRAIN_SAMPLE_COUNT = _PRESET["train_sample_count"]
    MASK_RATIO = 0.8
    EVAL_MASK_SEED = 0


class _DiffusionConfig:
    TIMESTEPS = 1000
    BETA_START = 0.0001
    BETA_END = 0.02
    NOISE_SCHEDULE = "cosine"
    PHYSICS_GUIDANCE_INTERVAL = 25


class _ModelConfig:
    BASE_CHANNELS = 64
    TIME_EMB_DIM = 256
    MODEL_DROPOUT = 0.10
    ATTENTION_HEADS = 4


class _TrainingConfig:
    BATCH_SIZE = _PRESET["batch_size"]
    LEARNING_RATE = _PRESET["learning_rate"]
    NUM_EPOCHS = _PRESET["num_epochs"]
    GRAD_CLIP_NORM = 1.0


class _LossConfig:
    MISSING_REGION_WEIGHT = 3.0
    LAMBDA_X0 = 0.5
    LAMBDA_PHYS = 1e-3


class _PinnConfig:
    # scripts/train/train_pinn.py writes here (never into checkpoints/).
    SAVE_PATH = run_path("baselines", "pinn.pth")
    WEIGHT_DECAY = 1e-4
    MIN_WARMUP_EPOCHS = 5
    MAX_LAMBDA_POWER = 1e-3
    MAX_LAMBDA_PDE = 1e-4
    MAX_LAMBDA_PARAM = 1e-3


class _PIDRConfig:
    TRAIN_DIFFUSION = True
    TRAIN_PINN = True

    DIFFUSION_EPOCHS = _TrainingConfig.NUM_EPOCHS
    PINN_EPOCHS = _TrainingConfig.NUM_EPOCHS

    SAMPLE_STRIDE = 1
    PINN_TRAIN_MASK_SEED = 12345
    # Batch size for building the diffusion-reconstruction cache (inference only; larger is faster).
    CACHE_BATCH_SIZE = 256
    # Refiner variant (see pidr.models.pinn_refiner.PINNCSIReconstructor and docs/physics_refiner.md):
    # field-response LMMSE physics branch + coordinates, residual on the physics estimate.
    # The paper (v13) refiner was dict(prior="radial", use_coords=True, residual="none").
    REFINER_KWARGS = dict(prior="field_response", use_coords=True, residual="physics")
    FORCE_REBUILD_CACHE = _PRESET["force_rebuild_cache"]

    # scripts/train/train_pidr.py writes here (never into checkpoints/).
    DIFFUSION_CHECKPOINT = run_path("pidr", "diffusion.pth")
    REFINER_CHECKPOINT = run_path("pidr", "refiner.pth")
    CACHE_TRAIN = _PathsConfig.PIDR_CACHE_TRAIN
    CACHE_TEST = _PathsConfig.PIDR_CACHE_TEST

    # With TRAIN_DIFFUSION = False the refiner is trained on this curated diffusion model.
    LEGACY_DIFFUSION_CHECKPOINT = _PathsConfig.PIDR_DIFFUSION_CHECKPOINT
    LEGACY_REFINER_CHECKPOINT = _PathsConfig.PIDR_REFINER_CHECKPOINT
    LEGACY_CACHE_TRAIN = _PathsConfig.LEGACY_PIDR_CACHE_TRAIN
    LEGACY_CACHE_TEST = _PathsConfig.LEGACY_PIDR_CACHE_TEST

    PINN_CHECKPOINT = REFINER_CHECKPOINT


class _PartialSupervisionPIDRConfig:
    TRAIN_DIFFUSION = True
    TRAIN_REFINER = True

    DIFFUSION_EPOCHS = _TrainingConfig.NUM_EPOCHS
    REFINER_EPOCHS = _TrainingConfig.NUM_EPOCHS

    SAMPLE_STRIDE = 1
    FORCE_REBUILD_CACHE = False

    # Ratio of points that remain available in the partial training map.
    # available = input points + supervision points
    TOTAL_AVAILABLE_RATIO = 0.20

    # Fraction of available points assigned to the model input.
    # The remaining available points become supervision-only labels.
    INPUT_FRACTION_WITHIN_AVAILABLE = 0.50

    MASK_STRATEGY = "random"
    TRAIN_SPLIT_SEED = 2026
    EVAL_SPLIT_SEED = 0

    # scripts/train/train_pidr_semisupervised.py writes here (never into checkpoints/).
    # With TRAIN_DIFFUSION = False the refiner is trained on the curated alpha model instead.
    _ALPHA_DIR = f"alpha{TOTAL_AVAILABLE_RATIO:g}"
    DIFFUSION_CHECKPOINT = (
        run_path("semisup", _ALPHA_DIR, "diffusion.pth") if TRAIN_DIFFUSION
        else ckpt("semisup", _ALPHA_DIR, "diffusion.pth")
    )
    REFINER_CHECKPOINT = run_path("semisup", _ALPHA_DIR, "refiner.pth")
    CACHE_TRAIN = run_path("cache", f"semisup_{_ALPHA_DIR}_recon_train.pt")
    CACHE_TEST = run_path("cache", f"semisup_{_ALPHA_DIR}_recon_test.pt")


class _PIDRBenchmarkConfig:
    OUTPUT_DIR = "results/method_comparison"
    ENFORCE_OBSERVED = True
    SAMPLING_STRIDE = 1
    MAX_PLOTS = 50
    NUM_EVAL_SAMPLES = 50
    PLOT_FORMAT = "pdf"

    DIFFUSION_CHECKPOINT_CANDIDATES = [
        _PathsConfig.DIFFUSION_CHECKPOINT,
    ]
    PINN_CHECKPOINT_CANDIDATES = [
        _PathsConfig.PINN_CHECKPOINT,
    ]
    # No fallback to the baseline diffusion model: a missing PIDR checkpoint must fail
    # loudly instead of silently reporting baseline-diffusion numbers as PIDR-Net.
    PIDR_DIFFUSION_CHECKPOINT_CANDIDATES = [
        _PathsConfig.PIDR_DIFFUSION_CHECKPOINT,
        _PathsConfig.LEGACY_PIDR_DIFFUSION_CHECKPOINT,
    ]
    PIDR_REFINER_CHECKPOINT_CANDIDATES = [
        _PathsConfig.PIDR_REFINER_CHECKPOINT,
        _PathsConfig.LEGACY_PIDR_REFINER_CHECKPOINT,
    ]

    CASCADE_DIFFUSION_CHECKPOINT_CANDIDATES = PIDR_DIFFUSION_CHECKPOINT_CANDIDATES
    CASCADE_PINN_CHECKPOINT_CANDIDATES = PIDR_REFINER_CHECKPOINT_CANDIDATES


class _MaskStrategyBenchmarkConfig:
    OUTPUT_DIR = "results/mask_strategies"
    ENFORCE_OBSERVED = True
    SAMPLING_STRIDE = 1
    NUM_EVAL_SAMPLES = 50

    # The 2026-11 paper uses the main PIDR-Net (pidr/, field-response refiner); v13 used pidr_cascade.
    DIFFUSION_CHECKPOINT_CANDIDATES = [ckpt("baselines", "diffusion.pth")]
    PINN_CHECKPOINT_CANDIDATES = [ckpt("baselines", "pinn.pth")]
    PIDR_DIFFUSION_CHECKPOINT_CANDIDATES = [ckpt("pidr", "diffusion.pth"), ckpt("pidr_cascade", "diffusion.pth")]
    PIDR_REFINER_CHECKPOINT_CANDIDATES = [ckpt("pidr", "refiner.pth"), ckpt("pidr_cascade", "refiner.pth")]


class _PIDRMaskRatioBenchmarkConfig:
    OUTPUT_DIR = "results/mask_ratios"
    ENFORCE_OBSERVED = True
    SAMPLING_STRIDE = 1
    NUM_EVAL_SAMPLES = 50
    MASK_STRATEGY = "random"
    PLOT_FORMAT = "pdf"

    # mask_ratio is the missing ratio (as Config.MASK_RATIO). Weights live in
    # checkpoints/<dataset>/mask_ratio/rho<ratio>/ (FR refiners, 2026-11; each trained at its own ratio).
    EXPERIMENTS = [
        {
            "name": f"mr{round(ratio * 100)}",
            "label": f"Missing {round(ratio * 100)}%",
            "mask_ratio": ratio,
            "pidr_diffusion_checkpoint_candidates": [ckpt("mask_ratio", f"rho{ratio:.2f}", "diffusion.pth")],
            "pidr_refiner_checkpoint_candidates": [ckpt("mask_ratio", f"rho{ratio:.2f}", "refiner.pth")],
        }
        for ratio in (0.70, 0.80, 0.90, 0.95)
    ]


class _PartialSupervisionPIDRBenchmarkConfig:
    OUTPUT_DIR = "results/semisupervised"
    ENFORCE_OBSERVED = True
    SAMPLING_STRIDE = _PartialSupervisionPIDRConfig.SAMPLE_STRIDE
    NUM_EVAL_SAMPLES = 50
    MAX_PLOTS = 50
    PLOT_FORMAT = "pdf"

    TOTAL_AVAILABLE_RATIO = _PartialSupervisionPIDRConfig.TOTAL_AVAILABLE_RATIO
    INPUT_FRACTION_WITHIN_AVAILABLE = _PartialSupervisionPIDRConfig.INPUT_FRACTION_WITHIN_AVAILABLE
    MASK_STRATEGY = _PartialSupervisionPIDRConfig.MASK_STRATEGY
    EVAL_SPLIT_SEED = _PartialSupervisionPIDRConfig.EVAL_SPLIT_SEED

    # Curated weights for the configured alpha (checkpoints/<dataset>/semisup/alpha<a>/).
    DIFFUSION_CHECKPOINT_CANDIDATES = [ckpt("semisup", f"alpha{TOTAL_AVAILABLE_RATIO:g}", "diffusion.pth")]
    REFINER_CHECKPOINT_CANDIDATES = [ckpt("semisup", f"alpha{TOTAL_AVAILABLE_RATIO:g}", "refiner.pth")]


class _BenchmarkConfig:
    DIFFUSION_PINN_PIDR = _PIDRBenchmarkConfig
    DIFFUSION_PINN_CASCADE = _PIDRBenchmarkConfig
    MASK_STRATEGIES = _MaskStrategyBenchmarkConfig
    PIDR_MASK_RATIOS = _PIDRMaskRatioBenchmarkConfig
    PIDR_PARTIAL_SUPERVISION = _PartialSupervisionPIDRBenchmarkConfig


class Config:
    """Grouped project configuration with flat aliases for legacy callers."""

    DATASET = DATASET

    PATHS = _PathsConfig
    RUNTIME = _RuntimeConfig
    DATA = _DataConfig
    DIFFUSION = _DiffusionConfig
    MODEL = _ModelConfig
    TRAINING = _TrainingConfig
    LOSS = _LossConfig
    PINN = _PinnConfig
    PIDR = _PIDRConfig
    PIDR_PARTIAL = _PartialSupervisionPIDRConfig
    CASCADE = _PIDRConfig
    BENCHMARK = _BenchmarkConfig

    DATA_PATH = PATHS.DATA_PATH
    LOG_DIR = PATHS.LOG_DIR
    SAVE_PATH = run_path("baselines", "diffusion.pth")  # train_diffusion.py output
    DEVICE = RUNTIME.DEVICE

    IMG_HEIGHT = DATA.IMG_HEIGHT
    IMG_WIDTH = DATA.IMG_WIDTH
    IMG_CHANNELS = DATA.IMG_CHANNELS

    TIMESTEPS = DIFFUSION.TIMESTEPS
    BETA_START = DIFFUSION.BETA_START
    BETA_END = DIFFUSION.BETA_END
    NOISE_SCHEDULE = DIFFUSION.NOISE_SCHEDULE
    PHYSICS_GUIDANCE_INTERVAL = DIFFUSION.PHYSICS_GUIDANCE_INTERVAL

    BASE_CHANNELS = MODEL.BASE_CHANNELS
    TIME_EMB_DIM = MODEL.TIME_EMB_DIM
    MODEL_DROPOUT = MODEL.MODEL_DROPOUT
    ATTENTION_HEADS = MODEL.ATTENTION_HEADS

    BATCH_SIZE = TRAINING.BATCH_SIZE
    LEARNING_RATE = TRAINING.LEARNING_RATE
    NUM_EPOCHS = TRAINING.NUM_EPOCHS
    GRAD_CLIP_NORM = TRAINING.GRAD_CLIP_NORM

    TRAIN_SAMPLE_COUNT = DATA.TRAIN_SAMPLE_COUNT
    MASK_RATIO = DATA.MASK_RATIO
    DATA_SPLIT_RATIO = DATA.DATA_SPLIT_RATIO
    EVAL_MASK_SEED = DATA.EVAL_MASK_SEED

    MISSING_REGION_WEIGHT = LOSS.MISSING_REGION_WEIGHT
    LAMBDA_X0 = LOSS.LAMBDA_X0
    LAMBDA_PHYS = LOSS.LAMBDA_PHYS
