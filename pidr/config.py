"""Project-wide configuration with grouped sections and compatibility aliases."""

import os

import torch


class _PathsConfig:
    DATA_PATH = "./data/el.mat"
    LOG_DIR = "./runs/diffusion"

    CHECKPOINT_DIR = "./checkpoints"
    CACHE_DIR = "./cache"

    DIFFUSION_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "diffusion_el.pth")
    PINN_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "cnn_pinn_best_el.pth")
    PIDR_DIFFUSION_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "pidr_diffusion_el (2).pth")
    PIDR_REFINER_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "pidr_refiner_eltest.pth")
    PIDR_CACHE_TRAIN = os.path.join(CACHE_DIR, "pidr_diffusion_recon_train_for_refiner_el.pt")
    PIDR_CACHE_TEST = os.path.join(CACHE_DIR, "pidr_diffusion_recon_test_for_refiner_el.pt")

    LEGACY_PIDR_DIFFUSION_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "cascade_diffusion_transformer_for_pinn_best_el.pth")
    LEGACY_PIDR_REFINER_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "cascade_pinn_best_el.pth")
    LEGACY_PIDR_CACHE_TRAIN = os.path.join(CACHE_DIR, "cascade_diffusion_transformer_recon_train_for_pinn_el.pt")
    LEGACY_PIDR_CACHE_TEST = os.path.join(CACHE_DIR, "cascade_diffusion_transformer_recon_test_for_pinn_el.pt")

    CASCADE_DIFFUSION_CHECKPOINT = PIDR_DIFFUSION_CHECKPOINT
    CASCADE_PINN_CHECKPOINT = PIDR_REFINER_CHECKPOINT
    CASCADE_CACHE_TRAIN = PIDR_CACHE_TRAIN
    CASCADE_CACHE_TEST = PIDR_CACHE_TEST


class _RuntimeConfig:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _DataConfig:
    IMG_HEIGHT = 21
    IMG_WIDTH = 41
    IMG_CHANNELS = 3

    DATA_SPLIT_RATIO = 0.90
    TRAIN_SAMPLE_COUNT = 864
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
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 30000
    GRAD_CLIP_NORM = 1.0


class _LossConfig:
    MISSING_REGION_WEIGHT = 3.0
    LAMBDA_X0 = 0.5
    LAMBDA_PHYS = 1e-3


class _PinnConfig:
    SAVE_PATH = _PathsConfig.PINN_CHECKPOINT
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
    FORCE_REBUILD_CACHE = True

    DIFFUSION_CHECKPOINT = _PathsConfig.PIDR_DIFFUSION_CHECKPOINT
    REFINER_CHECKPOINT = _PathsConfig.PIDR_REFINER_CHECKPOINT
    CACHE_TRAIN = _PathsConfig.PIDR_CACHE_TRAIN
    CACHE_TEST = _PathsConfig.PIDR_CACHE_TEST

    LEGACY_DIFFUSION_CHECKPOINT = _PathsConfig.LEGACY_PIDR_DIFFUSION_CHECKPOINT
    LEGACY_REFINER_CHECKPOINT = _PathsConfig.LEGACY_PIDR_REFINER_CHECKPOINT
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

    DIFFUSION_CHECKPOINT = os.path.join(_PathsConfig.CHECKPOINT_DIR, "pidr_partial_diffusion_0.2sr.pth")
    REFINER_CHECKPOINT = os.path.join(_PathsConfig.CHECKPOINT_DIR, "pidr_partial_refiner_0.2sr.pth")
    CACHE_TRAIN = os.path.join(_PathsConfig.CACHE_DIR, "pidr_partial_diffusion_recon_train_0.2sr.pt")
    CACHE_TEST = os.path.join(_PathsConfig.CACHE_DIR, "pidr_partial_diffusion_recon_test_0.2sr.pt")


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
    PIDR_DIFFUSION_CHECKPOINT_CANDIDATES = [
        _PathsConfig.PIDR_DIFFUSION_CHECKPOINT,
        _PathsConfig.LEGACY_PIDR_DIFFUSION_CHECKPOINT,
        _PathsConfig.DIFFUSION_CHECKPOINT,
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

    DIFFUSION_CHECKPOINT_CANDIDATES = [
        os.path.join(_PathsConfig.CHECKPOINT_DIR, "physics_model_best_nop_sr.pth"),
    ]
    PINN_CHECKPOINT_CANDIDATES = [
        os.path.join(_PathsConfig.CHECKPOINT_DIR, "cnn_pinn_best_sr.pth"),
    ]
    PIDR_DIFFUSION_CHECKPOINT_CANDIDATES = [
        os.path.join(_PathsConfig.CHECKPOINT_DIR, "pidr_diffusion_best_sr.pth"),
        os.path.join(_PathsConfig.CHECKPOINT_DIR, "cascade_diffusion_transformer_for_pinn_best_sr.pth"),
    ]
    PIDR_REFINER_CHECKPOINT_CANDIDATES = [
        os.path.join(_PathsConfig.CHECKPOINT_DIR, "pidr_refiner_best_sr.pth"),
        os.path.join(_PathsConfig.CHECKPOINT_DIR, "cascade_pinn_best_sr.pth"),
    ]


class _PIDRMaskRatioBenchmarkConfig:
    OUTPUT_DIR = "results/mask_ratios"
    ENFORCE_OBSERVED = True
    SAMPLING_STRIDE = 1
    NUM_EVAL_SAMPLES = 50
    MASK_STRATEGY = "random"
    PLOT_FORMAT = "pdf"

    # NOTE:
    # - mask_ratio means missing ratio, consistent with Config.MASK_RATIO.
    # - Replace checkpoint candidate lists with your own pretrained weights.
    EXPERIMENTS = [
        {
            "name": "mr70",
            "label": "Missing 70%",
            "mask_ratio": 0.70,
            "pidr_diffusion_checkpoint_candidates": [
                os.path.join(_PathsConfig.CHECKPOINT_DIR, "pidr_diffusion_0.7sr.pth"),
            ],
            "pidr_refiner_checkpoint_candidates": [
                os.path.join(_PathsConfig.CHECKPOINT_DIR, "pidr_refiner_0.7sr.pth"),
            ],
        },
        {
            "name": "mr80",
            "label": "Missing 80%",
            "mask_ratio": 0.80,
            "pidr_diffusion_checkpoint_candidates": [
                os.path.join(_PathsConfig.CHECKPOINT_DIR, "pidr_diffusion_0.8sr.pth"),
            ],
            "pidr_refiner_checkpoint_candidates": [
                os.path.join(_PathsConfig.CHECKPOINT_DIR, "pidr_refiner_0.8sr.pth"),
            ],
        },
        {
            "name": "mr90",
            "label": "Missing 90%",
            "mask_ratio": 0.90,
            "pidr_diffusion_checkpoint_candidates": [
                os.path.join(_PathsConfig.CHECKPOINT_DIR, "pidr_diffusion_0.9sr.pth"),
            ],
            "pidr_refiner_checkpoint_candidates": [
                os.path.join(_PathsConfig.CHECKPOINT_DIR, "pidr_refiner_0.9sr.pth"),
            ],
        },
        {
            "name": "mr95",
            "label": "Missing 95%",
            "mask_ratio": 0.95,
            "pidr_diffusion_checkpoint_candidates": [
                os.path.join(_PathsConfig.CHECKPOINT_DIR, "pidr_diffusion_0.95sr.pth"),
                _PathsConfig.PIDR_DIFFUSION_CHECKPOINT,
                _PathsConfig.LEGACY_PIDR_DIFFUSION_CHECKPOINT,
            ],
            "pidr_refiner_checkpoint_candidates": [
                os.path.join(_PathsConfig.CHECKPOINT_DIR, "pidr_refiner_0.99sr.pth"),
                _PathsConfig.PIDR_REFINER_CHECKPOINT,
                _PathsConfig.LEGACY_PIDR_REFINER_CHECKPOINT,
            ],
        },
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

    DIFFUSION_CHECKPOINT_CANDIDATES = [
        _PartialSupervisionPIDRConfig.DIFFUSION_CHECKPOINT,
    ]
    REFINER_CHECKPOINT_CANDIDATES = [
        _PartialSupervisionPIDRConfig.REFINER_CHECKPOINT,
    ]


class _BenchmarkConfig:
    DIFFUSION_PINN_PIDR = _PIDRBenchmarkConfig
    DIFFUSION_PINN_CASCADE = _PIDRBenchmarkConfig
    MASK_STRATEGIES = _MaskStrategyBenchmarkConfig
    PIDR_MASK_RATIOS = _PIDRMaskRatioBenchmarkConfig
    PIDR_PARTIAL_SUPERVISION = _PartialSupervisionPIDRBenchmarkConfig


class Config:
    """Grouped project configuration with flat aliases for legacy callers."""

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
    SAVE_PATH = PATHS.DIFFUSION_CHECKPOINT
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
