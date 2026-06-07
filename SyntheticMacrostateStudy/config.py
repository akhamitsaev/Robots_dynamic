from dataclasses import dataclass
from pathlib import Path


@dataclass
class ResearchConfig:
    """Central configuration for autonomous multistep + macrostate research."""

    # Data / model paths. By default they are relative to this package folder.
    data_path: str = "data/processed_robots_data.parquet"
    model_path: str = "models/gnn_model_weights.pth"
    output_dir: str = "results/research"
    # Aliases used by the original StepPrediction GNNTrainer.
    model_save_path: str = "models/gnn_model_weights.pth"
    results_save_path: str = "results/gnn_predictions.parquet"

    # GNN architecture. The same model is trained one-step and reused autoregressively.
    k_neighbors: int = 5
    hidden_dim: int = 128
    edge_dim: int = 32
    gnn_layers: int = 2
    gnn_type: str = "GIN"
    dropout: float = 0.1

    # One-step GNN training.
    train_one_step: bool = True
    skip_training_if_model_exists: bool = True
    batch_size: int = 512
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    # Progress/logging flags expected by the original StepPrediction GraphDataset/GNNTrainer.
    show_progress: bool = True
    log_interval: int = 10
    grad_clip: float = 1.0
    lr_patience: int = 5
    lr_factor: float = 0.5
    # Fraction of fact experiment files used before train/val/test split.
    # 1.0 = use all files; 0.5 = sample 50% of files with random_state.
    train_sample_ratio: float = 1.0
    # Original GitHub behavior saves predictions for every file after training.
    # This can be slow because it performs one-step inference for all timesteps.
    # Keep True for GitHub-compatible behavior and for one-step video generation.
    # Set False only when you need to finish training quickly and do not need
    # results/gnn_predictions.parquet immediately.
    save_one_step_predictions_after_training: bool = True
    test_size: float = 0.2
    val_size: float = 0.1
    num_workers: int = 0
    data_device: str = "cpu"
    one_step_min_t: int = 2
    # GNN standardization follows the original StepPrediction/GNNTrainer behavior:
    # StandardScaler is always fitted on train node/edge/target tensors.
    max_train_samples_per_file: int | None = None
    max_val_samples_per_file: int | None = None
    max_test_samples_per_file: int | None = None

    # Dynamics / sampling.
    dt: float = 0.1
    start_step: int = 3  # seed states 0..3 are known; first predicted state is 4.
    synthetic_total_steps: int = 600
    n_robots: int = 50

    # Geometry of synthetic experiments. The platform is a circular dish of diameter 1000.
    robot_size: float = 60.0  # robot diameter; center-to-center distance must be >= this value
    dish_diameter: float = 1000.0
    min_center_distance: float | None = None  # None -> robot_size

    # Synthetic sweep parameters. Speeds are sampled as continuous random values.
    speed_min: float = 10.0
    speed_max: float = 35.0
    n_speed_samples: int = 3

    # Macrostate clustering.
    n_clusters: int = 4
    random_state: int = 42
    angle_mode: str = "project_raw"  # "project_raw" preserves current Macrostate behavior.

    # Runtime.
    device: str = "auto"
    save_plots: bool = True
    max_robots_to_plot: int = 50

    # Animations. Uses the same Matplotlib FuncAnimation + Ellipse approach as the
    # original StepPrediction notebook.
    # save_animations controls whether separate video-generation cells/methods are enabled.
    # auto_save_animations_during_pipeline=False keeps training/evaluation fast and lets you
    # create videos later without retraining.
    save_animations: bool = True
    auto_save_animations_during_pipeline: bool = False
    animation_fps: int = 10
    animation_max_frames: int | None = 300
    animation_robot_width: float = 60.0
    animation_robot_height: float = 36.0
    # True = fallback to GIF if ffmpeg is absent; False = original GitHub writer="ffmpeg" behavior.
    animation_windows_safe: bool = True
    animation_use_normalized_coords: bool = False

    # Demo experiment selection for one-step/multistep plots and videos.
    # Preserves the original StepPrediction notebook logic where a concrete
    # experiment is selected from test_files, e.g. test_files[2].
    demo_split: str = "test"  # "test", "train", "val", or "all"
    demo_file_index: int = 2
    demo_file_name: str | None = None

    def ensure_dirs(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir, "plots").mkdir(parents=True, exist_ok=True)
        Path(self.output_dir, "tables").mkdir(parents=True, exist_ok=True)
        Path(self.output_dir, "models").mkdir(parents=True, exist_ok=True)
        Path(self.output_dir, "videos").mkdir(parents=True, exist_ok=True)
