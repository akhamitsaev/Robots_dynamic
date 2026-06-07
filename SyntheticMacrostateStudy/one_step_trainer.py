from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from data_preprocessor import DataPreprocessor
from gnn_trainer import GNNTrainer
from visualization import animate_real_vs_predicted_from_df


def save_one_step_animation_from_predictions(
    config,
    predictions_path: str | Path | None = None,
    output_path: str | Path | None = None,
    file_name: str | None = None,
) -> Path | None:
    """Save one-step real-vs-predicted animation from an existing predictions table.

    This function does not train or run inference. It only reads the saved
    one-step predictions produced by GNNTrainer and creates an animation.
    If file_name is provided, it preserves the original GitHub notebook logic:
    filter df by one concrete experiment before plotting/video export.
    """
    predictions_path = Path(predictions_path or config.results_save_path)
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"One-step predictions not found: {predictions_path}. Run train_one_step_model(...) first "
            "or run one-step inference to create results/gnn_predictions.parquet."
        )

    preds_df = pd.read_parquet(predictions_path)
    if preds_df.empty:
        print("One-step animation skipped: empty predictions table.")
        return None

    if file_name is not None:
        preds_df = preds_df[preds_df["file_name"] == file_name].copy()
        if preds_df.empty:
            available = pd.read_parquet(predictions_path, columns=["file_name"])["file_name"].drop_duplicates().head(10).tolist()
            raise ValueError(
                f"No one-step predictions found for file_name={file_name!r}. "
                f"First available file_names: {available}"
            )

    df_anim = preds_df.rename(
        columns={
            "gnn_coord_x_pred": "coord_x_pred",
            "gnn_coord_y_pred": "coord_y_pred",
            "gnn_angle_pred": "angle_pred",
        }
    )
    video_dir = Path(getattr(config, "video_dir", str(Path(getattr(config, "output_dir", "results/research")) / "videos")))
    video_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        suffix = "" if file_name is None else "_" + safe_name_local(file_name)
        output_path = video_dir / f"one_step_real_vs_predicted{suffix}.mp4"
    output_path = Path(output_path)

    title = "One-step: Real vs GNN predicted" if file_name is None else f"One-step: Real vs GNN predicted\n{file_name}"
    return animate_real_vs_predicted_from_df(
        df_anim,
        output_path,
        title=title,
        fps=getattr(config, "animation_fps", 10),
        max_frames=getattr(config, "animation_max_frames", 300),
        robot_width=getattr(config, "animation_robot_width", 60.0),
        robot_height=getattr(config, "animation_robot_height", 36.0),
        normalize=getattr(config, "animation_use_normalized_coords", False),
        windows_safe=getattr(config, "animation_windows_safe", True),
    )


def safe_name_local(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))[:120]


class OneStepGNNTrainer:
    """Autonomous wrapper around the original GitHub StepPrediction GNNTrainer.

    The actual model architecture, GraphDataset, scaler fitting, batch normalization,
    training loop, evaluation, denormalization and saved prediction table are delegated
    to gnn_trainer.GNNTrainer, copied from the GitHub StepPrediction implementation.

    This wrapper only adds:
    - loading fact data inside the autonomous package;
    - configurable train_sample_ratio before the original train/val/test split;
    - optional one-step video after original predictions are saved.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.device = self._resolve_device(getattr(config, "device", "auto"))
        self.trainer: GNNTrainer | None = None

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if str(device) == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def split_files(self, experiments_dict: Dict[str, Dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
        files = list(experiments_dict.keys())
        if len(files) < 3:
            raise ValueError("Need at least 3 experiments/files for train/val/test split.")

        sample_ratio = float(getattr(self.config, "train_sample_ratio", 1.0))
        if not (0.0 < sample_ratio <= 1.0):
            raise ValueError(f"train_sample_ratio must be in (0, 1], got {sample_ratio}")
        if sample_ratio < 1.0:
            rng = np.random.default_rng(getattr(self.config, "random_state", 42))
            sample_size = max(3, int(round(len(files) * sample_ratio)))
            sample_size = min(sample_size, len(files))
            files = list(rng.choice(files, size=sample_size, replace=False))
            print(f"Using train_sample_ratio={sample_ratio:.3f}: {sample_size}/{len(experiments_dict)} fact files")

        train_val_files, test_files = train_test_split(
            files,
            test_size=getattr(self.config, "test_size", 0.2),
            random_state=getattr(self.config, "random_state", 42),
        )
        val_fraction = getattr(self.config, "val_size", 0.1) / max(1e-9, (1.0 - getattr(self.config, "test_size", 0.2)))
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=val_fraction,
            random_state=getattr(self.config, "random_state", 42),
        )
        return list(train_files), list(val_files), list(test_files)

    def train_from_data(self, experiments_dict=None) -> dict[str, float]:
        self._seed_everything(getattr(self.config, "random_state", 42))

        preprocessor = DataPreprocessor(self.config.data_path, dt=self.config.dt)
        # Original StepPrediction DataPreprocessor expects .config.dt inside compute_kinematics_vectorized.
        preprocessor.config = self.config

        if experiments_dict is None:
            experiments_dict = preprocessor.load_and_vectorize()
        kinematics_dict = preprocessor.compute_kinematics_vectorized(experiments_dict)

        train_files, val_files, test_files = self.split_files(experiments_dict)
        print("One-step split:")
        print(f"  train files: {len(train_files)}")
        print(f"  val files:   {len(val_files)}")
        print(f"  test files:  {len(test_files)}")

        # Save the exact split so demo visualizations can use the same logic
        # as the original GitHub notebook: e.g. file_name = test_files[2].
        split_path = Path(getattr(self.config, "output_dir", "results/research")) / "tables" / "one_step_split.json"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(
            json.dumps(
                {"train_files": train_files, "val_files": val_files, "test_files": test_files},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self.trainer = GNNTrainer(self.config, self.device)
        metrics = self.trainer.train_and_evaluate(experiments_dict, kinematics_dict, train_files, val_files, test_files)

        # Video is intentionally not created here by default.
        # Use pipeline.save_one_step_video() after training to generate it from
        # the saved predictions table without retraining. For legacy one-command
        # behavior, set config.auto_save_animations_during_pipeline=True.
        if getattr(self.config, "save_animations", True) and getattr(self.config, "auto_save_animations_during_pipeline", False):
            save_one_step_animation_from_predictions(self.config)

        report = {
            **metrics,
            "train_sample_ratio": getattr(self.config, "train_sample_ratio", 1.0),
            "device": str(self.device),
            "model_path": self.config.model_save_path,
            "results_path": self.config.results_save_path,
        }
        out_dir = Path(getattr(self.config, "output_dir", "results/research"))
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "one_step_metrics.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
