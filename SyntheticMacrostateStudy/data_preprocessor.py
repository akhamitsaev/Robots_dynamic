from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch


@dataclass
class DataPreprocessor:
    """Loads fact experiments and computes kinematic tensors.

    Expected parquet columns:
    file_name, slice_id, bot_id, coord_x, coord_y, angle
    """

    data_path: str
    dt: float = 0.1
    device: str | torch.device = "auto"

    def __post_init__(self) -> None:
        if self.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device)

    def load_and_vectorize(self) -> Dict[str, Dict[str, Any]]:
        """Векторизованная загрузка данных в torch тензоры.

        This intentionally mirrors StepPrediction/data_preprocessor.py from the
        GitHub project so one-step training receives the same tensors.
        """
        print("🚀 Векторизованная загрузка данных...")
        df = pd.read_parquet(self.data_path)

        experiments: Dict[str, Dict[str, Any]] = {}
        for file_name, exp_df in df.groupby('file_name'):
            exp_df = exp_df.sort_values(['slice_id', 'bot_id'])
            unique_bots = sorted(exp_df['bot_id'].unique())
            global_to_local = {bot_id: idx for idx, bot_id in enumerate(unique_bots)}
            local_to_global = {idx: bot_id for idx, bot_id in enumerate(unique_bots)}

            n_robots = len(unique_bots)
            n_timesteps = exp_df['slice_id'].nunique()

            tensor_data = exp_df[['coord_x', 'coord_y', 'angle']].values
            tensor = torch.tensor(
                tensor_data.reshape(n_timesteps, n_robots, 3),
                device=self.device,
                dtype=torch.float32,
            )

            experiments[file_name] = {
                'tensor': tensor,
                'global_to_local': global_to_local,
                'local_to_global': local_to_global,
                'n_robots': n_robots,
                'n_timesteps': n_timesteps,
            }

        print(f"📊 Загружено {len(experiments)} экспериментов на {self.device}")
        return experiments

    def compute_kinematics_vectorized(self, experiments_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        kinematics_dict = {}
        for file_name, exp_data in experiments_dict.items():
            kinematics_dict[file_name] = compute_kinematics_from_tensor(exp_data["tensor"], self.dt)
        return kinematics_dict


def cyclic_angle_diff_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Smallest signed difference a-b for degree angles."""
    delta = a - b
    delta = torch.where(delta > 180.0, delta - 360.0, delta)
    delta = torch.where(delta < -180.0, delta + 360.0, delta)
    return delta


def compute_kinematics_from_tensor(tensor: torch.Tensor, dt: float = 0.1) -> Dict[str, torch.Tensor]:
    """Computes coords, angles, velocities and accelerations for a [T, N, 3] tensor."""
    coords = tensor[:, :, 0:2]
    angles = tensor[:, :, 2]

    velocities = torch.zeros_like(coords)
    velocities[1:] = (coords[1:] - coords[:-1]) / dt

    accelerations = torch.zeros_like(coords)
    accelerations[2:] = (velocities[2:] - velocities[1:-1]) / dt

    # Match the original StepPrediction implementation exactly:
    # angular velocities use raw angle difference, without cyclic correction.
    angular_velocities = torch.zeros_like(angles)
    angular_velocities[1:] = (angles[1:] - angles[:-1]) / dt

    angular_accelerations = torch.zeros_like(angles)
    angular_accelerations[2:] = (angular_velocities[2:] - angular_velocities[1:-1]) / dt

    return {
        "coords": coords,
        "angles": angles,
        "velocities": velocities,
        "accelerations": accelerations,
        "angular_velocities": angular_velocities,
        "angular_accelerations": angular_accelerations,
    }


def dataframe_from_tensor(tensor: torch.Tensor, file_name: str = "synthetic", start_step: int = 0) -> pd.DataFrame:
    """Converts [T, N, 3] trajectory tensor to long dataframe."""
    tensor_cpu = tensor.detach().cpu().numpy()
    records = []
    for t in range(tensor_cpu.shape[0]):
        for bot_id in range(tensor_cpu.shape[1]):
            records.append({
                "file_name": file_name,
                "slice_id": start_step + t,
                "bot_id": bot_id,
                "coord_x": float(tensor_cpu[t, bot_id, 0]),
                "coord_y": float(tensor_cpu[t, bot_id, 1]),
                "angle": float(tensor_cpu[t, bot_id, 2]),
            })
    return pd.DataFrame(records)
