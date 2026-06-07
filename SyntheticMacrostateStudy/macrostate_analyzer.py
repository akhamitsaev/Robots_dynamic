from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data_preprocessor import compute_kinematics_from_tensor


FEATURE_NAMES = [
    "Polar_Order",
    "Mean_Angle",
    "Coordination_Num",
    "Mean_Distance",
    "Angular_Vel",
    "Rotation_Direction",
    "Rot_Order",
    "Center_Vel",
    "Velocity_Dispersion",
    "Std_Nearest_Dist",
    "Mean_Velocity",
]


class SystemStateAnalyzer:
    """Macrostate features copied from the Macrostate folder logic.

    angle_mode="project_raw" intentionally preserves the current project behavior where
    angles are passed directly into exp(1j*angle). Use angle_mode="degrees" only if you
    intentionally want to treat angle values as degrees.
    """

    def __init__(self, dt: float = 0.1, robot_size: float = 60.0, angle_mode: str = "project_raw"):
        self.dt = dt
        self.robot_size = robot_size
        self.cutoff = robot_size * 2
        self.angle_mode = angle_mode

    def _angles_for_complex(self, angles: torch.Tensor) -> torch.Tensor:
        if self.angle_mode == "degrees":
            return torch.deg2rad(angles)
        return angles

    def compute_polar_order(self, angles: torch.Tensor) -> torch.Tensor:
        a = self._angles_for_complex(angles)
        return torch.abs(torch.mean(torch.exp(1j * a), dim=-1))

    def compute_mean_angle(self, angles: torch.Tensor) -> torch.Tensor:
        a = self._angles_for_complex(angles)
        return torch.angle(torch.mean(torch.exp(1j * a), dim=-1))

    def compute_coordination_number(self, coords: torch.Tensor) -> torch.Tensor:
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        distances = torch.norm(diff, dim=3)
        mask = distances < self.cutoff
        eye_mask = torch.eye(coords.shape[1], device=coords.device, dtype=torch.bool)
        mask[:, eye_mask] = False
        return mask.sum(dim=2).float().mean(dim=1)

    def compute_mean_distance(self, coords: torch.Tensor) -> torch.Tensor:
        n = coords.shape[1]
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        distances = torch.norm(diff, dim=3)
        mask = torch.eye(n, device=coords.device, dtype=torch.bool)
        distances = distances[:, ~mask].view(-1, n, n - 1)
        return torch.mean(distances, dim=(1, 2))

    def compute_std_nearest_distance(self, coords: torch.Tensor) -> torch.Tensor:
        n_robots = coords.shape[1]
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        distances = torch.norm(diff, dim=3)
        mask = torch.eye(n_robots, device=coords.device, dtype=torch.bool)
        distances_masked = distances.masked_fill(mask, 1e6)
        nearest_dist = torch.min(distances_masked, dim=2)[0]
        return torch.std(nearest_dist, dim=1)

    def compute_angular_velocity_system(self, coords: torch.Tensor, velocities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centers = torch.mean(coords, dim=1, keepdim=True)
        radii = coords - centers
        distances = torch.norm(radii, dim=2)
        radial_vectors = radii / (distances.unsqueeze(2) + 1e-10)
        tangential_vectors = torch.stack([-radial_vectors[:, :, 1], radial_vectors[:, :, 0]], dim=2)
        tangential_vel = torch.sum(velocities * tangential_vectors, dim=2)
        angular_vel = tangential_vel / (distances + 1e-10)
        return torch.mean(angular_vel, dim=1), torch.sign(torch.mean(angular_vel, dim=1))

    def compute_rotational_order(self, coords: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
        centers = torch.mean(coords, dim=1, keepdim=True)
        radii = coords - centers
        radial_vectors = radii / (torch.norm(radii, dim=2, keepdim=True) + 1e-10)
        tangential_vectors = torch.stack([-radial_vectors[:, :, 1], radial_vectors[:, :, 0]], dim=2)
        tangential_velocities = torch.sum(velocities * tangential_vectors, dim=2)
        return torch.mean(torch.abs(tangential_velocities), dim=1)

    def compute_center_velocity(self, velocities: torch.Tensor) -> torch.Tensor:
        return torch.mean(velocities, dim=1)

    def compute_velocity_dispersion(self, velocities: torch.Tensor) -> torch.Tensor:
        speed = torch.norm(velocities, dim=2)
        speed_mean = torch.mean(speed, dim=1)
        speed_std = torch.std(speed, dim=1)
        return speed_std / (speed_mean + 1e-10)

    def compute_mean_velocity(self, velocities: torch.Tensor) -> torch.Tensor:
        speed = torch.norm(velocities, dim=2)
        return torch.mean(speed, dim=1)

    def extract_features_from_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = tensor.float()
        kinematics = compute_kinematics_from_tensor(tensor, self.dt)
        coords = tensor[:, :, :2]
        angles = tensor[:, :, 2]
        velocities = kinematics["velocities"]

        angular_vel, rotation_dir = self.compute_angular_velocity_system(coords, velocities)
        center_vel = self.compute_center_velocity(velocities)

        features = torch.stack([
            self.compute_polar_order(angles),
            self.compute_mean_angle(angles),
            self.compute_coordination_number(coords),
            self.compute_mean_distance(coords),
            angular_vel,
            rotation_dir,
            self.compute_rotational_order(coords, velocities),
            torch.norm(center_vel, dim=1),
            self.compute_velocity_dispersion(velocities),
            self.compute_std_nearest_distance(coords),
            self.compute_mean_velocity(velocities),
        ], dim=1)
        return features.detach().cpu().numpy()

    def extract_features_batch(self, experiments_dict: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        return {
            file_name: self.extract_features_from_tensor(exp_data["tensor"])
            for file_name, exp_data in experiments_dict.items()
        }


@dataclass
class ClusterInterpretation:
    cluster: int
    samples: int
    state_type: str
    po: float
    angular_vel: float
    coord_num: float
    mean_dist: float
    vel_disp: float


class MacrostateClassifier:
    """KMeans macrostate classifier trained on fact-data features."""

    def __init__(self, n_clusters: int = 4, random_state: int = 42, robot_size: float = 60.0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.robot_size = robot_size
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_stats: dict[int, dict] = {}
        self.interpretation_df: pd.DataFrame | None = None

    @staticmethod
    def features_dict_to_array(features_dict: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arrays, files, steps = [], [], []
        for file_name, arr in features_dict.items():
            arrays.append(arr)
            files.extend([file_name] * len(arr))
            steps.extend(range(len(arr)))
        return np.vstack(arrays), np.array(files), np.array(steps)

    def fit(self, features_array: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.fit_transform(features_array)
        labels = self.kmeans.fit_predict(x_scaled)
        self.cluster_stats = self.analyze_clusters(features_array, labels)
        self.interpretation_df = self.create_state_interpretation_table(self.cluster_stats)
        return labels

    def predict(self, features_array: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(self.scaler.transform(features_array))

    def predict_dataframe(self, features_array: np.ndarray, steps: np.ndarray | None = None) -> pd.DataFrame:
        labels = self.predict(features_array)
        df = pd.DataFrame(features_array, columns=FEATURE_NAMES)
        df["cluster"] = labels
        if steps is not None:
            df["step"] = steps
        if self.interpretation_df is not None:
            mapping = self.interpretation_df.set_index("Cluster")["State_Type"].to_dict()
            df["state_type"] = df["cluster"].map(mapping)
            df["is_micelle"] = df["state_type"].str.contains("Мицелла|Regular Structure", regex=True, na=False)
        else:
            df["state_type"] = "unknown"
            df["is_micelle"] = False
        return df

    def analyze_clusters(self, features_array: np.ndarray, labels: np.ndarray) -> dict[int, dict]:
        stats = {}
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            if not np.any(mask):
                continue
            arr = features_array[mask]
            stats[cluster_id] = {
                "count": int(mask.sum()),
                "means": arr.mean(axis=0),
                "stds": arr.std(axis=0),
                "feature_names": FEATURE_NAMES,
            }
        return stats

    def create_state_interpretation_table(self, cluster_stats: dict[int, dict]) -> pd.DataFrame:
        rows = []
        for cluster_id, stats in cluster_stats.items():
            means = stats["means"]
            po = means[0]
            coord_num = means[2]
            mean_dist = means[3]
            angular_vel = means[4]
            rotation_dir = means[5]
            center_vel = means[7]
            vel_disp = means[8]
            std_nearest = means[9]
            mean_vel = means[10]

            if po > 0.7 and mean_dist < self.robot_size * 2:
                state_type = "Compact Swarm (Рой)"
            elif abs(angular_vel) > 0.1 and rotation_dir > 0:
                state_type = "Clockwise Rotation (Правое вращение)"
            elif abs(angular_vel) > 0.1 and rotation_dir < 0:
                state_type = "Counter-Clockwise Rotation (Левое вращение)"
            elif coord_num > 4 and std_nearest < self.robot_size * 0.5:
                state_type = "Regular Structure (Мицелла)"
            elif po < 0.3 and vel_disp > 0.8:
                state_type = "Chaotic Motion (Хаос)"
            elif mean_dist > self.robot_size * 3:
                state_type = "Sparse Distribution (Разреженное)"
            elif center_vel > mean_vel * 0.7:
                state_type = "Collective Translation (Поступательное движение)"
            else:
                state_type = "Mixed/Transition State (Смешанное)"

            rows.append({
                "Cluster": cluster_id,
                "Samples": stats["count"],
                "State_Type": state_type,
                "PO": round(float(po), 3),
                "Angular_Vel": round(float(angular_vel), 3),
                "Coord_Num": round(float(coord_num), 1),
                "Mean_Dist": round(float(mean_dist), 1),
                "Vel_Disp": round(float(vel_disp), 3),
            })
        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "robot_size": self.robot_size,
            "scaler": self.scaler,
            "kmeans": self.kmeans,
            "cluster_stats": self.cluster_stats,
            "interpretation_df": self.interpretation_df,
            "feature_names": FEATURE_NAMES,
        }, path)

    @classmethod
    def load(cls, path: str) -> "MacrostateClassifier":
        payload = joblib.load(path)
        obj = cls(payload["n_clusters"], payload["random_state"], payload["robot_size"])
        obj.scaler = payload["scaler"]
        obj.kmeans = payload["kmeans"]
        obj.cluster_stats = payload["cluster_stats"]
        obj.interpretation_df = payload["interpretation_df"]
        return obj
