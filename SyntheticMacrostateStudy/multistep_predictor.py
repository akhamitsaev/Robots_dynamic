from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import joblib
import numpy as np
import pandas as pd
import torch

from data_preprocessor import compute_kinematics_from_tensor, cyclic_angle_diff_deg
from gnn_model import GNNModel
from graph_utils import build_knn_graph_from_kinematics, node_features_from_kinematics


class MultiStepPredictor:
    """Autoregressive multistep rollout for the trained one-step GNN.

    The predictor uses only predicted states after the seed segment. With the default
    start_step=3, states 0..3 are factual/generated and the first predicted state is 4.
    """

    def __init__(self, config, load_weights: bool = True, allow_untrained: bool = False):
        self.config = config
        self.device = self._resolve_device(getattr(config, "device", "auto"))
        self.model = GNNModel(config).to(self.device)
        self.model.eval()

        self.target_scaler = None
        self.node_feature_scaler = None
        self.edge_feature_scaler = None

        if load_weights:
            self.load_model_and_scalers(config.model_path, allow_untrained=allow_untrained)

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if str(device) == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def load_model_and_scalers(self, model_path: str, allow_untrained: bool = False) -> None:
        path = Path(model_path)
        if not path.exists():
            if allow_untrained:
                print(f"WARNING: model weights not found: {path}. Using randomly initialized model.")
                return
            raise FileNotFoundError(
                f"GNN model weights not found: {path}. Train StepPrediction first or copy "
                "gnn_model_weights.pth into models/."
            )

        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        base = str(path).replace(".pth", "")
        scaler_paths = {
            "target": Path(base + "_target_scaler.pkl"),
            "node": Path(base + "_node_scaler.pkl"),
            "edge": Path(base + "_edge_scaler.pkl"),
        }
        if scaler_paths["target"].exists():
            self.target_scaler = joblib.load(scaler_paths["target"])
        if scaler_paths["node"].exists():
            self.node_feature_scaler = joblib.load(scaler_paths["node"])
        if scaler_paths["edge"].exists():
            self.edge_feature_scaler = joblib.load(scaler_paths["edge"])

        missing = [name for name, p in scaler_paths.items() if not p.exists()]
        if missing:
            print(f"WARNING: missing scaler files {missing}. Prediction will use raw features for those parts.")

    def _normalize_tensor(self, x: torch.Tensor, scaler) -> torch.Tensor:
        if scaler is None:
            return x.to(self.device).float()
        arr = x.detach().cpu().numpy()
        return torch.tensor(scaler.transform(arr), dtype=torch.float32, device=self.device)

    def _inverse_target(self, pred: torch.Tensor) -> torch.Tensor:
        pred_cpu = pred.detach().cpu()
        if self.target_scaler is None:
            return pred_cpu.float()
        return torch.tensor(self.target_scaler.inverse_transform(pred_cpu.numpy()), dtype=torch.float32)

    def predict_next_delta(self, trajectory: torch.Tensor, t: int) -> torch.Tensor:
        """Predicts deltas for state t -> t+1 using trajectory[0..t]."""
        if t < 2:
            raise ValueError("Need at least states 0..2 to compute acceleration features.")

        trajectory = trajectory.to(self.device).float()
        kinematics = compute_kinematics_from_tensor(trajectory[: t + 1], dt=self.config.dt)
        node_features = node_features_from_kinematics(kinematics, t)
        edge_index, edge_features = build_knn_graph_from_kinematics(kinematics, t, self.config.k_neighbors)

        node_features = self._normalize_tensor(node_features, self.node_feature_scaler)
        edge_features = self._normalize_tensor(edge_features, self.edge_feature_scaler)
        edge_index = edge_index.to(self.device)

        with torch.no_grad():
            pred_norm = self.model(node_features, edge_index, edge_features)
        return self._inverse_target(pred_norm)

    def rollout_from_history(self, history_tensor: torch.Tensor, total_steps: int = 600) -> torch.Tensor:
        """Rolls out full trajectory of length total_steps.

        history_tensor shape: [history_len, n_robots, 3]. history_len must be >= 4
        if start_step=3 is used.
        """
        if history_tensor.ndim != 3 or history_tensor.shape[-1] != 3:
            raise ValueError("history_tensor must have shape [history_len, n_robots, 3].")
        if total_steps <= history_tensor.shape[0]:
            return history_tensor[:total_steps].detach().cpu()
        if history_tensor.shape[0] < self.config.start_step + 1:
            raise ValueError(
                f"history_tensor length={history_tensor.shape[0]} is too short for start_step={self.config.start_step}."
            )

        n_robots = history_tensor.shape[1]
        full = torch.zeros((total_steps, n_robots, 3), dtype=torch.float32, device=self.device)
        full[: history_tensor.shape[0]] = history_tensor.to(self.device).float()

        first_t = history_tensor.shape[0] - 1
        for t in range(first_t, total_steps - 1):
            delta = self.predict_next_delta(full[: t + 1], t).to(self.device)
            full[t + 1, :, 0:2] = full[t, :, 0:2] + delta[:, 0:2]
            full[t + 1, :, 2] = (full[t, :, 2] + delta[:, 2]) % 360.0

        return full.detach().cpu()

    def rollout_real_experiment(self, exp_data: Dict[str, Any], start_step: int | None = None, horizon: int | None = None) -> torch.Tensor:
        """Uses factual experiment states 0..start_step and then rolls out autoregressively."""
        tensor = exp_data["tensor"].detach().cpu().float()
        start_step = self.config.start_step if start_step is None else start_step
        total_steps = tensor.shape[0] if horizon is None else min(tensor.shape[0], start_step + 1 + horizon)
        history = tensor[: start_step + 1]
        return self.rollout_from_history(history, total_steps=total_steps)

    def evaluate_on_real_experiments(
        self,
        experiments_dict: Dict[str, Dict[str, Any]],
        file_names: Iterable[str] | None = None,
        start_step: int | None = None,
        horizon: int | None = None,
    ) -> tuple[pd.DataFrame, Dict[str, float]]:
        """Runs multistep rollout on real experiments and returns long predictions + metrics."""
        start_step = self.config.start_step if start_step is None else start_step
        selected = list(file_names) if file_names is not None else list(experiments_dict.keys())
        rows = []

        for file_name in selected:
            exp_data = experiments_dict[file_name]
            real = exp_data["tensor"].detach().cpu().float()
            pred = self.rollout_real_experiment(exp_data, start_step=start_step, horizon=horizon)
            t_max = min(real.shape[0], pred.shape[0])
            local_to_global = exp_data.get("local_to_global", {i: i for i in range(real.shape[1])})

            for t in range(start_step + 1, t_max):
                for i in range(real.shape[1]):
                    rows.append({
                        "file_name": file_name,
                        "slice_id": t,
                        "forecast_horizon": t - start_step,
                        "bot_id": local_to_global.get(i, i),
                        "coord_x_real": float(real[t, i, 0]),
                        "coord_y_real": float(real[t, i, 1]),
                        "angle_real": float(real[t, i, 2]),
                        "coord_x_pred": float(pred[t, i, 0]),
                        "coord_y_pred": float(pred[t, i, 1]),
                        "angle_pred": float(pred[t, i, 2]),
                    })

        df = pd.DataFrame(rows)
        metrics = self.calculate_multistep_metrics(df)
        return df, metrics

    @staticmethod
    def _cyclic_np(pred: np.ndarray, real: np.ndarray) -> np.ndarray:
        delta = pred - real
        delta = np.where(delta > 180.0, delta - 360.0, delta)
        delta = np.where(delta < -180.0, delta + 360.0, delta)
        return delta

    def calculate_multistep_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {}
        ex = df["coord_x_pred"].to_numpy() - df["coord_x_real"].to_numpy()
        ey = df["coord_y_pred"].to_numpy() - df["coord_y_real"].to_numpy()
        ea = self._cyclic_np(df["angle_pred"].to_numpy(), df["angle_real"].to_numpy())
        pos_err = np.sqrt(ex ** 2 + ey ** 2)

        by_h = df.assign(pos_error=pos_err, angle_error_abs=np.abs(ea)).groupby("forecast_horizon")
        horizon_stats = by_h.agg(
            rmse_pos=("pos_error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            mae_pos=("pos_error", "mean"),
            mae_angle=("angle_error_abs", "mean"),
        )
        final_h = int(horizon_stats.index.max())
        final = horizon_stats.loc[final_h]

        return {
            "multistep_rmse_x": float(np.sqrt(np.mean(ex ** 2))),
            "multistep_rmse_y": float(np.sqrt(np.mean(ey ** 2))),
            "multistep_rmse_angle": float(np.sqrt(np.mean(ea ** 2))),
            "multistep_ade_pos": float(np.mean(pos_err)),
            "multistep_fde_pos": float(final["mae_pos"]),
            "multistep_final_horizon": final_h,
            "total_predictions": int(len(df)),
        }
