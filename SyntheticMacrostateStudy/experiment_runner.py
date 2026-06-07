from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from tqdm.auto import tqdm

from config import ResearchConfig
from data_preprocessor import DataPreprocessor, dataframe_from_tensor
from initial_state_generator import InitialConditionSpec, InitialStateGenerator
from macrostate_analyzer import FEATURE_NAMES, MacrostateClassifier, SystemStateAnalyzer
from multistep_predictor import MultiStepPredictor
from one_step_trainer import OneStepGNNTrainer, save_one_step_animation_from_predictions
from visualization import (
    animate_real_vs_predicted_from_df,
    animate_synthetic_trajectory_from_df,
    plot_cluster_timeline,
    plot_micelle_summary,
    plot_synthetic_research_overview,
    plot_multistep_error_by_horizon,
    plot_real_vs_predicted_trajectories,
)


class ResearchPipeline:
    """End-to-end pipeline:
    fact data -> KMeans macrostates;
    trained one-step GNN -> multistep rollout;
    synthetic initial conditions -> 600-step macrostate study.
    """

    def __init__(self, config: ResearchConfig):
        self.config = config
        # Keep autonomous aliases and original StepPrediction aliases synchronized.
        self.config.model_save_path = self.config.model_path
        self.config.ensure_dirs()
        self.analyzer = SystemStateAnalyzer(
            dt=config.dt,
            robot_size=config.robot_size,
            angle_mode=config.angle_mode,
        )
        self.macro_classifier: MacrostateClassifier | None = None
        self.predictor: MultiStepPredictor | None = None
        self.one_step_metrics: dict | None = None

    @property
    def output_dir(self) -> Path:
        return Path(self.config.output_dir)

    def load_fact_data(self):
        preprocessor = DataPreprocessor(self.config.data_path, dt=self.config.dt, device=getattr(self.config, "data_device", "cpu"))
        experiments = preprocessor.load_and_vectorize()
        return experiments

    def get_one_step_split(self, experiments_dict=None) -> dict[str, list[str]]:
        """Return the one-step train/val/test split used for demo selection.

        If training already saved results/research/tables/one_step_split.json,
        read it. Otherwise reconstruct the split using the same deterministic
        logic as OneStepGNNTrainer.split_files(...). This preserves the original
        GitHub notebook logic where a concrete experiment is selected as
        test_files[2] before plots/video are created.
        """
        split_path = self.output_dir / "tables" / "one_step_split.json"
        if split_path.exists():
            return json.loads(split_path.read_text(encoding="utf-8"))

        if experiments_dict is None:
            experiments_dict = self.load_fact_data()
        trainer = OneStepGNNTrainer(self.config)
        train_files, val_files, test_files = trainer.split_files(experiments_dict)
        return {"train_files": train_files, "val_files": val_files, "test_files": test_files}

    def select_demo_file(
        self,
        experiments_dict=None,
        file_name: str | None = None,
        file_index: int | None = None,
        split: str | None = None,
    ) -> str:
        """Select a concrete demo experiment for plots/video.

        Priority:
        1. explicit file_name;
        2. config.demo_file_name;
        3. split[file_index], defaulting to test_files[2] like the GitHub notebook.
        """
        if file_name is not None:
            return file_name
        if getattr(self.config, "demo_file_name", None):
            return str(self.config.demo_file_name)

        split = split or getattr(self.config, "demo_split", "test")
        file_index = getattr(self.config, "demo_file_index", 2) if file_index is None else file_index

        split_dict = self.get_one_step_split(experiments_dict)
        key_map = {"train": "train_files", "val": "val_files", "test": "test_files", "all": "all_files"}
        if split == "all":
            if experiments_dict is None:
                experiments_dict = self.load_fact_data()
            candidates = list(experiments_dict.keys())
        else:
            key = key_map.get(split, split)
            if key not in split_dict:
                raise ValueError(f"Unknown demo split {split!r}. Use 'test', 'train', 'val', or 'all'.")
            candidates = split_dict[key]

        if not candidates:
            raise ValueError(f"No files available in demo split {split!r}.")
        if file_index >= len(candidates) or file_index < -len(candidates):
            raise IndexError(
                f"demo_file_index={file_index} is out of range for split {split!r} with {len(candidates)} files."
            )
        selected = str(candidates[file_index])
        print(f"Selected demo experiment: {selected} ({split}_files[{file_index}])")
        return selected


    def train_one_step_model(self, experiments_dict=None, force: bool = False) -> dict:
        """Train the one-step GNN inside this autonomous package.

        The trained weights are saved to config.model_path and then reused by
        MultiStepPredictor for autoregressive multistep rollout. If
        skip_training_if_model_exists=True, existing weights are reused unless
        force=True.
        """
        model_path = Path(self.config.model_path)
        if model_path.exists() and self.config.skip_training_if_model_exists and not force:
            print(f"One-step model already exists: {model_path}. Skipping training.")
            self.one_step_metrics = {"model_path": str(model_path), "training_skipped": True}
            return self.one_step_metrics

        if experiments_dict is None:
            experiments_dict = self.load_fact_data()

        trainer = OneStepGNNTrainer(self.config)
        self.one_step_metrics = trainer.train_from_data(experiments_dict)
        return self.one_step_metrics

    def ensure_one_step_model(self, experiments_dict=None, force: bool = False) -> dict:
        """Make sure config.model_path exists; train it if needed and allowed."""
        model_path = Path(self.config.model_path)
        if model_path.exists() and not force:
            return {"model_path": str(model_path), "training_skipped": True}
        if not self.config.train_one_step and not force:
            raise FileNotFoundError(
                f"GNN weights not found: {model_path}. Set config.train_one_step=True or call train_one_step_model(...)."
            )
        return self.train_one_step_model(experiments_dict=experiments_dict, force=force)

    def fit_macrostate_classifier(self, experiments_dict=None, save: bool = True) -> MacrostateClassifier:
        if experiments_dict is None:
            experiments_dict = self.load_fact_data()
        print("Extracting fact-data macrostate features...")
        features_dict = self.analyzer.extract_features_batch(experiments_dict)
        features_array, file_names, time_steps = MacrostateClassifier.features_dict_to_array(features_dict)

        classifier = MacrostateClassifier(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
            robot_size=self.config.robot_size,
        )
        labels = classifier.fit(features_array)
        self.macro_classifier = classifier

        # Match the original Macrostate/StateClusterer.save_results logic:
        # one row per time slice with the 11 macrostate features, cluster, file_name, time_step.
        results_df = pd.DataFrame(features_array, columns=FEATURE_NAMES)
        results_df["cluster"] = labels
        results_df["file_name"] = file_names
        results_df["time_step"] = time_steps

        tables_dir = self.output_dir / "tables"
        results_df.to_csv(tables_dir / "fact_cluster_results.csv", index=False)
        classifier.interpretation_df.to_csv(tables_dir / "fact_state_interpretation.csv", index=False)
        if save:
            classifier.save(self.output_dir / "models" / "macrostate_kmeans.pkl")
        print(classifier.interpretation_df.to_string(index=False))
        return classifier

    def load_macrostate_classifier(self, path: str | None = None) -> MacrostateClassifier:
        path = path or str(self.output_dir / "models" / "macrostate_kmeans.pkl")
        self.macro_classifier = MacrostateClassifier.load(path)
        return self.macro_classifier

    def load_predictor(self, allow_untrained: bool = False, train_if_missing: bool = True) -> MultiStepPredictor:
        if train_if_missing and not Path(self.config.model_path).exists() and not allow_untrained:
            self.ensure_one_step_model()
        self.predictor = MultiStepPredictor(self.config, load_weights=True, allow_untrained=allow_untrained)
        return self.predictor

    def evaluate_multistep_on_fact_data(
        self,
        experiments_dict=None,
        file_names: Iterable[str] | None = None,
        horizon: int | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        if self.predictor is None:
            self.load_predictor()
        if experiments_dict is None:
            experiments_dict = self.load_fact_data()

        predictions_df, metrics = self.predictor.evaluate_on_real_experiments(
            experiments_dict,
            file_names=file_names,
            start_step=self.config.start_step,
            horizon=horizon,
        )
        tables_dir = self.output_dir / "tables"
        predictions_df.to_parquet(tables_dir / "multistep_fact_predictions.parquet", index=False)
        pd.DataFrame([metrics]).to_csv(tables_dir / "multistep_fact_metrics.csv", index=False)
        plot_multistep_error_by_horizon(predictions_df, self.output_dir / "plots" / "multistep_error_by_horizon.png")

        # Save trajectory plot for the demo experiment. This mirrors the original
        # GitHub notebook logic: select a concrete file, e.g. test_files[2], and
        # then build plots/video only for that experiment.
        try:
            fn = self.select_demo_file(experiments_dict=experiments_dict) if file_names is None else list(file_names)[0]
        except Exception:
            fn = str(predictions_df["file_name"].iloc[0]) if not predictions_df.empty else None
        if fn is not None and fn in experiments_dict:
            pred_tensor = self.predictor.rollout_real_experiment(experiments_dict[fn], start_step=self.config.start_step, horizon=horizon)
            real_tensor = experiments_dict[fn]["tensor"].detach().cpu()[: pred_tensor.shape[0]]
            plot_real_vs_predicted_trajectories(
                real_tensor,
                pred_tensor,
                save_path=str(self.output_dir / "plots" / f"real_vs_predicted_{safe_name(fn)}.png"),
                title=f"Real vs predicted: {fn}",
                max_robots=self.config.max_robots_to_plot,
            )

            if (
                getattr(self.config, "save_animations", True)
                and getattr(self.config, "auto_save_animations_during_pipeline", False)
                and not predictions_df.empty
            ):
                self.save_multistep_video(predictions_df=predictions_df, file_name=fn)

        return predictions_df, metrics

    def predict_one_step_for_file(self, experiments_dict, file_name: str) -> pd.DataFrame:
        """Run one-step teacher-forcing inference only for one experiment.

        This is a fallback for the common workflow where the model checkpoint
        already exists but the full results/gnn_predictions.parquet was not
        exported because it is slow. It keeps the same row schema as
        GNNTrainer._predict_all for the selected file only.
        """
        if file_name not in experiments_dict:
            raise KeyError(f"file_name={file_name!r} not found in experiments_dict")
        if self.predictor is None:
            self.load_predictor(train_if_missing=False)

        exp_data = experiments_dict[file_name]
        tensor = exp_data["tensor"].detach().cpu().float()
        n_timesteps, n_robots, _ = tensor.shape
        local_to_global = exp_data.get("local_to_global", {i: i for i in range(n_robots)})
        rows = []

        for t in range(2, n_timesteps - 1):
            # Teacher forcing one-step: predict t -> t+1 from factual states 0..t.
            delta = self.predictor.predict_next_delta(tensor[: t + 1], t)
            coord_pred = tensor[t, :, 0:2] + delta[:, 0:2]
            angle_pred = (tensor[t, :, 2] + delta[:, 2]) % 360.0

            for i in range(n_robots):
                rows.append({
                    "file_name": file_name,
                    "slice_id": t,
                    "bot_id": local_to_global.get(i, i) if hasattr(local_to_global, "get") else local_to_global[i],
                    "gnn_coord_x_pred": float(coord_pred[i, 0]),
                    "gnn_coord_y_pred": float(coord_pred[i, 1]),
                    "gnn_angle_pred": float(angle_pred[i]),
                    "coord_x_real": float(tensor[t + 1, i, 0]),
                    "coord_y_real": float(tensor[t + 1, i, 1]),
                    "angle_real": float(tensor[t + 1, i, 2]),
                })
        return pd.DataFrame(rows)

    def save_one_step_video(
        self,
        predictions_path: str | Path | None = None,
        output_path: str | Path | None = None,
        file_name: str | None = None,
        file_index: int | None = None,
        split: str | None = None,
        experiments_dict=None,
    ) -> Path | None:
        """Create one-step animation for one concrete experiment after training.

        This preserves the original GitHub notebook logic:
            sub_df = df[df['file_name'] == test_files[2]]
        By default it selects test_files[2]. You can override with file_name.
        The function does not retrain and does not rerun model inference; it
        reads config.results_save_path produced by one-step prediction export.
        """
        if not getattr(self.config, "save_animations", True):
            print("One-step video skipped because config.save_animations=False.")
            return None
        selected_file = self.select_demo_file(
            experiments_dict=experiments_dict,
            file_name=file_name,
            file_index=file_index,
            split=split,
        )
        predictions_path = Path(predictions_path or self.config.results_save_path)
        if predictions_path.exists():
            return save_one_step_animation_from_predictions(
                self.config,
                predictions_path=predictions_path,
                output_path=output_path,
                file_name=selected_file,
            )

        # Fallback: no full one-step predictions table. Generate only the selected
        # demo experiment from the trained checkpoint, without training and without
        # exporting predictions for all files.
        print(
            f"One-step predictions table not found: {predictions_path}. "
            f"Running one-step inference only for demo file: {selected_file}"
        )
        if experiments_dict is None:
            experiments_dict = self.load_fact_data()
        df_file = self.predict_one_step_for_file(experiments_dict, selected_file).rename(
            columns={
                "gnn_coord_x_pred": "coord_x_pred",
                "gnn_coord_y_pred": "coord_y_pred",
                "gnn_angle_pred": "angle_pred",
            }
        )
        videos_dir = self.output_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_path or (videos_dir / f"one_step_real_vs_predicted_{safe_name(selected_file)}.mp4"))
        return animate_real_vs_predicted_from_df(
            df_file,
            save_path=output_path,
            title=f"One-step: real vs predicted\n{selected_file}",
            fps=getattr(self.config, "animation_fps", 10),
            max_frames=getattr(self.config, "animation_max_frames", None),
            robot_width=getattr(self.config, "animation_robot_width", self.config.robot_size),
            robot_height=getattr(self.config, "animation_robot_height", self.config.robot_size * 0.6),
            normalize=getattr(self.config, "animation_use_normalized_coords", False),
            windows_safe=getattr(self.config, "animation_windows_safe", True),
        )

    def save_multistep_video(
        self,
        predictions_df: pd.DataFrame | None = None,
        predictions_path: str | Path | None = None,
        file_name: str | None = None,
        file_index: int | None = None,
        split: str | None = None,
        experiments_dict=None,
        output_path: str | Path | None = None,
    ) -> Path | None:
        """Create multistep animation after evaluation, without rerunning rollout.

        Uses an in-memory predictions_df if provided; otherwise reads
        output_dir/tables/multistep_fact_predictions.parquet.
        """
        if not getattr(self.config, "save_animations", True):
            print("Multistep video skipped because config.save_animations=False.")
            return None

        if predictions_df is None:
            predictions_path = Path(predictions_path or (self.output_dir / "tables" / "multistep_fact_predictions.parquet"))
            if not predictions_path.exists():
                raise FileNotFoundError(
                    f"Multistep predictions not found: {predictions_path}. Run evaluate_multistep_on_fact_data(...) first."
                )
            predictions_df = pd.read_parquet(predictions_path)

        if predictions_df.empty:
            print("Multistep animation skipped: empty predictions table.")
            return None

        if file_name is None:
            try:
                file_name = self.select_demo_file(
                    experiments_dict=experiments_dict,
                    file_name=None,
                    file_index=file_index,
                    split=split,
                )
            except Exception:
                # Fallback for old prediction tables when the split cannot be reconstructed.
                file_name = str(predictions_df["file_name"].iloc[0])
                print(f"Selected demo experiment from predictions table: {file_name}")

        df_anim = predictions_df[predictions_df["file_name"] == file_name].copy()
        if df_anim.empty:
            available = predictions_df["file_name"].drop_duplicates().head(10).tolist()
            raise ValueError(
                f"No multistep predictions found for file_name={file_name!r}. "
                f"First available file_names: {available}"
            )

        videos_dir = self.output_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_path or (videos_dir / f"multistep_real_vs_predicted_{safe_name(file_name)}.mp4"))

        return animate_real_vs_predicted_from_df(
            df_anim,
            save_path=output_path,
            title=f"Multistep: real vs predicted\n{file_name}",
            fps=getattr(self.config, "animation_fps", 10),
            max_frames=getattr(self.config, "animation_max_frames", None),
            robot_width=getattr(self.config, "animation_robot_width", self.config.robot_size),
            robot_height=getattr(self.config, "animation_robot_height", self.config.robot_size * 0.6),
            normalize=getattr(self.config, "animation_use_normalized_coords", False),
            windows_safe=getattr(self.config, "animation_windows_safe", True),
        )

    def classify_trajectory(self, trajectory: torch.Tensor) -> pd.DataFrame:
        if self.macro_classifier is None:
            self.load_macrostate_classifier()
        features = self.analyzer.extract_features_from_tensor(trajectory)
        steps = torch.arange(trajectory.shape[0]).numpy()
        return self.macro_classifier.predict_dataframe(features, steps=steps)

    def save_synthetic_trajectory(
        self,
        trajectory: torch.Tensor,
        experiment_id: str,
        save_pt: bool = True,
        save_parquet: bool = True,
    ) -> dict[str, Path]:
        """Save synthetic rollout as tensor and fact-data-style parquet.

        Parquet columns match fact data:
        file_name, slice_id, bot_id, coord_x, coord_y, angle.
        """
        saved: dict[str, Path] = {}
        safe_id = safe_name(experiment_id)

        if save_pt:
            traj_dir = self.output_dir / "trajectories"
            traj_dir.mkdir(parents=True, exist_ok=True)
            pt_path = traj_dir / f"{safe_id}.pt"
            torch.save(trajectory.detach().cpu(), pt_path)
            saved["pt"] = pt_path

        if save_parquet:
            parquet_dir = self.output_dir / "synthetic_trajectories"
            parquet_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = parquet_dir / f"{safe_id}.parquet"
            df = dataframe_from_tensor(trajectory.detach().cpu(), file_name=experiment_id, start_step=0)
            df.to_parquet(parquet_path, index=False)
            saved["parquet"] = parquet_path

        return saved

    def load_synthetic_trajectory(self, experiment_id: str) -> torch.Tensor:
        """Load a saved synthetic trajectory from .pt or fact-data-style parquet."""
        safe_id = safe_name(experiment_id)
        pt_path = self.output_dir / "trajectories" / f"{safe_id}.pt"
        if pt_path.exists():
            return torch.load(pt_path, map_location="cpu")

        parquet_path = self.output_dir / "synthetic_trajectories" / f"{safe_id}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Synthetic trajectory not found for experiment_id={experiment_id!r}. "
                f"Checked: {pt_path} and {parquet_path}"
            )
        df = pd.read_parquet(parquet_path).sort_values(["slice_id", "bot_id"])
        slice_ids = sorted(df["slice_id"].unique())
        bot_ids = sorted(df["bot_id"].unique())
        tensor = torch.zeros((len(slice_ids), len(bot_ids), 3), dtype=torch.float32)
        slice_to_i = {s: i for i, s in enumerate(slice_ids)}
        bot_to_i = {b: i for i, b in enumerate(bot_ids)}
        for row in df.itertuples(index=False):
            tensor[slice_to_i[row.slice_id], bot_to_i[row.bot_id], 0] = float(row.coord_x)
            tensor[slice_to_i[row.slice_id], bot_to_i[row.bot_id], 1] = float(row.coord_y)
            tensor[slice_to_i[row.slice_id], bot_to_i[row.bot_id], 2] = float(row.angle)
        return tensor

    def _augment_synthetic_summary(self, summary: dict, states: pd.DataFrame) -> dict:
        """Add final/dominant macrostate fields for presentation and filtering."""
        if states.empty:
            summary.update({
                "final_cluster": None,
                "final_state_type": None,
                "dominant_cluster": None,
                "dominant_state_type": None,
            })
            return summary

        last = states.sort_values("step").iloc[-1]
        summary["final_cluster"] = int(last["cluster"]) if pd.notna(last.get("cluster")) else None
        summary["final_state_type"] = last.get("state_type", None)

        if "state_type" in states.columns:
            dominant_state = states["state_type"].mode(dropna=True)
            summary["dominant_state_type"] = dominant_state.iloc[0] if not dominant_state.empty else None
        else:
            summary["dominant_state_type"] = None
        if "cluster" in states.columns:
            dominant_cluster = states["cluster"].mode(dropna=True)
            summary["dominant_cluster"] = int(dominant_cluster.iloc[0]) if not dominant_cluster.empty else None
        else:
            summary["dominant_cluster"] = None
        return summary

    def run_synthetic_study(
        self,
        specs: list[InitialConditionSpec],
        total_steps: int | None = None,
        save_trajectories: bool = True,
        save_parquet: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.predictor is None:
            self.load_predictor()
        if self.macro_classifier is None:
            self.load_macrostate_classifier()

        total_steps = total_steps or self.config.synthetic_total_steps
        generator = InitialStateGenerator(robot_size=self.config.robot_size, dish_diameter=self.config.dish_diameter)
        all_states = []
        summaries = []

        # for spec in specs:
        #     print(f"Synthetic rollout: {spec.experiment_id}")

        for spec in tqdm(specs, desc="Synthetic experiments", total=len(specs)):
            tqdm.write(f"Synthetic rollout: {spec.experiment_id}")

            history = generator.generate(spec)
            trajectory = self.predictor.rollout_from_history(history, total_steps=total_steps)
            states = self.classify_trajectory(trajectory)
            states["experiment_id"] = spec.experiment_id
            for key, value in asdict(spec).items():
                states[key] = value
            all_states.append(states)

            summary = summarize_micelle(states, dt=self.config.dt)
            summary = self._augment_synthetic_summary(summary, states)
            summary.update(asdict(spec))
            summaries.append(summary)

            if save_trajectories or save_parquet:
                self.save_synthetic_trajectory(
                    trajectory,
                    spec.experiment_id,
                    save_pt=save_trajectories,
                    save_parquet=save_parquet,
                )

            plot_cluster_timeline(
                states,
                self.output_dir / "plots" / f"timeline_{safe_name(spec.experiment_id)}.png",
                title=f"Macrostate timeline: {spec.experiment_id}",
            )

        states_df = pd.concat(all_states, ignore_index=True) if all_states else pd.DataFrame()
        summary_df = pd.DataFrame(summaries)

        tables_dir = self.output_dir / "tables"
        states_df.to_csv(tables_dir / "synthetic_step_states.csv", index=False)
        summary_df.to_csv(tables_dir / "synthetic_summary.csv", index=False)
        plot_micelle_summary(summary_df, self.output_dir / "plots" / "micelle_summary.png")
        plot_synthetic_research_overview(summary_df, self.output_dir / "plots" / "synthetic_overview")
        return states_df, summary_df

    def find_synthetic_experiment(
        self,
        kind: str = "first_micelle",
        summary_df: pd.DataFrame | None = None,
        state_name: str | None = None,
    ) -> str:
        """Select synthetic experiment by outcome.

        kind options:
        - first_micelle: earliest first_micelle_step;
        - longest_micelle: maximum micelle lifetime/total duration;
        - no_micelle: first experiment where micelle_formed=False;
        - target_state: experiment whose final/dominant state contains state_name.
        """
        if summary_df is None:
            path = self.output_dir / "tables" / "synthetic_summary.csv"
            if not path.exists():
                raise FileNotFoundError(f"Synthetic summary not found: {path}. Run run_synthetic_study(...) first.")
            summary_df = pd.read_csv(path)
        if summary_df.empty:
            raise ValueError("Synthetic summary is empty.")

        kind = kind.lower()
        df = summary_df.copy()
        if kind == "first_micelle":
            formed = df[df["micelle_formed"].fillna(False).astype(bool)]
            if formed.empty:
                raise ValueError("No synthetic experiment formed a micelle.")
            row = formed.sort_values("first_micelle_step", ascending=True).iloc[0]
        elif kind == "longest_micelle":
            formed = df[df["micelle_formed"].fillna(False).astype(bool)]
            if formed.empty:
                raise ValueError("No synthetic experiment formed a micelle.")
            metric = "micelle_max_lifetime_steps" if "micelle_max_lifetime_steps" in formed.columns else "micelle_total_steps"
            row = formed.sort_values(metric, ascending=False).iloc[0]
        elif kind == "no_micelle":
            no = df[~df["micelle_formed"].fillna(False).astype(bool)]
            if no.empty:
                raise ValueError("All synthetic experiments formed a micelle.")
            row = no.iloc[0]
        elif kind == "target_state":
            if not state_name:
                raise ValueError("state_name is required when kind='target_state'.")
            cols = [c for c in ["final_state_type", "dominant_state_type"] if c in df.columns]
            if not cols:
                raise ValueError("Synthetic summary has no final_state_type/dominant_state_type columns.")
            mask = False
            for col in cols:
                mask = mask | df[col].astype(str).str.contains(state_name, case=False, regex=False, na=False)
            matches = df[mask]
            if matches.empty:
                raise ValueError(f"No synthetic experiment matched state_name={state_name!r}.")
            row = matches.iloc[0]
        else:
            raise ValueError("Unknown kind. Use first_micelle, longest_micelle, no_micelle, or target_state.")

        experiment_id = str(row["experiment_id"])
        print(f"Selected synthetic experiment: {experiment_id} ({kind})")
        return experiment_id

    def save_synthetic_video(
        self,
        experiment_id: str,
        states_df: pd.DataFrame | None = None,
        output_path: str | Path | None = None,
    ) -> Path | None:
        """Save video for one synthetic rollout with macrostate/cluster in frame title."""
        if not getattr(self.config, "save_animations", True):
            print("Synthetic video skipped because config.save_animations=False.")
            return None

        trajectory = self.load_synthetic_trajectory(experiment_id)
        traj_df = dataframe_from_tensor(trajectory, file_name=experiment_id, start_step=0)

        if states_df is None:
            states_path = self.output_dir / "tables" / "synthetic_step_states.csv"
            if states_path.exists():
                all_states = pd.read_csv(states_path)
                states_df = all_states[all_states["experiment_id"] == experiment_id].copy()
            else:
                states_df = self.classify_trajectory(trajectory)
                states_df["experiment_id"] = experiment_id
        else:
            states_df = states_df[states_df["experiment_id"] == experiment_id].copy() if "experiment_id" in states_df.columns else states_df.copy()

        videos_dir = self.output_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_path or (videos_dir / f"synthetic_{safe_name(experiment_id)}.mp4"))

        return animate_synthetic_trajectory_from_df(
            traj_df,
            states_df=states_df,
            save_path=output_path,
            title=f"Synthetic rollout\n{experiment_id}",
            fps=getattr(self.config, "animation_fps", 10),
            max_frames=getattr(self.config, "animation_max_frames", None),
            robot_width=getattr(self.config, "animation_robot_width", self.config.robot_size),
            robot_height=getattr(self.config, "animation_robot_height", self.config.robot_size * 0.6),
            normalize=getattr(self.config, "animation_use_normalized_coords", False),
            windows_safe=getattr(self.config, "animation_windows_safe", True),
        )

    def create_synthetic_presentation_plots(self, summary_df: pd.DataFrame | None = None) -> list[Path]:
        """Create overview plots intended for reports/presentations."""
        if summary_df is None:
            path = self.output_dir / "tables" / "synthetic_summary.csv"
            if not path.exists():
                raise FileNotFoundError(f"Synthetic summary not found: {path}. Run run_synthetic_study(...) first.")
            summary_df = pd.read_csv(path)
        return plot_synthetic_research_overview(summary_df, self.output_dir / "plots" / "synthetic_overview")


    def run_full_research(
        self,
        specs: list[InitialConditionSpec],
        experiments_dict=None,
        train_one_step: bool | None = None,
        force_retrain: bool = False,
        evaluate_real_horizon: int | None = 200,
        synthetic_total_steps: int | None = None,
        save_trajectories: bool = False,
    ) -> dict:
        """Full autonomous workflow:

        1. load fact data;
        2. train one-step GNN or reuse existing checkpoint;
        3. fit Macrostate KMeans on fact data;
        4. evaluate multistep rollout on fact data;
        5. run synthetic 600-step study and classify each step.
        """
        if experiments_dict is None:
            experiments_dict = self.load_fact_data()

        should_train = self.config.train_one_step if train_one_step is None else train_one_step
        if should_train:
            one_step_metrics = self.train_one_step_model(experiments_dict, force=force_retrain)
        else:
            one_step_metrics = self.ensure_one_step_model(experiments_dict, force=False)

        macro_classifier = self.fit_macrostate_classifier(experiments_dict)
        self.load_predictor(train_if_missing=False)

        multistep_predictions, multistep_metrics = self.evaluate_multistep_on_fact_data(
            experiments_dict, horizon=evaluate_real_horizon
        )
        if getattr(self.config, "save_animations", True):
            # These calls are after training/evaluation and reuse saved/in-memory prediction tables.
            try:
                self.save_one_step_video(experiments_dict=experiments_dict)
            except Exception as exc:
                print(f"One-step video was not created: {exc}")
            try:
                self.save_multistep_video(predictions_df=multistep_predictions, experiments_dict=experiments_dict)
            except Exception as exc:
                print(f"Multistep video was not created: {exc}")

        synthetic_states, synthetic_summary = self.run_synthetic_study(
            specs,
            total_steps=synthetic_total_steps or self.config.synthetic_total_steps,
            save_trajectories=save_trajectories,
        )

        return {
            "one_step_metrics": one_step_metrics,
            "macrostate_interpretation": macro_classifier.interpretation_df,
            "multistep_metrics": multistep_metrics,
            "multistep_predictions": multistep_predictions,
            "synthetic_states": synthetic_states,
            "synthetic_summary": synthetic_summary,
        }


def summarize_micelle(states_df: pd.DataFrame, dt: float = 0.1) -> dict:
    micelle = states_df["is_micelle"].fillna(False).to_numpy(dtype=bool)
    steps = states_df["step"].to_numpy()
    if len(micelle) == 0 or not micelle.any():
        return {
            "micelle_formed": False,
            "first_micelle_step": None,
            "micelle_total_steps": 0,
            "micelle_total_time": 0.0,
            "micelle_max_lifetime_steps": 0,
            "micelle_max_lifetime_time": 0.0,
            "micelle_segments": 0,
        }

    first_step = int(steps[micelle][0])
    total_steps = int(micelle.sum())
    max_run = 0
    segments = 0
    cur = 0
    for flag in micelle:
        if flag:
            if cur == 0:
                segments += 1
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0

    return {
        "micelle_formed": True,
        "first_micelle_step": first_step,
        "micelle_total_steps": total_steps,
        "micelle_total_time": total_steps * dt,
        "micelle_max_lifetime_steps": int(max_run),
        "micelle_max_lifetime_time": max_run * dt,
        "micelle_segments": int(segments),
    }


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))[:120]
