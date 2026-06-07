"""CLI entry point for the autonomous research package.

Example:
    python run_research.py --fit-macrostate --evaluate-real --run-synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import ResearchConfig
from experiment_runner import ResearchPipeline
from initial_state_generator import InitialConditionSpec, InitialStateGenerator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/processed_robots_data.parquet")
    p.add_argument("--model-path", default="models/gnn_model_weights.pth")
    p.add_argument("--output-dir", default="results/research")
    p.add_argument("--full", action="store_true", help="Run train one-step -> fit Macrostate -> evaluate multistep -> synthetic study.")
    p.add_argument("--train-onestep", action="store_true", help="Train the one-step GNN inside this package.")
    p.add_argument("--force-retrain", action="store_true", help="Retrain even if model_path already exists.")
    p.add_argument("--fit-macrostate", action="store_true")
    p.add_argument("--evaluate-real", action="store_true")
    p.add_argument("--run-synthetic", action="store_true")
    p.add_argument("--synthetic-total-steps", type=int, default=600)
    p.add_argument("--train-sample-ratio", type=float, default=1.0, help="Fraction of fact experiment files used for one-step training before train/val/test split.")
    p.add_argument("--synthetic-repeats", type=int, default=2)
    p.add_argument("--speed-min", type=float, default=10.0, help="Lower bound for continuous random speed_mean sampling.")
    p.add_argument("--speed-max", type=float, default=35.0, help="Upper bound for continuous random speed_mean sampling.")
    p.add_argument("--n-speed-samples", type=int, default=3, help="How many continuous speed_mean values to sample for the synthetic sweep.")
    p.add_argument("--dish-diameter", type=float, default=1000.0)
    p.add_argument("--robot-size", type=float, default=60.0, help="Robot diameter. Synthetic centers are generated at least this far apart by default.")
    p.add_argument("--n-robots", type=int, default=50)
    p.add_argument("--allow-untrained", action="store_true", help="For smoke tests only; do not use for research results.")
    p.add_argument("--evaluate-real-horizon", type=int, default=200)
    p.add_argument("--no-animations", action="store_true", help="Disable one-step and multistep animation export.")
    p.add_argument("--animation-max-frames", type=int, default=300, help="Maximum frames per exported animation; use -1 for all frames.")
    p.add_argument("--demo-file-name", default=None, help="Concrete experiment file_name for plots/videos. Overrides --demo-file-index.")
    p.add_argument("--demo-file-index", type=int, default=2, help="Index inside demo split, default test_files[2] as in the original notebook.")
    p.add_argument("--demo-split", default="test", choices=["test", "train", "val", "all"], help="Split used with --demo-file-index.")
    p.add_argument("--save-onestep-video", action="store_true", help="Create one-step video from saved predictions without retraining.")
    p.add_argument("--save-multistep-video", action="store_true", help="Create multistep video from saved predictions without rerunning rollout.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ResearchConfig(
        data_path=args.data_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        synthetic_total_steps=args.synthetic_total_steps,
        n_robots=args.n_robots,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        n_speed_samples=args.n_speed_samples,
        dish_diameter=args.dish_diameter,
        robot_size=args.robot_size,
        train_sample_ratio=args.train_sample_ratio,
        save_animations=not args.no_animations,
        animation_max_frames=None if args.animation_max_frames == -1 else args.animation_max_frames,
        demo_file_name=args.demo_file_name,
        demo_file_index=args.demo_file_index,
        demo_split=args.demo_split,
    )
    pipe = ResearchPipeline(cfg)

    base = InitialConditionSpec(
        experiment_id="base",
        n_robots=cfg.n_robots,
        dt=cfg.dt,
        history_len=cfg.start_step + 1,
        dish_diameter=cfg.dish_diameter,
        robot_diameter=cfg.robot_size,
        min_center_distance=cfg.min_center_distance,
        seed=cfg.random_state,
    )
    specs = InitialStateGenerator.make_sweep_specs(
        base,
        radius_values=[250, 350, 470],
        speed_values=None,
        speed_range=(cfg.speed_min, cfg.speed_max),
        n_speed_samples=cfg.n_speed_samples,
        speed_seed=cfg.random_state,
        velocity_modes=["random", "aligned", "rotational", "inward"],
        repeats=args.synthetic_repeats,
    )
    print("Sampled continuous speed_mean values:", sorted({round(s.speed_mean, 3) for s in specs}))

    if args.full:
        result = pipe.run_full_research(
            specs,
            train_one_step=True,
            force_retrain=args.force_retrain,
            evaluate_real_horizon=args.evaluate_real_horizon,
            synthetic_total_steps=args.synthetic_total_steps,
        )
        print("One-step metrics:", result["one_step_metrics"])
        print("Multistep metrics:", result["multistep_metrics"])
        print(result["synthetic_summary"].head())
        print(f"Saved results to {cfg.output_dir}")
        return

    experiments = None
    if args.train_onestep or args.fit_macrostate or args.evaluate_real:
        experiments = pipe.load_fact_data()

    if args.train_onestep:
        metrics = pipe.train_one_step_model(experiments, force=args.force_retrain)
        print("One-step metrics:", metrics)

    if args.fit_macrostate:
        pipe.fit_macrostate_classifier(experiments)
    else:
        path = Path(cfg.output_dir) / "models" / "macrostate_kmeans.pkl"
        if path.exists():
            pipe.load_macrostate_classifier(str(path))

    if args.evaluate_real:
        pipe.load_predictor(allow_untrained=args.allow_untrained)
        _, metrics = pipe.evaluate_multistep_on_fact_data(experiments, horizon=args.evaluate_real_horizon)
        print("Multistep metrics:", metrics)

    if args.save_onestep_video:
        saved = pipe.save_one_step_video(file_name=args.demo_file_name, file_index=args.demo_file_index, split=args.demo_split)
        print("One-step video:", saved)

    if args.save_multistep_video:
        saved = pipe.save_multistep_video(file_name=args.demo_file_name, file_index=args.demo_file_index, split=args.demo_split)
        print("Multistep video:", saved)

    if args.run_synthetic:
        if pipe.macro_classifier is None:
            pipe.load_macrostate_classifier()
        pipe.load_predictor(allow_untrained=args.allow_untrained)
        _, summary = pipe.run_synthetic_study(specs, total_steps=cfg.synthetic_total_steps)
        print(summary.head())
        print(f"Saved results to {cfg.output_dir}")


if __name__ == "__main__":
    main()
