from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse, Patch
import numpy as np
import pandas as pd
import torch


def plot_real_vs_predicted_trajectories(
    real_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    save_path: str,
    title: str = "Real vs predicted multistep trajectories",
    max_robots: int = 50,
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    real = real_tensor.detach().cpu().numpy()
    pred = pred_tensor.detach().cpu().numpy()
    n = min(real.shape[1], pred.shape[1], max_robots)

    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.plot(real[:, i, 0], real[:, i, 1], linewidth=1, alpha=0.7)
        plt.plot(pred[:, i, 0], pred[:, i, 1], linestyle="--", linewidth=1, alpha=0.7)
    plt.scatter(real[0, :n, 0], real[0, :n, 1], marker="o", s=15, label="start real")
    plt.scatter(pred[-1, :n, 0], pred[-1, :n, 1], marker="x", s=15, label="end predicted")
    plt.xlabel("coord_x")
    plt.ylabel("coord_y")
    plt.title(title)
    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_multistep_error_by_horizon(predictions_df: pd.DataFrame, save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if predictions_df.empty:
        return
    df = predictions_df.copy()
    df["pos_error"] = np.sqrt((df.coord_x_pred - df.coord_x_real) ** 2 + (df.coord_y_pred - df.coord_y_real) ** 2)
    grouped = df.groupby("forecast_horizon")["pos_error"].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(grouped.index, grouped.values)
    plt.xlabel("Forecast horizon, steps")
    plt.ylabel("Mean position error")
    plt.title("Autoregressive multistep error accumulation")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_cluster_timeline(states_df: pd.DataFrame, save_path: str, title: str = "Macrostate timeline") -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if states_df.empty:
        return
    plt.figure(figsize=(12, 4))
    plt.scatter(states_df["step"], states_df["cluster"], s=8, c=states_df["cluster"], cmap="tab10")
    plt.xlabel("Step")
    plt.ylabel("Cluster")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_micelle_summary(summary_df: pd.DataFrame, save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    summary_df.groupby("velocity_mode")["micelle_formed"].mean().plot(kind="bar", ax=axes[0])
    axes[0].set_ylabel("Micelle probability")
    axes[0].set_title("Micelle formation probability")
    axes[0].grid(axis="y", alpha=0.3)

    formed = summary_df[summary_df["micelle_formed"]]
    if not formed.empty:
        formed.groupby("velocity_mode")["first_micelle_step"].mean().plot(kind="bar", ax=axes[1])
        axes[1].set_ylabel("First micelle step")
    axes[1].set_title("Mean first micelle step")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def _angle_for_ellipse(angle_values: np.ndarray) -> np.ndarray:
    """Ellipse expects degrees. If values look like radians, convert them."""
    values = np.asarray(angle_values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    if np.nanmax(np.abs(finite)) <= 2 * np.pi + 0.1:
        return np.degrees(values)
    return values


def _prepare_animation_df(
    df: pd.DataFrame,
    normalize: bool = False,
    real_x_col: str = "coord_x_real",
    real_y_col: str = "coord_y_real",
    real_angle_col: str = "angle_real",
    pred_x_col: str = "coord_x_pred",
    pred_y_col: str = "coord_y_pred",
    pred_angle_col: str = "angle_pred",
) -> pd.DataFrame:
    required = {real_x_col, real_y_col, real_angle_col, pred_x_col, pred_y_col, pred_angle_col, "slice_id", "bot_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Cannot animate predictions; missing columns: {missing}")

    out = df.copy()
    out["_real_angle_plot"] = _angle_for_ellipse(out[real_angle_col].to_numpy())
    out["_pred_angle_plot"] = _angle_for_ellipse(out[pred_angle_col].to_numpy())

    if normalize:
        xs = np.concatenate([out[real_x_col].to_numpy(), out[pred_x_col].to_numpy()])
        ys = np.concatenate([out[real_y_col].to_numpy(), out[pred_y_col].to_numpy()])
        x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
        y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
        x_span = max(x_max - x_min, 1e-9)
        y_span = max(y_max - y_min, 1e-9)
        out["_real_x_plot"] = (out[real_x_col] - x_min) / x_span
        out["_pred_x_plot"] = (out[pred_x_col] - x_min) / x_span
        out["_real_y_plot"] = (out[real_y_col] - y_min) / y_span
        out["_pred_y_plot"] = (out[pred_y_col] - y_min) / y_span
    else:
        out["_real_x_plot"] = out[real_x_col]
        out["_pred_x_plot"] = out[pred_x_col]
        out["_real_y_plot"] = out[real_y_col]
        out["_pred_y_plot"] = out[pred_y_col]
    return out


def _save_animation_windows_safe(ani: FuncAnimation, save_path: Path, fps: int, windows_safe: bool = True) -> Path:
    """Save MP4 with ffmpeg when available; otherwise save GIF with Pillow.

    This avoids the common Windows 11 failure mode where Matplotlib cannot find
    the ffmpeg binary. The original project used writer="ffmpeg"; this function
    keeps that path when available and provides a deterministic fallback.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    writers = animation.writers.list()

    if save_path.suffix.lower() == ".mp4":
        # Original GitHub notebook behavior: Matplotlib + writer="ffmpeg".
        # With windows_safe=False this intentionally fails if ffmpeg is missing,
        # matching the original behavior exactly.
        if "ffmpeg" in writers or not windows_safe:
            ani.save(save_path, writer="ffmpeg", fps=fps)
            return save_path

    # Windows-safe fallback: GIF via Pillow.
    gif_path = save_path.with_suffix(".gif")
    ani.save(gif_path, writer=PillowWriter(fps=fps))
    print(f"ffmpeg writer is unavailable; saved GIF instead: {gif_path}")
    return gif_path


def animate_real_vs_predicted_from_df(
    df: pd.DataFrame,
    save_path: str | Path,
    title: str = "Robot Movement: Real vs Predicted",
    fps: int = 10,
    max_frames: int | None = None,
    robot_width: float = 60.0,
    robot_height: float = 36.0,
    normalize: bool = False,
    real_color: str = "blue",
    pred_color: str = "red",
    windows_safe: bool = True,
) -> Path | None:
    """Create real-vs-predicted robot animation.

    The implementation follows the original StepPrediction notebook style:
    Matplotlib FuncAnimation + Ellipse patches, real robots in blue and predicted
    robots in red. It is used for both one-step and multistep prediction tables.
    """
    if df is None or df.empty:
        print("Animation skipped: empty dataframe.")
        return None

    plot_df = _prepare_animation_df(df, normalize=normalize)
    slice_ids = sorted(plot_df["slice_id"].unique())
    if max_frames is not None and len(slice_ids) > max_frames:
        idx = np.linspace(0, len(slice_ids) - 1, max_frames).round().astype(int)
        slice_ids = [slice_ids[i] for i in idx]

    unique_bot_ids = sorted(plot_df["bot_id"].unique())
    if normalize:
        x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0
        width, height = 0.055, 0.032
    else:
        xs = np.concatenate([plot_df["_real_x_plot"].to_numpy(), plot_df["_pred_x_plot"].to_numpy()])
        ys = np.concatenate([plot_df["_real_y_plot"].to_numpy(), plot_df["_pred_y_plot"].to_numpy()])
        x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
        y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
        margin = max(x_max - x_min, y_max - y_min, robot_width, robot_height) * 0.08
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        width, height = robot_width, robot_height

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X coordinate" + (" (normalized)" if normalize else ""))
    ax.set_ylabel("Y coordinate" + (" (normalized)" if normalize else ""))

    robots = []
    for bot_idx in range(len(unique_bot_ids) * 2):
        patch = Ellipse((0, 0), width, height, alpha=0.7)
        patch.set_color(real_color if bot_idx < len(unique_bot_ids) else pred_color)
        ax.add_patch(patch)
        robots.append(patch)

    legend_elements = [
        Patch(facecolor=real_color, alpha=0.7, label="Real Robots"),
        Patch(facecolor=pred_color, alpha=0.7, label="Predicted Robots"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    def update(frame_idx: int):
        current_slice = slice_ids[frame_idx]
        current_data = plot_df[plot_df["slice_id"] == current_slice]
        by_bot = {bot_id: group.iloc[0] for bot_id, group in current_data.groupby("bot_id")}

        for i, bot_id in enumerate(unique_bot_ids):
            row = by_bot.get(bot_id)
            if row is None:
                continue
            robots[i].set_center((row["_real_x_plot"], row["_real_y_plot"]))
            robots[i].set_angle(float(row["_real_angle_plot"]))

            pred_patch = robots[i + len(unique_bot_ids)]
            pred_patch.set_center((row["_pred_x_plot"], row["_pred_y_plot"]))
            pred_patch.set_angle(float(row["_pred_angle_plot"]))

        ax.set_title(f"{title}\nSlice ID: {current_slice}")
        return robots

    ani = FuncAnimation(fig, update, frames=len(slice_ids), interval=1000 / max(fps, 1), blit=True)
    saved_path = _save_animation_windows_safe(ani, Path(save_path), fps=fps, windows_safe=windows_safe)
    plt.close(fig)
    print(f"Animation created for {len(slice_ids)} frames and {len(unique_bot_ids)} robots: {saved_path}")
    return saved_path


def animate_synthetic_trajectory_from_df(
    df: pd.DataFrame,
    save_path: str | Path,
    states_df: pd.DataFrame | None = None,
    title: str = "Synthetic robot rollout",
    fps: int = 10,
    max_frames: int | None = None,
    robot_width: float = 60.0,
    robot_height: float = 36.0,
    normalize: bool = False,
    windows_safe: bool = True,
) -> Path | None:
    """Animate one synthetic trajectory with macrostate labels in frame title.

    Expected df columns: file_name, slice_id, bot_id, coord_x, coord_y, angle.
    Optional states_df columns: step, cluster, state_type, is_micelle.
    """
    if df is None or df.empty:
        print("Synthetic animation skipped: empty dataframe.")
        return None

    required = {"slice_id", "bot_id", "coord_x", "coord_y", "angle"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Cannot animate synthetic trajectory; missing columns: {missing}")

    plot_df = df.copy()
    plot_df["_angle_plot"] = _angle_for_ellipse(plot_df["angle"].to_numpy())

    if normalize:
        x_min, x_max = float(plot_df["coord_x"].min()), float(plot_df["coord_x"].max())
        y_min, y_max = float(plot_df["coord_y"].min()), float(plot_df["coord_y"].max())
        x_span = max(x_max - x_min, 1e-9)
        y_span = max(y_max - y_min, 1e-9)
        plot_df["_x_plot"] = (plot_df["coord_x"] - x_min) / x_span
        plot_df["_y_plot"] = (plot_df["coord_y"] - y_min) / y_span
        x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0
        width, height = 0.055, 0.032
    else:
        plot_df["_x_plot"] = plot_df["coord_x"]
        plot_df["_y_plot"] = plot_df["coord_y"]
        x_min, x_max = float(plot_df["_x_plot"].min()), float(plot_df["_x_plot"].max())
        y_min, y_max = float(plot_df["_y_plot"].min()), float(plot_df["_y_plot"].max())
        margin = max(x_max - x_min, y_max - y_min, robot_width, robot_height) * 0.08
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        width, height = robot_width, robot_height

    slice_ids = sorted(plot_df["slice_id"].unique())
    if max_frames is not None and len(slice_ids) > max_frames:
        idx = np.linspace(0, len(slice_ids) - 1, max_frames).round().astype(int)
        slice_ids = [slice_ids[i] for i in idx]

    state_lookup: dict[int, dict] = {}
    if states_df is not None and not states_df.empty:
        step_col = "step" if "step" in states_df.columns else "slice_id"
        for _, row in states_df.iterrows():
            state_lookup[int(row[step_col])] = row.to_dict()

    unique_bot_ids = sorted(plot_df["bot_id"].unique())
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X coordinate" + (" (normalized)" if normalize else ""))
    ax.set_ylabel("Y coordinate" + (" (normalized)" if normalize else ""))

    robots = []
    for _ in unique_bot_ids:
        patch = Ellipse((0, 0), width, height, alpha=0.75)
        ax.add_patch(patch)
        robots.append(patch)

    def update(frame_idx: int):
        current_slice = int(slice_ids[frame_idx])
        current_data = plot_df[plot_df["slice_id"] == current_slice]
        by_bot = {bot_id: group.iloc[0] for bot_id, group in current_data.groupby("bot_id")}

        for i, bot_id in enumerate(unique_bot_ids):
            row = by_bot.get(bot_id)
            if row is None:
                continue
            robots[i].set_center((row["_x_plot"], row["_y_plot"]))
            robots[i].set_angle(float(row["_angle_plot"]))

        subtitle = f"step={current_slice}"
        state_row = state_lookup.get(current_slice)
        if state_row is not None:
            cluster = state_row.get("cluster", "?")
            state_type = state_row.get("state_type", "unknown")
            is_micelle = bool(state_row.get("is_micelle", False))
            subtitle += f", cluster={cluster}, state={state_type}, micelle={is_micelle}"
        ax.set_title(f"{title}\n{subtitle}")
        return robots

    ani = FuncAnimation(fig, update, frames=len(slice_ids), interval=1000 / max(fps, 1), blit=True)
    saved_path = _save_animation_windows_safe(ani, Path(save_path), fps=fps, windows_safe=windows_safe)
    plt.close(fig)
    print(f"Synthetic animation created for {len(slice_ids)} frames and {len(unique_bot_ids)} robots: {saved_path}")
    return saved_path



def _find_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _plot_micelle_onset_lifetime_by_group(
    summary_df: pd.DataFrame,
    group_col: str,
    save_path: str | Path,
    title: str,
    group_label: str,
    total_steps: int | None = None,
) -> Path | None:
    """Plot mean micelle onset and mean continuous lifetime for a grouping variable.

    The plot is timeline-like:
    - marker position = mean first micelle step, conditional on micelle_formed=True;
    - thick segment length = mean max continuous micelle lifetime;
    - thin transparent segment and error bars show ±1 std spread;
    - right-side text shows formation probability over all experiments in the group.

    Experiments without micelle do not have a meaningful first_micelle_step, so they are
    not used for onset/lifetime averages. They are still represented through p(form). This
    avoids artificially treating "no micelle" as step 0 or step N.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if summary_df is None or summary_df.empty:
        return None

    df = summary_df.copy()

    # Support both old and new column names.
    if group_col not in df.columns:
        if group_col == "radius" and "initial_radius" in df.columns:
            df["radius"] = df["initial_radius"]
        elif group_col == "initial_radius" and "radius" in df.columns:
            df["initial_radius"] = df["radius"]
        else:
            return None

    micelle_col = _find_existing_column(df, ["micelle_formed", "is_micelle_formed"])
    onset_col = _find_existing_column(df, ["first_micelle_step", "first_micelle"])
    lifetime_col = _find_existing_column(
        df,
        ["micelle_max_lifetime_steps", "max_micelle_lifetime_steps", "micelle_total_steps"],
    )

    if micelle_col is None or onset_col is None or lifetime_col is None:
        return None

    df[micelle_col] = df[micelle_col].fillna(False).astype(bool)

    # Formation probability uses every experiment in the group.
    prob = (
        df.groupby(group_col, dropna=False)[micelle_col]
        .agg(formation_probability="mean", n_total="count")
        .reset_index()
    )

    # Onset/lifetime are defined only when micelle exists.
    formed = df[df[micelle_col]].copy()
    formed = formed.dropna(subset=[onset_col, lifetime_col])

    if formed.empty:
        return None

    formed[onset_col] = pd.to_numeric(formed[onset_col], errors="coerce")
    formed[lifetime_col] = pd.to_numeric(formed[lifetime_col], errors="coerce")
    formed = formed.dropna(subset=[onset_col, lifetime_col])

    agg = (
        formed.groupby(group_col, dropna=False)
        .agg(
            mean_onset=(onset_col, "mean"),
            std_onset=(onset_col, "std"),
            mean_lifetime=(lifetime_col, "mean"),
            std_lifetime=(lifetime_col, "std"),
            n_formed=(micelle_col, "count"),
        )
        .reset_index()
    )

    plot_df = prob.merge(agg, on=group_col, how="left")
    plot_df["std_onset"] = plot_df["std_onset"].fillna(0.0)
    plot_df["std_lifetime"] = plot_df["std_lifetime"].fillna(0.0)
    plot_df["n_formed"] = plot_df["n_formed"].fillna(0).astype(int)

    # Sort numeric groups numerically, categorical groups by mean onset.
    if pd.api.types.is_numeric_dtype(plot_df[group_col]):
        plot_df = plot_df.sort_values(group_col)
    else:
        plot_df = plot_df.sort_values("mean_onset", na_position="last")

    # Labels: keep concise and stable for presentation.
    def _label(value):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    labels = [_label(v) for v in plot_df[group_col].tolist()]
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(13, max(4.8, 0.72 * len(plot_df))))

    x_max_values = []
    for yi, (_, row) in zip(y, plot_df.iterrows()):
        if pd.isna(row.get("mean_onset")):
            # Group exists but no micelle was ever formed.
            ax.scatter(0, yi, marker="x", s=80)
            ax.text(
                5,
                yi,
                f"p={row['formation_probability']:.2f}, n=0/{int(row['n_total'])}",
                va="center",
                fontsize=10,
            )
            continue

        onset = float(row["mean_onset"])
        lifetime = float(row["mean_lifetime"])
        end = onset + lifetime
        std_onset = float(row["std_onset"])
        std_lifetime = float(row["std_lifetime"])
        spread_start = max(0.0, onset - std_onset)
        spread_end = max(spread_start, end + std_lifetime)
        x_max_values.append(spread_end)

        # ± spread band.
        ax.hlines(yi, spread_start, spread_end, linewidth=14, alpha=0.18)
        # Mean lifetime interval.
        ax.hlines(yi, onset, end, linewidth=7, alpha=0.85)
        # Mean onset marker + onset std.
        ax.scatter(onset, yi, s=80, zorder=3)
        ax.errorbar(onset, yi, xerr=std_onset, fmt="none", capsize=4, alpha=0.85)

        ax.text(
            end + 5,
            yi,
            f"p={row['formation_probability']:.2f}, n={int(row['n_formed'])}/{int(row['n_total'])}",
            va="center",
            fontsize=10,
        )

    if total_steps is None:
        if "total_steps" in df.columns:
            total_steps = int(pd.to_numeric(df["total_steps"], errors="coerce").max())
        else:
            total_steps = int(max([600] + [v for v in x_max_values if np.isfinite(v)]))

    ax.set_xlim(0, max(total_steps, max(x_max_values) if x_max_values else total_steps) * 1.08)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Step")
    ax.set_ylabel(group_label)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    ax.text(
        0.0,
        -0.14,
        "Marker = mean first micelle step; thick line = mean max continuous micelle lifetime; "
        "transparent band/error bars = ±1 std; p = formation probability over all experiments.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    return save_path


def plot_synthetic_research_overview(summary_df: pd.DataFrame, save_dir: str | Path) -> list[Path]:
    """Create presentation-oriented plots for synthetic initial-condition study.

    Returns created file paths. The plots summarize how radius/speed/velocity_mode
    affect micelle formation probability, first micelle time, and lifetime.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if summary_df is None or summary_df.empty:
        return []

    df = summary_df.copy()
    paths: list[Path] = []
    if "speed_mean" in df.columns:
        df["speed_group"] = df["speed_mean"].round(2).astype(str)
    elif "speed" in df.columns:
        df["speed_group"] = df["speed"].round(2).astype(str)
    else:
        df["speed_group"] = "unknown"

    # 1) Formation probability by velocity mode and radius.
    if {"velocity_mode", "radius", "micelle_formed"}.issubset(df.columns):
        pivot = df.pivot_table(index="velocity_mode", columns="radius", values="micelle_formed", aggfunc="mean")
        path = save_dir / "synthetic_micelle_probability_by_mode_radius.png"
        plt.figure(figsize=(10, 5))
        im = plt.imshow(pivot.values, aspect="auto", vmin=0, vmax=1)
        plt.colorbar(im, label="Micelle formation probability")
        plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
        plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
        plt.xlabel("Initial radius")
        plt.ylabel("Velocity mode")
        plt.title("Micelle probability by initial density and velocity mode")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    plt.text(j, i, f"{val:.2f}", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
        paths.append(path)

    # 2) First micelle step by mode; shows speed of formation, not only fact of formation.
    if {"velocity_mode", "first_micelle_step", "micelle_formed"}.issubset(df.columns):
        formed = df[df["micelle_formed"].fillna(False)].copy()
        if not formed.empty:
            grouped = formed.groupby("velocity_mode")["first_micelle_step"].mean().sort_values()
            path = save_dir / "synthetic_mean_first_micelle_step_by_mode.png"
            plt.figure(figsize=(9, 5))
            grouped.plot(kind="bar")
            plt.ylabel("Mean first micelle step")
            plt.xlabel("Velocity mode")
            plt.title("How fast micelle forms by initial velocity mode")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(path, dpi=180, bbox_inches="tight")
            plt.close()
            paths.append(path)

    # 3) Lifetime by radius/speed scatter-like aggregate; useful for comparing initial conditions.
    lifetime_col = "micelle_max_lifetime_steps" if "micelle_max_lifetime_steps" in df.columns else "micelle_total_steps"
    if {"radius", "speed_mean", lifetime_col}.issubset(df.columns):
        path = save_dir / "synthetic_micelle_lifetime_radius_speed.png"
        plt.figure(figsize=(9, 6))
        sc = plt.scatter(df["radius"], df["speed_mean"], c=df[lifetime_col], s=80, alpha=0.85)
        plt.colorbar(sc, label=lifetime_col)
        plt.xlabel("Initial radius")
        plt.ylabel("Initial speed_mean")
        plt.title("Micelle persistence across synthetic initial conditions")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
        paths.append(path)

    # 4) State endpoint composition if available.
    if "final_state_type" in df.columns and "velocity_mode" in df.columns:
        table = pd.crosstab(df["velocity_mode"], df["final_state_type"], normalize="index")
        path = save_dir / "synthetic_final_state_distribution_by_mode.png"
        ax = table.plot(kind="bar", stacked=True, figsize=(12, 5))
        ax.set_ylabel("Share of experiments")
        ax.set_xlabel("Velocity mode")
        ax.set_title("Final macrostate distribution by initial velocity mode")
        ax.grid(axis="y", alpha=0.3)
        plt.legend(title="Final state", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
        paths.append(path)

    # 5) Timeline-like onset/lifetime plot by initial velocity mode.
    # This directly shows when micelle appears and how long it survives on average.
    path = _plot_micelle_onset_lifetime_by_group(
        df,
        group_col="velocity_mode",
        save_path=save_dir / "synthetic_micelle_onset_lifetime_by_velocity_mode.png",
        title="Micelle onset and lifetime by initial velocity mode",
        group_label="Velocity mode",
    )
    if path is not None:
        paths.append(path)

    # 6) Same idea by initial speed_mean.
    if "speed_mean" in df.columns:
        path = _plot_micelle_onset_lifetime_by_group(
            df,
            group_col="speed_mean",
            save_path=save_dir / "synthetic_micelle_onset_lifetime_by_speed.png",
            title="Micelle onset and lifetime by initial speed",
            group_label="Initial speed_mean",
        )
        if path is not None:
            paths.append(path)

    # 7) Same idea by initial radius of occupied robot area.
    radius_col = "radius" if "radius" in df.columns else "initial_radius" if "initial_radius" in df.columns else None
    if radius_col is not None:
        path = _plot_micelle_onset_lifetime_by_group(
            df,
            group_col=radius_col,
            save_path=save_dir / "synthetic_micelle_onset_lifetime_by_radius.png",
            title="Micelle onset and lifetime by initial radius",
            group_label="Initial radius",
        )
        if path is not None:
            paths.append(path)

    return paths

