from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import pandas as pd
import torch


@dataclass
class InitialConditionSpec:
    experiment_id: str
    n_robots: int = 50
    history_len: int = 4
    dt: float = 0.1

    # Position generation.
    # plate/dish: robots are generated inside a circular dish of diameter dish_diameter.
    position_mode: str = "disk"  # disk, ring, uniform, clusters
    center_x: float = 0.0
    center_y: float = 0.0
    dish_diameter: float = 1000.0
    radius: float = 400.0
    radius_std: float = 60.0
    domain_size: float = 1000.0  # kept for backward compatibility; dish_diameter is used for circular plate
    n_position_clusters: int = 3
    cluster_std: float = 80.0

    # Non-overlap constraints. Robot diameter is 60 by default, so center distance must be >= 60.
    robot_diameter: float = 60.0
    min_center_distance: float | None = None
    boundary_margin: float | None = None
    max_position_attempts: int = 30000

    # Velocity generation.
    velocity_mode: str = "random"  # random, aligned, rotational, inward, outward
    speed_mean: float = 20.0
    speed_std: float = 5.0
    heading_mean_deg: float = 0.0
    heading_noise_deg: float = 30.0
    acceleration_noise: float = 1.0
    angular_speed_mean: float = 0.0
    angular_speed_std: float = 5.0
    seed: int = 42


class InitialStateGenerator:
    """Generates synthetic initial histories of length >=4 for autoregressive GNN rollout.

    Important geometry convention:
    - Coordinates are robot centers.
    - dish_diameter=1000 means centers must stay inside radius 500 - robot_radius.
    - robot_diameter=60 means generated centers are at least 60 units apart.
    """

    def __init__(self, robot_size: float = 60.0, dish_diameter: float = 1000.0):
        self.robot_size = robot_size
        self.dish_diameter = dish_diameter

    def generate(self, spec: InitialConditionSpec) -> torch.Tensor:
        rng = np.random.default_rng(spec.seed)
        xy0 = self._generate_positions(spec, rng)
        velocity = self._generate_velocities(spec, xy0, rng)
        angles0 = np.degrees(np.arctan2(velocity[:, 1], velocity[:, 0])) % 360.0
        angular_velocity = rng.normal(spec.angular_speed_mean, spec.angular_speed_std, size=spec.n_robots)

        tensor = np.zeros((spec.history_len, spec.n_robots, 3), dtype=np.float32)
        tensor[0, :, 0:2] = xy0
        tensor[0, :, 2] = angles0

        current_xy = xy0.copy()
        current_v = velocity.copy()
        current_angle = angles0.copy()
        for t in range(1, spec.history_len):
            current_v = current_v + rng.normal(0.0, spec.acceleration_noise, size=current_v.shape) * spec.dt
            current_xy = current_xy + current_v * spec.dt
            current_angle = (current_angle + angular_velocity * spec.dt) % 360.0
            tensor[t, :, 0:2] = current_xy
            tensor[t, :, 2] = current_angle

        return torch.tensor(tensor, dtype=torch.float32)

    def _geometry_params(self, spec: InitialConditionSpec) -> tuple[np.ndarray, float, float, float]:
        center = np.array([spec.center_x, spec.center_y], dtype=float)
        robot_diameter = float(getattr(spec, "robot_diameter", self.robot_size))
        robot_radius = robot_diameter / 2.0
        boundary_margin = robot_radius if spec.boundary_margin is None else float(spec.boundary_margin)
        dish_diameter = float(getattr(spec, "dish_diameter", self.dish_diameter))
        allowed_plate_radius = dish_diameter / 2.0 - boundary_margin
        if allowed_plate_radius <= 0:
            raise ValueError("dish_diameter is too small for the requested boundary_margin/robot size.")
        min_dist = robot_diameter if spec.min_center_distance is None else float(spec.min_center_distance)
        return center, allowed_plate_radius, min_dist, robot_diameter

    @staticmethod
    def _inside_circle(xy: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
        rel = xy - center[None, :]
        return np.sum(rel * rel, axis=1) <= radius * radius

    @staticmethod
    def _hex_candidate_pool(center: np.ndarray, radius: float, spacing: float) -> np.ndarray:
        """Dense deterministic non-overlapping candidate pool inside a disk."""
        spacing = max(float(spacing), 1e-6)
        dy = spacing * np.sqrt(3.0) / 2.0
        pts: list[tuple[float, float]] = []
        row = 0
        y = -radius
        while y <= radius:
            offset = spacing / 2.0 if row % 2 else 0.0
            x = -radius + offset
            while x <= radius:
                if x * x + y * y <= radius * radius:
                    pts.append((center[0] + x, center[1] + y))
                x += spacing
            y += dy
            row += 1
        return np.asarray(pts, dtype=float)

    def _candidate_by_mode(self, spec: InitialConditionSpec, rng: np.random.Generator, n: int) -> np.ndarray:
        center, allowed_plate_radius, _, _ = self._geometry_params(spec)
        requested_radius = min(float(spec.radius), allowed_plate_radius)

        if spec.position_mode == "uniform":
            # Uniform over the whole circular plate, not the old square domain.
            theta = rng.uniform(0, 2 * np.pi, size=n)
            r = allowed_plate_radius * np.sqrt(rng.uniform(0, 1, size=n))
            return np.column_stack([np.cos(theta) * r, np.sin(theta) * r]) + center

        if spec.position_mode == "ring":
            theta = rng.uniform(0, 2 * np.pi, size=n)
            r = np.clip(rng.normal(requested_radius, spec.radius_std, size=n), 0.0, allowed_plate_radius)
            return np.column_stack([np.cos(theta) * r, np.sin(theta) * r]) + center

        if spec.position_mode == "clusters":
            # Cluster centers are sampled inside requested_radius, then points are sampled around centers.
            theta_c = rng.uniform(0, 2 * np.pi, size=spec.n_position_clusters)
            r_c = requested_radius * np.sqrt(rng.uniform(0, 1, size=spec.n_position_clusters))
            cluster_centers = np.column_stack([np.cos(theta_c) * r_c, np.sin(theta_c) * r_c]) + center
            labels = rng.integers(0, spec.n_position_clusters, size=n)
            return cluster_centers[labels] + rng.normal(0.0, spec.cluster_std, size=(n, 2))

        # disk: uniform inside requested initial radius.
        theta = rng.uniform(0, 2 * np.pi, size=n)
        r = requested_radius * np.sqrt(rng.uniform(0, 1, size=n))
        return np.column_stack([np.cos(theta) * r, np.sin(theta) * r]) + center

    def _generate_positions(self, spec: InitialConditionSpec, rng: np.random.Generator) -> np.ndarray:
        center, allowed_plate_radius, min_dist, _ = self._geometry_params(spec)
        requested_radius = min(float(spec.radius), allowed_plate_radius)

        # First try stochastic rejection sampling so the distribution follows position_mode.
        points: list[np.ndarray] = []
        attempts = 0
        batch_size = max(256, spec.n_robots * 8)
        while len(points) < spec.n_robots and attempts < spec.max_position_attempts:
            candidates = self._candidate_by_mode(spec, rng, batch_size)
            inside = self._inside_circle(candidates, center, allowed_plate_radius)
            for cand in candidates[inside]:
                attempts += 1
                if len(points) == 0:
                    points.append(cand)
                else:
                    d = np.linalg.norm(np.asarray(points) - cand[None, :], axis=1)
                    if np.all(d >= min_dist):
                        points.append(cand)
                if len(points) >= spec.n_robots or attempts >= spec.max_position_attempts:
                    break

        if len(points) >= spec.n_robots:
            return np.asarray(points[: spec.n_robots], dtype=np.float32)

        # Robust fallback: non-overlapping hexagonal candidate pool.
        # For disk/ring/clusters, use requested_radius; for uniform, use full plate radius.
        fallback_radius = allowed_plate_radius if spec.position_mode == "uniform" else requested_radius
        spacing = min_dist
        for _ in range(5):
            pool = self._hex_candidate_pool(center, fallback_radius, spacing)
            if spec.position_mode == "ring":
                rel = pool - center[None, :]
                rr = np.linalg.norm(rel, axis=1)
                lo = max(0.0, requested_radius - max(spec.radius_std, min_dist))
                hi = min(allowed_plate_radius, requested_radius + max(spec.radius_std, min_dist))
                pool = pool[(rr >= lo) & (rr <= hi)]
            if len(pool) >= spec.n_robots:
                idx = rng.choice(len(pool), size=spec.n_robots, replace=False)
                return pool[idx].astype(np.float32)
            # Expand the effective placement radius if the requested one is physically too dense.
            fallback_radius = min(allowed_plate_radius, fallback_radius + min_dist)
            if fallback_radius >= allowed_plate_radius and len(pool) < spec.n_robots:
                # If still insufficient, slightly reduce spacing only as last resort to avoid total failure.
                spacing *= 0.98

        raise ValueError(
            "Could not generate non-overlapping initial positions. "
            f"n_robots={spec.n_robots}, min_center_distance={min_dist}, "
            f"requested_radius={spec.radius}, dish_diameter={spec.dish_diameter}. "
            "Increase radius/dish_diameter, decrease n_robots, or decrease min_center_distance."
        )

    def _generate_velocities(self, spec: InitialConditionSpec, xy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        speed = np.clip(rng.normal(spec.speed_mean, spec.speed_std, size=spec.n_robots), 0.0, None)
        rel = xy - np.array([spec.center_x, spec.center_y], dtype=float)
        base_angle = np.arctan2(rel[:, 1], rel[:, 0])

        if spec.velocity_mode == "aligned":
            heading = np.deg2rad(spec.heading_mean_deg + rng.normal(0.0, spec.heading_noise_deg, size=spec.n_robots))
        elif spec.velocity_mode == "rotational":
            heading = base_angle + np.pi / 2.0 + np.deg2rad(rng.normal(0.0, spec.heading_noise_deg, size=spec.n_robots))
        elif spec.velocity_mode == "inward":
            heading = base_angle + np.pi + np.deg2rad(rng.normal(0.0, spec.heading_noise_deg, size=spec.n_robots))
        elif spec.velocity_mode == "outward":
            heading = base_angle + np.deg2rad(rng.normal(0.0, spec.heading_noise_deg, size=spec.n_robots))
        else:
            heading = rng.uniform(0, 2 * np.pi, size=spec.n_robots)

        return np.column_stack([np.cos(heading) * speed, np.sin(heading) * speed]).astype(np.float32)

    @staticmethod
    def sample_speed_values(
        speed_range: tuple[float, float] = (10.0, 35.0),
        n_speed_samples: int = 3,
        seed: int = 42,
    ) -> list[float]:
        """Sample continuous speed_mean values for a synthetic sweep.

        This replaces fixed speed_values=[10, 20, 35] with 3 random continuous values
        by default. The values are sampled once and then reused across all radius/mode/repeat
        combinations, so the experiment grid remains interpretable.
        """
        lo, hi = map(float, speed_range)
        if hi <= lo:
            raise ValueError("speed_range must satisfy max > min.")
        if n_speed_samples <= 0:
            raise ValueError("n_speed_samples must be positive.")
        rng = np.random.default_rng(seed)
        return sorted(float(x) for x in rng.uniform(lo, hi, size=n_speed_samples))

    @staticmethod
    def make_sweep_specs(
        base: InitialConditionSpec,
        radius_values: list[float],
        speed_values: list[float] | None = None,
        velocity_modes: list[str] | None = None,
        repeats: int = 3,
        speed_range: tuple[float, float] = (10.0, 35.0),
        n_speed_samples: int = 3,
        speed_seed: int | None = None,
    ) -> list[InitialConditionSpec]:
        if velocity_modes is None:
            velocity_modes = ["random", "aligned", "rotational", "inward"]
        if speed_values is None:
            speed_values = InitialStateGenerator.sample_speed_values(
                speed_range=speed_range,
                n_speed_samples=n_speed_samples,
                seed=base.seed if speed_seed is None else speed_seed,
            )

        specs: list[InitialConditionSpec] = []
        idx = 0
        for radius in radius_values:
            for speed in speed_values:
                for velocity_mode in velocity_modes:
                    for rep in range(repeats):
                        d = asdict(base)
                        d.update({
                            "experiment_id": f"synthetic_r{radius:g}_v{speed:.2f}_{velocity_mode}_rep{rep}",
                            "radius": float(radius),
                            "speed_mean": float(speed),
                            "velocity_mode": velocity_mode,
                            "seed": base.seed + idx,
                        })
                        specs.append(InitialConditionSpec(**d))
                        idx += 1
        return specs

    @staticmethod
    def specs_to_dataframe(specs: list[InitialConditionSpec]) -> pd.DataFrame:
        return pd.DataFrame([asdict(s) for s in specs])
