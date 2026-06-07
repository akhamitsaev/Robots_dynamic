from __future__ import annotations

import torch

def node_features_from_kinematics(kinematics: dict, t: int) -> torch.Tensor:
    return torch.cat([
        kinematics["coords"][t],
        kinematics["angles"][t].unsqueeze(1),
        kinematics["velocities"][t],
        kinematics["angular_velocities"][t].unsqueeze(1),
        kinematics["accelerations"][t],
        kinematics["angular_accelerations"][t].unsqueeze(1),
    ], dim=1)


def build_knn_graph_from_kinematics(kinematics: dict, t: int, k_neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
    coords_t = kinematics["coords"][t]
    n_robots = coords_t.shape[0]
    k = min(k_neighbors, max(1, n_robots - 1))

    distances = torch.cdist(coords_t, coords_t)
    _, neighbor_indices = torch.topk(distances, k=k + 1, dim=1, largest=False)
    valid_neighbor_indices = neighbor_indices[:, 1:k + 1]

    source_indices = torch.arange(n_robots, device=coords_t.device).unsqueeze(1).repeat(1, k)
    edge_index = torch.stack([source_indices.flatten(), valid_neighbor_indices.flatten()])
    edge_features = compute_edge_features(kinematics, t, source_indices.flatten(), valid_neighbor_indices.flatten())
    return edge_index.long(), edge_features


def compute_edge_features(kinematics: dict, t: int, source_indices: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    source_coords = kinematics["coords"][t, source_indices]
    source_angles = kinematics["angles"][t, source_indices]
    source_vels = kinematics["velocities"][t, source_indices]
    source_accs = kinematics["accelerations"][t, source_indices]
    source_ang_vels = kinematics["angular_velocities"][t, source_indices]
    source_ang_accs = kinematics["angular_accelerations"][t, source_indices]

    target_coords = kinematics["coords"][t, target_indices]
    target_angles = kinematics["angles"][t, target_indices]
    target_vels = kinematics["velocities"][t, target_indices]
    target_accs = kinematics["accelerations"][t, target_indices]
    target_ang_vels = kinematics["angular_velocities"][t, target_indices]
    target_ang_accs = kinematics["angular_accelerations"][t, target_indices]

    delta_coords = target_coords - source_coords
    delta_x = delta_coords[:, 0]
    delta_y = delta_coords[:, 1]
    distances = torch.sqrt(delta_x ** 2 + delta_y ** 2)
    relative_angles = torch.atan2(delta_y, delta_x)
    # Match StepPrediction/GraphDataset: raw angle difference, no cyclic correction.
    angle_diffs = target_angles - source_angles

    relative_vels = target_vels - source_vels
    relative_accs = target_accs - source_accs
    angular_vel_diffs = target_ang_vels - source_ang_vels
    angular_acc_diffs = target_ang_accs - source_ang_accs

    return torch.stack([
        delta_x,
        delta_y,
        distances,
        relative_angles,
        angle_diffs,
        relative_vels[:, 0],
        relative_vels[:, 1],
        relative_accs[:, 0],
        relative_accs[:, 1],
        angular_vel_diffs,
        angular_acc_diffs,
    ], dim=1)
