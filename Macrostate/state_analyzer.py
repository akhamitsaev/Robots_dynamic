import torch
import torch.nn.functional as F

class SystemStateAnalyzer:
    def __init__(self, dt=0.1, robot_size=60.0):
        self.dt = dt
        self.robot_size = robot_size
        self.cutoff = robot_size * 2
        
    def compute_polar_order(self, angles):
        complex_vectors = torch.exp(1j * angles)
        return torch.abs(torch.mean(complex_vectors, dim=-1))
    
    def compute_mean_angle(self, angles):
        complex_mean = torch.mean(torch.exp(1j * angles), dim=-1)
        return torch.angle(complex_mean)
    
    def compute_coordination_number(self, coords):
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        distances = torch.norm(diff, dim=3)
        mask = distances < self.cutoff
        eye_mask = torch.eye(coords.shape[1], device=coords.device, dtype=torch.bool)
        mask[:, eye_mask] = False
        return mask.sum(dim=2).float().mean(dim=1)
    
    def compute_mean_distance(self, coords):
        n = coords.shape[1]
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        distances = torch.norm(diff, dim=3)
        mask = torch.eye(n, device=coords.device, dtype=torch.bool)
        distances = distances[:, ~mask].view(-1, n, n-1)
        return torch.mean(distances, dim=(1, 2))
    
    def compute_std_nearest_distance(self, coords):
        n_robots = coords.shape[1]
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        distances = torch.norm(diff, dim=3)
        mask = torch.eye(n_robots, device=coords.device, dtype=torch.bool)
        distances_masked = distances.masked_fill(mask, 1e6)
        nearest_dist = torch.min(distances_masked, dim=2)[0]
        return torch.std(nearest_dist, dim=1)
    
    def compute_angular_velocity_system(self, coords, velocities):
        centers = torch.mean(coords, dim=1, keepdim=True)
        radii = coords - centers
        distances = torch.norm(radii, dim=2)
        radial_vectors = radii / (distances.unsqueeze(2) + 1e-10)
        tangential_vectors = torch.stack([-radial_vectors[:, :, 1], 
                                         radial_vectors[:, :, 0]], dim=2)
        tangential_vel = torch.sum(velocities * tangential_vectors, dim=2)
        angular_vel = tangential_vel / (distances + 1e-10)
        return torch.mean(angular_vel, dim=1), torch.sign(torch.mean(angular_vel, dim=1))
    
    def compute_rotational_order(self, coords, velocities):
        centers = torch.mean(coords, dim=1, keepdim=True)
        radii = coords - centers
        radial_vectors = radii / (torch.norm(radii, dim=2, keepdim=True) + 1e-10)
        tangential_vectors = torch.stack([-radial_vectors[:, :, 1], 
                                         radial_vectors[:, :, 0]], dim=2)
        tangential_velocities = torch.sum(velocities * tangential_vectors, dim=2)
        return torch.mean(torch.abs(tangential_velocities), dim=1)
    
    def compute_center_velocity(self, velocities):
        return torch.mean(velocities, dim=1)
    
    def compute_velocity_dispersion(self, velocities):
        speed = torch.norm(velocities, dim=2)
        speed_mean = torch.mean(speed, dim=1)
        speed_std = torch.std(speed, dim=1)
        return speed_std / (speed_mean + 1e-10)
    
    def compute_mean_velocity(self, velocities):
        speed = torch.norm(velocities, dim=2)
        return torch.mean(speed, dim=1)
    
    def extract_features_batch(self, experiments_dict, kinematics_dict):
        features_dict = {}
        
        for file_name, exp_data in experiments_dict.items():
            tensor = exp_data['tensor']
            kinematics = kinematics_dict[file_name]
            
            coords = tensor[:, :, :2]
            angles = tensor[:, :, 2]
            velocities = kinematics['velocities']
            
            po = self.compute_polar_order(angles)
            mean_angle = self.compute_mean_angle(angles)
            coord_num = self.compute_coordination_number(coords)
            mean_dist = self.compute_mean_distance(coords)
            std_nearest_dist = self.compute_std_nearest_distance(coords)
            
            angular_vel, rotation_dir = self.compute_angular_velocity_system(coords, velocities)
            rot_order = self.compute_rotational_order(coords, velocities)
            center_vel = self.compute_center_velocity(velocities)
            velocity_disp = self.compute_velocity_dispersion(velocities)
            mean_vel = self.compute_mean_velocity(velocities)
            
            center_vel_norm = torch.norm(center_vel, dim=1)
            
            features = torch.stack([
                po,                    # 0: polar_order
                mean_angle,            # 1: mean_angle
                coord_num,             # 2: coord_num
                mean_dist,             # 3: mean_dist
                angular_vel,           # 4: angular_vel
                rotation_dir,          # 5: rotation_direction
                rot_order,             # 6: rot_order
                center_vel_norm,       # 7: center_vel
                velocity_disp,         # 8: velocity_dispersion
                std_nearest_dist,      # 9: std_nearest_dist
                mean_vel               # 10: mean_vel
            ], dim=1)
            
            features_dict[file_name] = features.cpu().numpy()
        
        return features_dict