import torch
import pandas as pd
import numpy as np
from pathlib import Path

class DataPreprocessor:
    def __init__(self, data_path, dt=0.1):
        self.data_path = data_path
        self.dt = dt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_and_vectorize(self):
        """Векторизованная загрузка данных в torch тензоры"""
        print("📥 Векторизованная загрузка данных...")
        
        df = pd.read_parquet(self.data_path)
        experiments = {}
        
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
                device=self.device, dtype=torch.float32
            )
            
            experiments[file_name] = {
                'tensor': tensor,
                'global_to_local': global_to_local,
                'local_to_global': local_to_global,
                'n_robots': n_robots,
                'n_timesteps': n_timesteps
            }
        
        print(f"🎯 Загружено {len(experiments)} экспериментов на {self.device}")
        return experiments
    
    def compute_kinematics_vectorized(self, experiments_dict):
        """Векторизованное вычисление кинематических характеристик"""
        print("🔬 Векторизованное вычисление кинематики...")
        
        dt = self.dt
        kinematics_dict = {}
        
        for file_name, exp_data in experiments_dict.items():
            tensor = exp_data['tensor']  # [timesteps, robots, 3]
            
            with torch.no_grad():
                coords = tensor[:, :, 0:2]  # [timesteps, robots, 2]
                angles = tensor[:, :, 2]    # [timesteps, robots]
                
                # Скорости: V(t) = (x(t) - x(t-1)) / dt
                velocities = torch.zeros_like(coords)
                velocities[1:] = (coords[1:] - coords[:-1]) / dt
                
                # Ускорения: a(t) = (V(t) - V(t-1)) / dt
                accelerations = torch.zeros_like(coords)
                accelerations[2:] = (velocities[2:] - velocities[1:-1]) / dt
                
                # Угловые скорости: ω(t) = (θ(t) - θ(t-1)) / dt
                angular_velocities = torch.zeros_like(angles)
                angular_velocities[1:] = (angles[1:] - angles[:-1]) / dt
                
                # Угловые ускорения: α(t) = (ω(t) - ω(t-1)) / dt
                angular_accelerations = torch.zeros_like(angles)
                angular_accelerations[2:] = (angular_velocities[2:] - angular_velocities[1:-1]) / dt
            
            kinematics_dict[file_name] = {
                'coords': coords,
                'angles': angles,
                'velocities': velocities,
                'accelerations': accelerations,
                'angular_velocities': angular_velocities,
                'angular_accelerations': angular_accelerations
            }
        
        return kinematics_dict