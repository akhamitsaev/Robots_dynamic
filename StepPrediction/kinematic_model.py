import torch
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import re

class KinematicBaselineGPU:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Используется устройство: {self.device}")
    
    def predict_one_step_vectorized(self, experiments_dict, kinematics_dict):
        """Векторизованный one-step прогноз с t=2 на t=3"""
        print("🔮 Векторизованный one-step прогноз...")
        
        dt = self.config.dt
        predictions_dict = {}
        
        for file_name, exp_data in experiments_dict.items():
            kinematics = kinematics_dict[file_name]
            n_timesteps, n_robots, _ = exp_data['tensor'].shape
            
            with torch.no_grad():
                predictions = torch.zeros_like(exp_data['tensor'])
                
                # Создаем тензоры индексов вместо slice
                # valid_indices - индексы времени t (начиная с 3)
                # prev_indices - индексы времени t-1 (начиная с 2)
                valid_indices = torch.arange(3, n_timesteps, device=self.device)
                prev_indices = valid_indices - 1
                
                # Прогноз координат: x_pred(t) = x(t-1) + V(t-1)*dt + 0.5*a(t-1)*dt²
                predictions[valid_indices, :, 0:2] = (
                    kinematics['coords'][prev_indices] +                    # x(t-1)
                    kinematics['velocities'][prev_indices] * dt +          # V(t-1)
                    0.5 * kinematics['accelerations'][prev_indices] * (dt**2)  # a(t-1)
                )
                
                # Прогноз углов: θ_pred(t) = θ(t-1) + ω(t-1)*dt + 0.5*α(t-1)*dt²
                angle_predictions = (
                    kinematics['angles'][prev_indices] +
                    kinematics['angular_velocities'][prev_indices] * dt +
                    0.5 * kinematics['angular_accelerations'][prev_indices] * (dt**2)
                )
                predictions[valid_indices, :, 2] = angle_predictions % 360
                
            predictions_dict[file_name] = predictions
        
        return predictions_dict
    
    def create_results_dataframe_vectorized(self, experiments_dict, predictions_dict, kinematics_dict):
        """Создание DataFrame с консистентными данными на время t"""
        print("💾 Векторизованное формирование результатов (консистентные данные на время t)...")
        
        results_list = []
        
        for file_name, exp_data in experiments_dict.items():
            pred_data = predictions_dict[file_name]
            kinematics = kinematics_dict[file_name]
            tensor = exp_data['tensor']
            local_to_global = exp_data['local_to_global']
            
            n_timesteps, n_robots, _ = tensor.shape
            
            # Прогноз начинается с t=2 (т.к. используем данные t-1)
            t_start, t_end = 2, n_timesteps
            n_valid_timesteps = t_end - t_start
            
            t_indices = torch.arange(t_start, t_end).repeat_interleave(n_robots)
            r_indices = torch.arange(n_robots).repeat(n_valid_timesteps)
            
            # ИЗМЕНЕНИЕ: Все данные берём для времени t
            real_vals_t = tensor[t_indices, r_indices, :]          # реальные на t
            pred_vals_t = pred_data[t_indices, r_indices, :]       # прогноз на t (вычислен из t-1)
            
            # Кинематика на время t (для GNN)
            coords_t = kinematics['coords'][t_indices, r_indices, :]
            angles_t = kinematics['angles'][t_indices, r_indices]
            vel_t = kinematics['velocities'][t_indices, r_indices, :]
            acc_t = kinematics['accelerations'][t_indices, r_indices, :]
            ang_vel_t = kinematics['angular_velocities'][t_indices, r_indices]
            ang_acc_t = kinematics['angular_accelerations'][t_indices, r_indices]
            
            df = pd.DataFrame({
                'file_name': file_name,
                'slice_id': t_indices.cpu().numpy(),  # время t
                
                # Реальные значения на момент t
                'coord_x_real': real_vals_t[:, 0].cpu().numpy(),
                'coord_y_real': real_vals_t[:, 1].cpu().numpy(),
                'angle_real': real_vals_t[:, 2].cpu().numpy(),
                
                # Прогнозные значения на момент t (вычисленные из t-1)
                'coord_x_pred': pred_vals_t[:, 0].cpu().numpy(),
                'coord_y_pred': pred_vals_t[:, 1].cpu().numpy(),
                'angle_pred': pred_vals_t[:, 2].cpu().numpy(),
                
                # Кинематика на момент t (для GNN)
                'coord_x': coords_t[:, 0].cpu().numpy(),
                'coord_y': coords_t[:, 1].cpu().numpy(),
                'angle': angles_t.cpu().numpy(),
                'velocity_x': vel_t[:, 0].cpu().numpy(),
                'velocity_y': vel_t[:, 1].cpu().numpy(),
                'acceleration_x': acc_t[:, 0].cpu().numpy(),
                'acceleration_y': acc_t[:, 1].cpu().numpy(),
                'angular_velocity': ang_vel_t.cpu().numpy(),
                'angular_acceleration': ang_acc_t.cpu().numpy(),
                
                # bot_id в конце
                'bot_id': [local_to_global[r.item()] for r in r_indices],
            })
            
            results_list.append(df)
        
        return pd.concat(results_list, ignore_index=True)
    
    def correct_cyclic_difference(self, delta):
        """Коррекция циклических углов"""
        delta = np.where(delta > 180, delta - 360, delta)
        delta = np.where(delta < -180, delta + 360, delta)
        return delta

    def calculate_metrics(self, results_df):
        """Вычисление метрик качества для дельт изменений"""
        print("📊 Вычисление метрик для дельт изменений...")
        
        # Для вычисления дельт нужны данные с предыдущего шага t-1
        # Группируем по файлу и роботу, сортируем по времени
        results_df = results_df.sort_values(['file_name', 'bot_id', 'slice_id'])
        
        # Вычисляем реальные дельты: real(t) - real(t-1)
        results_df['delta_x_real'] = results_df.groupby(['file_name', 'bot_id'])['coord_x_real'].diff()
        results_df['delta_y_real'] = results_df.groupby(['file_name', 'bot_id'])['coord_y_real'].diff()
        results_df['delta_angle_real'] = results_df.groupby(['file_name', 'bot_id'])['angle_real'].diff()
        
        # Вычисляем прогнозные дельты: pred(t) - real(t-1)
        results_df['delta_x_pred'] = results_df['coord_x_pred'] - results_df.groupby(['file_name', 'bot_id'])['coord_x_real'].shift(1)
        results_df['delta_y_pred'] = results_df['coord_y_pred'] - results_df.groupby(['file_name', 'bot_id'])['coord_y_real'].shift(1)
        results_df['delta_angle_pred'] = results_df['angle_pred'] - results_df.groupby(['file_name', 'bot_id'])['angle_real'].shift(1)
        
        # Циклическая коррекция для угловых дельт
        results_df['delta_angle_real'] = self.correct_cyclic_difference(results_df['delta_angle_real'])
        results_df['delta_angle_pred'] = self.correct_cyclic_difference(results_df['delta_angle_pred'])
        
        # Убираем NaN значения (первые строки в группах)
        valid_data = results_df.dropna(subset=['delta_x_real', 'delta_x_pred'])
        
        # Метрики
        epsilon = self.config.epsilon

        # Ошибки прогноза дельт
        delta_x_error = valid_data['delta_x_pred'] - valid_data['delta_x_real']+ epsilon
        delta_y_error = valid_data['delta_y_pred'] - valid_data['delta_y_real']+ epsilon
        delta_angle_error = valid_data['delta_angle_pred'] - valid_data['delta_angle_real']+ epsilon
        
        metrics = {
            'rmse_delta_x': np.sqrt(np.mean(delta_x_error**2)),
            'rmse_delta_y': np.sqrt(np.mean(delta_y_error**2)),
            'rmse_delta_angle': np.sqrt(np.mean(delta_angle_error**2)),
            
            'mape_delta_x': np.mean(np.abs(delta_x_error) / (np.abs(valid_data['delta_x_real']) + epsilon)) * 100,
            'mape_delta_y': np.mean(np.abs(delta_y_error) / (np.abs(valid_data['delta_y_real']) + epsilon)) * 100,
            'mape_delta_angle': np.mean(np.abs(delta_angle_error) / (np.abs(valid_data['delta_angle_real']) + epsilon)) * 100,
            
            'total_predictions': len(valid_data),
            'valid_delta_predictions': len(valid_data)
        }
        
        return metrics