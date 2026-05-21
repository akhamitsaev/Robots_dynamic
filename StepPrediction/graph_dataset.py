# graph_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm.auto import tqdm

class GraphDataset(Dataset):
    def __init__(self, experiments_dict, kinematics_dict, file_names, k_neighbors=5, config=None):
        self.samples = []
        self.config = config
        self.k_neighbors = k_neighbors
        self.device = kinematics_dict[list(kinematics_dict.keys())[0]]['coords'].device
        
        print("🚀 Запуск оптимизированного GraphDataset с полной векторизацией...")
        self._build_samples_optimized(experiments_dict, kinematics_dict, file_names, k_neighbors)
    
    def _build_samples_optimized(self, experiments_dict, kinematics_dict, file_names, k_neighbors):
        """Оптимизированное построение графов с максимальной векторизацией"""
        all_samples = []
        total_graphs = 0
        
        # Предварительный расчет общего количества графов для прогресс-бара
        for file_name in file_names:
            if file_name in experiments_dict:
                n_timesteps = experiments_dict[file_name]['n_timesteps']
                total_graphs += max(0, n_timesteps - 3)  # t от 2 до n_timesteps-2
        
        print(f"📊 Всего графов для создания: {total_graphs}")
        
        with tqdm(total=total_graphs, desc="🔧 Векторизованное создание графов",
                 disable=not self.config.show_progress) as pbar:
            
            for file_name in file_names:
                if file_name not in experiments_dict:
                    continue
                    
                exp_data = experiments_dict[file_name]
                kinematics = kinematics_dict[file_name]
                n_timesteps = exp_data['n_timesteps']
                n_robots = exp_data['n_robots']
                
                # Векторизованная обработка ВСЕХ временных шагов для этого файла
                file_samples = self._process_file_all_timesteps_vectorized(
                    kinematics, n_timesteps, n_robots, k_neighbors, file_name, pbar
                )
                all_samples.extend(file_samples)
        
        self.samples = all_samples
        print(f"✅ Создано {len(all_samples)} графов (оптимизированная векторизация)")
    
    def _process_file_all_timesteps_vectorized(self, kinematics, n_timesteps, n_robots, k_neighbors, file_name, pbar):
        """Векторизованная обработка ВСЕХ временных шагов одного файла"""
        file_samples = []
        
        # Предварительное вычисление ВСЕХ необходимых данных
        time_steps = list(range(2, n_timesteps - 1))
        if not time_steps:
            return file_samples
        
        # Векторизованное вычисление ВСЕХ node features для всех временных шагов
        all_node_features = self._get_all_timesteps_node_features(kinematics, time_steps, n_robots)
        
        # Векторизованное построение ВСЕХ графов
        for i, t in enumerate(time_steps):
            try:
                node_features = all_node_features[i]
                edge_index, edge_features = self._build_single_graph_edges_vectorized(
                    kinematics, t, n_robots, k_neighbors
                )
                targets = self._get_single_timestep_targets(kinematics, t, n_robots)
                
                sample = {
                    'node_features': node_features,
                    'edge_index': edge_index,
                    'edge_features': edge_features,
                    'targets': targets,
                    'num_nodes': n_robots,
                    'file_name': file_name,
                    'time_step': t
                }
                file_samples.append(sample)
                pbar.update(1)
                
            except Exception as e:
                print(f"⚠️ Ошибка в файле {file_name}, время {t}: {e}")
                continue
        
        return file_samples
    
    def _get_all_timesteps_node_features(self, kinematics, time_steps, n_robots):
        """Векторизованное получение node features для ВСЕХ временных шагов"""
        # Используем list comprehension для быстрого создания всех features
        return [self._get_single_timestep_node_features(kinematics, t, n_robots) for t in time_steps]
    
    def _get_single_timestep_node_features(self, kinematics, t, n_robots):
        """Векторизованное получение node features для одного временного шага"""
        return torch.cat([
            kinematics['coords'][t],                                    # [n_robots, 2]
            kinematics['angles'][t].unsqueeze(1),                       # [n_robots, 1]
            kinematics['velocities'][t],                                # [n_robots, 2]
            kinematics['angular_velocities'][t].unsqueeze(1),           # [n_robots, 1]
            kinematics['accelerations'][t],                             # [n_robots, 2]
            kinematics['angular_accelerations'][t].unsqueeze(1)         # [n_robots, 1]
        ], dim=1)                                                       # Итого: [n_robots, 9]
    
    def _build_single_graph_edges_vectorized(self, kinematics, t, n_robots, k_neighbors):
        """Полностью векторизованное построение ребер графа для одного временного шага"""
        # 1. Вычисление попарных расстояний [n_robots, n_robots]
        coords_t = kinematics['coords'][t]
        distances = torch.cdist(coords_t, coords_t)
        
        # 2. Нахождение k ближайших соседей для каждого робота
        # Получаем индексы k+1 ближайших (включая самого себя)
        _, neighbor_indices = torch.topk(distances, k=k_neighbors + 1, dim=1, largest=False)
        
        # Убираем самого себя - берем соседей со 2 по k+1 [n_robots, k]
        valid_neighbor_indices = neighbor_indices[:, 1:k_neighbors+1]
        
        # 3. Создание edge_index [2, n_robots * k]
        robot_indices = torch.arange(n_robots, device=self.device)
        robot_indices_expanded = robot_indices.unsqueeze(1).repeat(1, k_neighbors)
        
        edge_index = torch.stack([
            robot_indices_expanded.flatten(),
            valid_neighbor_indices.flatten()
        ])
        
        # 4. Векторизованное вычисление edge features [n_edges, 11]
        edge_features = self._compute_edge_features_batch_vectorized(
            kinematics, t, robot_indices_expanded.flatten(), valid_neighbor_indices.flatten()
        )
        
        return edge_index, edge_features
    
    def _compute_edge_features_batch_vectorized(self, kinematics, t, source_indices, target_indices):
        """Векторизованное вычисление признаков ребер для ВСЕХ ребер"""
        # Получаем данные для исходных и целевых узлов
        source_coords = kinematics['coords'][t, source_indices]
        source_angles = kinematics['angles'][t, source_indices]
        source_vels = kinematics['velocities'][t, source_indices]
        source_accs = kinematics['accelerations'][t, source_indices]
        source_ang_vels = kinematics['angular_velocities'][t, source_indices]
        source_ang_accs = kinematics['angular_accelerations'][t, source_indices]
        
        target_coords = kinematics['coords'][t, target_indices]
        target_angles = kinematics['angles'][t, target_indices]
        target_vels = kinematics['velocities'][t, target_indices]
        target_accs = kinematics['accelerations'][t, target_indices]
        target_ang_vels = kinematics['angular_velocities'][t, target_indices]
        target_ang_accs = kinematics['angular_accelerations'][t, target_indices]
        
        # Векторизованные вычисления всех признаков
        delta_coords = target_coords - source_coords
        delta_x = delta_coords[:, 0]
        delta_y = delta_coords[:, 1]
        distances = torch.sqrt(delta_x**2 + delta_y**2)
        relative_angles = torch.atan2(delta_y, delta_x)
        angle_diffs = target_angles - source_angles
        
        relative_vels = target_vels - source_vels
        relative_accs = target_accs - source_accs
        angular_vel_diffs = target_ang_vels - source_ang_vels
        angular_acc_diffs = target_ang_accs - source_ang_accs
        
        # Собираем все признаки [n_edges, 11]
        edge_features = torch.stack([
            delta_x, delta_y, distances, relative_angles, angle_diffs,
            relative_vels[:, 0], relative_vels[:, 1],
            relative_accs[:, 0], relative_accs[:, 1],
            angular_vel_diffs, angular_acc_diffs
        ], dim=1)
        
        return edge_features
    
    def _get_single_timestep_targets(self, kinematics, t, n_robots):
        """Векторизованное получение целевых значений для одного временного шага"""
        delta_coords = kinematics['coords'][t + 1] - kinematics['coords'][t]      # [n_robots, 2]
        delta_angles = kinematics['angles'][t + 1] - kinematics['angles'][t]      # [n_robots]
        
        # Коррекция циклических углов
        delta_angles = torch.where(delta_angles > 180, delta_angles - 360, delta_angles)
        delta_angles = torch.where(delta_angles < -180, delta_angles + 360, delta_angles)
        
        return torch.cat([delta_coords, delta_angles.unsqueeze(1)], dim=1)        # [n_robots, 3]
    
    def _validate_graph_structure(self, node_features, edge_index, edge_features, targets, n_robots):
        """Валидация структуры графа"""
        assert node_features.dim() == 2, f"node_features должен быть 2D, получен {node_features.dim()}D"
        assert edge_index.dim() == 2, f"edge_index должен быть 2D, получен {edge_index.dim()}D"
        assert edge_features.dim() == 2, f"edge_features должен быть 2D, получен {edge_features.dim()}D"
        assert targets.dim() == 2, f"targets должен быть 2D, получен {targets.dim()}D"
        
        assert node_features.shape == (n_robots, 9), f"node_features: ожидается ({n_robots}, 9), получен {node_features.shape}"
        assert edge_index.shape[0] == 2, f"edge_index: ожидается (2, n_edges), получен {edge_index.shape}"
        assert edge_features.shape[0] == edge_index.shape[1], f"Несоответствие: {edge_features.shape[0]} ребер в features vs {edge_index.shape[1]} в edge_index"
        assert targets.shape == (n_robots, 3), f"targets: ожидается ({n_robots}, 3), получен {targets.shape}"
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Дополнительная валидация при доступе
        self._validate_graph_structure(
            sample['node_features'],
            sample['edge_index'], 
            sample['edge_features'],
            sample['targets'],
            sample['num_nodes']
        )
        
        return sample