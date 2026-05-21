# gnn_trainer.py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import time
import os
from sklearn.preprocessing import StandardScaler
import joblib
from graph_dataset import GraphDataset
from gnn_model import GNNModel

class GNNTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        # Добавляем скейлеры для нормализации
        self.target_scaler = StandardScaler()
        self.node_feature_scaler = StandardScaler()
        self.edge_feature_scaler = StandardScaler()
        self.is_fitted = False
    
    def _fit_scalers(self, train_dataset):
        """Обучение скейлеров на тренировочных данных"""
        print("📊 Обучение скейлеров для нормализации...")
        
        # Собираем все таргеты для обучения скейлера
        all_targets = []
        all_node_features = []
        all_edge_features = []
        
        for sample in train_dataset.samples:
            # Переносим на CPU перед добавлением
            all_targets.append(sample['targets'].cpu())
            all_node_features.append(sample['node_features'].cpu())
            all_edge_features.append(sample['edge_features'].cpu())
        
        # Конкатенируем и обучаем скейлеры
        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_node_features = torch.cat(all_node_features, dim=0).numpy()
        all_edge_features = torch.cat(all_edge_features, dim=0).numpy()
        
        self.target_scaler.fit(all_targets)
        self.node_feature_scaler.fit(all_node_features)
        self.edge_feature_scaler.fit(all_edge_features)
        
        self.is_fitted = True
        print("✅ Скейлеры обучены")
        
        # Сохраняем скейлеры для inference
        os.makedirs(os.path.dirname(self.config.model_save_path), exist_ok=True)
        joblib.dump(self.target_scaler, self.config.model_save_path.replace('.pth', '_target_scaler.pkl'))
        joblib.dump(self.node_feature_scaler, self.config.model_save_path.replace('.pth', '_node_scaler.pkl'))
        joblib.dump(self.edge_feature_scaler, self.config.model_save_path.replace('.pth', '_edge_scaler.pkl'))

    def _normalize_batch(self, batch):
        """Нормализация батча данных"""
        if not self.is_fitted:
            return batch
            
        # Нормализация node features
        node_features_np = batch['node_features'].cpu().numpy()
        batch['node_features'] = torch.tensor(
            self.node_feature_scaler.transform(node_features_np), 
            dtype=torch.float32, device=self.device  # Указываем device
        )
        
        # Нормализация edge features  
        edge_features_np = batch['edge_features'].cpu().numpy()
        batch['edge_features'] = torch.tensor(
            self.edge_feature_scaler.transform(edge_features_np),
            dtype=torch.float32, device=self.device
        )
        
        # Нормализация targets
        targets_np = batch['targets'].cpu().numpy()
        batch['targets'] = torch.tensor(
            self.target_scaler.transform(targets_np),
            dtype=torch.float32, device=self.device
        )
        
        return batch
    
    def graph_collate_fn(self, batch):
        """Кастомная collate функция с нормализацией"""
        if not batch:
            return {}
        
        node_features_list = []
        edge_index_list = []
        edge_features_list = []
        targets_list = []
        batch_indices = []
        
        node_offset = 0
        
        for i, graph in enumerate(batch):
            n_nodes = graph['node_features'].size(0)
            
            node_features_list.append(graph['node_features'])
            edge_index = graph['edge_index'] + node_offset
            edge_index_list.append(edge_index)
            edge_features_list.append(graph['edge_features'])
            targets_list.append(graph['targets'])
            batch_indices.append(torch.full((n_nodes,), i, dtype=torch.long))
            node_offset += n_nodes
        
        batch_dict = {
            'node_features': torch.cat(node_features_list, dim=0),
            'edge_index': torch.cat(edge_index_list, dim=1),
            'edge_features': torch.cat(edge_features_list, dim=0),
            'targets': torch.cat(targets_list, dim=0),
            'batch': torch.cat(batch_indices, dim=0),
            'num_graphs': len(batch)
        }
        
        # Применяем нормализацию
        return self._normalize_batch(batch_dict)
    
    def train_and_evaluate(self, experiments_dict, kinematics_dict, train_files, val_files, test_files):
        print("🚀 Подготовка данных для GNN...")
        
        # Создание датасетов
        train_dataset = GraphDataset(experiments_dict, kinematics_dict, train_files, 
                                   self.config.k_neighbors, self.config)
        val_dataset = GraphDataset(experiments_dict, kinematics_dict, val_files, 
                                 self.config.k_neighbors, self.config)
        test_dataset = GraphDataset(experiments_dict, kinematics_dict, test_files, 
                                  self.config.k_neighbors, self.config)
        
        print(f"📊 Размеры данных: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        # Обучаем скейлеры на тренировочных данных
        self._fit_scalers(train_dataset)
        
        # DataLoader с кастомной collate функцией
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, 
                                shuffle=True, num_workers=0, collate_fn=self.graph_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, 
                              shuffle=False, num_workers=0, collate_fn=self.graph_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, 
                               shuffle=False, num_workers=0, collate_fn=self.graph_collate_fn)
        
        # Модель и оптимизатор
        self.model = GNNModel(self.config).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, 
                                   weight_decay=self.config.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
        # Обучение
        train_metrics = self._train(train_loader, val_loader)
        
        # Тестирование
        test_metrics = self._evaluate(test_loader, "Тестирование")
        
        # Прогнозирование на всех данных
        all_predictions = self._predict_all(experiments_dict, kinematics_dict)
        
        # Сохранение результатов
        self._save_results(all_predictions)
        
        return {**train_metrics, **test_metrics}
    
    def _train(self, train_loader, val_loader):
        """Главный метод обучения с early stopping"""
        print("🎯 Начало обучения GNN...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        start_time = time.time()
        
        epoch_bar = tqdm(range(self.config.num_epochs), desc="🎯 Обучение GNN", 
                        disable=not self.config.show_progress)
        
        for epoch in epoch_bar:
            # Обучение на одной эпохе
            train_loss = self._train_epoch(train_loader, epoch)
            train_losses.append(train_loss)
            
            # Валидация
            val_loss = self._validate_epoch(val_loader)
            val_losses.append(val_loss)
            
            # Обновление learning rate
            self.scheduler.step(val_loss)
            
            # Логирование
            epoch_bar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'patience': f'{patience_counter}/{self.config.early_stopping_patience}'
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                print("🛑 Early stopping!")
                break
        
        training_time = (time.time() - start_time) / 60
        print(f"⏱ Обучение заняло: {training_time:.1f} минут")
        
        return {
            'gnn_train_time_minutes': training_time,
            'final_train_loss': train_losses[-1],
            'best_val_loss': best_val_loss
        }
    
    def _train_epoch(self, train_loader, epoch):
        """Training loop для одной эпохи"""
        self.model.train()
        total_loss = 0
        total_batches = len(train_loader)
        
        batch_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}', 
                        leave=False, disable=not self.config.show_progress)
        
        for batch_idx, batch in enumerate(batch_bar):
            # Перенос данных на устройство
            node_features = batch['node_features'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            edge_features = batch['edge_features'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(node_features, edge_index, edge_features)
            
            # Loss computation
            loss = self.criterion(predictions, targets)
            
            # Backward pass с gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Логирование
            if batch_idx % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                batch_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                    'lr': f'{current_lr:.2e}'
                })
        
        return total_loss / total_batches
    
    def _validate_epoch(self, val_loader):
        """Валидация на одной эпохе"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                node_features = batch['node_features'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                edge_features = batch['edge_features'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                predictions = self.model(node_features, edge_index, edge_features)
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _evaluate(self, test_loader, phase="Тестирование"):
        """Оценка модели на тестовом наборе"""
        print(f"🔍 {phase}...")
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=phase, disable=not self.config.show_progress):
                node_features = batch['node_features'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                edge_features = batch['edge_features'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                predictions = self.model(node_features, edge_index, edge_features)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        # Денормализуем для вычисления метрик
        all_predictions_denorm = torch.tensor(
            self.target_scaler.inverse_transform(all_predictions.numpy()),
            dtype=torch.float32
        )
        all_targets_denorm = torch.tensor(
            self.target_scaler.inverse_transform(all_targets.numpy()),
            dtype=torch.float32
        )
        
        metrics = self._calculate_metrics(all_predictions_denorm, all_targets_denorm)
        
        print(f"📊 Результаты {phase}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, predictions, targets):
        """Вычисление метрик для node-level predictions"""
        # RMSE для каждой компоненты
        rmse = torch.sqrt(torch.mean((predictions - targets)**2, dim=0))
        
        # MAPE с защитой от деления на ноль
        epsilon = 1e-2
        mape = torch.mean(torch.abs(predictions - targets) / (torch.abs(targets) + epsilon), dim=0) * 100
        
        return {
            'gnn_rmse_delta_x': rmse[0].item(),
            'gnn_rmse_delta_y': rmse[1].item(),
            'gnn_rmse_delta_angle': rmse[2].item(),
            'gnn_mape_delta_x': mape[0].item(),
            'gnn_mape_delta_y': mape[1].item(),
            'gnn_mape_delta_angle': mape[2].item(),
            'total_predictions': len(predictions)
        }
    
    def _predict_all(self, experiments_dict, kinematics_dict):
        """Прогнозирование для всех данных с нормализацией/денормализацией"""
        print("🔮 Прогнозирование для всех данных...")
        self.model.eval()
        
        all_predictions = []
        start_time = time.time()
        
        with torch.no_grad():
            for file_name, exp_data in experiments_dict.items():
                kinematics = kinematics_dict[file_name]
                n_timesteps, n_robots, _ = exp_data['tensor'].shape
                local_to_global = exp_data['local_to_global']
                
                for t in range(2, n_timesteps - 1):
                    # Данные для одного графа
                    node_features = self._get_all_node_features(kinematics, t, n_robots)
                    edge_index, edge_features = self._build_graph_for_prediction(kinematics, t, n_robots)
                    
                    # Нормализуем фичи для модели (на CPU)
                    node_features_norm = torch.tensor(
                        self.node_feature_scaler.transform(node_features.cpu().numpy()),
                        dtype=torch.float32, device=self.device
                    )
                    edge_features_norm = torch.tensor(
                        self.edge_feature_scaler.transform(edge_features.cpu().numpy()),
                        dtype=torch.float32, device=self.device
                    )
                    edge_index = edge_index.to(self.device)
                    
                    # Предсказания для всех узлов
                    predictions_norm = self.model(node_features_norm, edge_index, edge_features_norm).cpu()
                    
                    # Денормализуем предсказания
                    predictions = torch.tensor(
                        self.target_scaler.inverse_transform(predictions_norm.numpy()),
                        dtype=torch.float32
                    )
                    
                    # Вычисляем абсолютные координаты
                    current_coords = kinematics['coords'][t].cpu()
                    current_angles = kinematics['angles'][t].cpu()
                    
                    coord_pred = current_coords + predictions[:, :2]
                    angle_pred = current_angles + predictions[:, 2]
                    angle_pred = angle_pred % 360
                    
                    # Сохраняем результаты для каждого робота
                    for i in range(n_robots):
                        all_predictions.append({
                            'file_name': file_name,
                            'slice_id': t,
                            'bot_id': local_to_global[i],
                            'gnn_coord_x_pred': coord_pred[i, 0].item(),
                            'gnn_coord_y_pred': coord_pred[i, 1].item(),
                            'gnn_angle_pred': angle_pred[i].item(),
                            'coord_x_real': kinematics['coords'][t+1, i, 0].item(),
                            'coord_y_real': kinematics['coords'][t+1, i, 1].item(),
                            'angle_real': kinematics['angles'][t+1, i].item()
                        })
        
        inference_time = (time.time() - start_time) / 60
        print(f"⏱ Inference занял: {inference_time:.1f} минут")
        
        return pd.DataFrame(all_predictions)

    def _get_all_node_features(self, kinematics, t, n_robots):
        """Получение признаков всех роботов"""
        return torch.cat([
            kinematics['coords'][t],
            kinematics['angles'][t].unsqueeze(1),
            kinematics['velocities'][t],
            kinematics['angular_velocities'][t].unsqueeze(1),
            kinematics['accelerations'][t],
            kinematics['angular_accelerations'][t].unsqueeze(1)
        ], dim=1)
    
    def _build_graph_for_prediction(self, kinematics, t, n_robots):
        """Построение графа для предсказания"""
        distances = torch.cdist(kinematics['coords'][t], kinematics['coords'][t])
        
        # Индексы k+1 ближайших соседей
        _, neighbor_indices = torch.topk(distances, k=self.config.k_neighbors + 1, dim=1, largest=False)
        
        # Убираем самого себя
        valid_neighbor_indices = neighbor_indices[:, 1:self.config.k_neighbors+1]
        
        # Создаем edge_index
        robot_indices = torch.arange(n_robots, device=distances.device)
        robot_indices_expanded = robot_indices.unsqueeze(1).repeat(1, self.config.k_neighbors)
        
        edge_index = torch.stack([
            robot_indices_expanded.flatten(),
            valid_neighbor_indices.flatten()
        ])
        
        # Вычисляем edge_features
        edge_features = self._compute_edge_features_vectorized(
            kinematics, t, robot_indices_expanded, valid_neighbor_indices, n_robots
        )
        
        return edge_index, edge_features
    
    def _compute_edge_features_vectorized(self, kinematics, t, robot_indices, neighbor_indices, n_robots):
        """Векторизованное вычисление признаков ребер"""
        robot_indices_flat = robot_indices.flatten()
        neighbor_indices_flat = neighbor_indices.flatten()
        
        robot_coords = kinematics['coords'][t, robot_indices_flat]
        robot_angles = kinematics['angles'][t, robot_indices_flat]
        robot_vels = kinematics['velocities'][t, robot_indices_flat]
        robot_accs = kinematics['accelerations'][t, robot_indices_flat]
        robot_ang_vels = kinematics['angular_velocities'][t, robot_indices_flat]
        robot_ang_accs = kinematics['angular_accelerations'][t, robot_indices_flat]
        
        neighbor_coords = kinematics['coords'][t, neighbor_indices_flat]
        neighbor_angles = kinematics['angles'][t, neighbor_indices_flat]
        neighbor_vels = kinematics['velocities'][t, neighbor_indices_flat]
        neighbor_accs = kinematics['accelerations'][t, neighbor_indices_flat]
        neighbor_ang_vels = kinematics['angular_velocities'][t, neighbor_indices_flat]
        neighbor_ang_accs = kinematics['angular_accelerations'][t, neighbor_indices_flat]
        
        delta_coords = neighbor_coords - robot_coords
        delta_x = delta_coords[:, 0]
        delta_y = delta_coords[:, 1]
        distances = torch.sqrt(delta_x**2 + delta_y**2)
        relative_angles = torch.atan2(delta_y, delta_x)
        angle_diffs = neighbor_angles - robot_angles
        
        relative_vels = neighbor_vels - robot_vels
        relative_accs = neighbor_accs - robot_accs
        angular_vel_diffs = neighbor_ang_vels - robot_ang_vels
        angular_acc_diffs = neighbor_ang_accs - robot_ang_accs
        
        edge_features = torch.stack([
            delta_x, delta_y, distances, relative_angles, angle_diffs,
            relative_vels[:, 0], relative_vels[:, 1],
            relative_accs[:, 0], relative_accs[:, 1],
            angular_vel_diffs, angular_acc_diffs
        ], dim=1)
        
        return edge_features
    
    def _save_checkpoint(self, epoch, val_loss):
        """Сохранение модели"""
        os.makedirs(os.path.dirname(self.config.model_save_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, self.config.model_save_path)
    
    def _save_results(self, predictions_df):
        """Сохранение результатов"""
        os.makedirs(os.path.dirname(self.config.results_save_path), exist_ok=True)
        predictions_df.to_parquet(self.config.results_save_path, index=False)
        print(f"💾 Результаты GNN сохранены: {self.config.results_save_path}")