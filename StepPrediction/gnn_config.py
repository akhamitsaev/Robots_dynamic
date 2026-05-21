class GNNConfig:
    def __init__(self):
        # Архитектура
        self.k_neighbors = 5
        self.hidden_dim = 128
        self.edge_dim = 32
        self.gnn_layers = 2
        self.gnn_type = "GIN"  # GIN, GAT, GCN
        
        # Обучение
        self.batch_size = 512
        self.num_epochs = 50
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        
        # Мониторинг
        self.show_progress = True
        self.log_interval = 10
        self.early_stopping_patience = 10
        
        # Пути
        self.model_save_path = "models/gnn_model_weights.pth"
        self.results_save_path = "results/gnn_predictions.parquet"