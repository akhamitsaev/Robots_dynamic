import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


class GNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Параметры конфигурации
        self.hidden_dim = config.hidden_dim          # 128
        self.num_layers = config.gnn_layers          # 2
        self.dropout_p = getattr(config, "dropout", 0.1)

        # Размерности входных признаков (как у тебя в dataset)
        self.node_in_dim = 9
        self.edge_in_dim = 11

        # --- Энкодеры ---
        self.node_encoder = nn.Linear(self.node_in_dim, self.hidden_dim)
        self.edge_encoder = nn.Linear(self.edge_in_dim, self.hidden_dim)

        # --- GINEConv слои ---
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            # MLP для GINEConv
            mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

            conv = GINEConv(mlp)
            self.layers.append(conv)
            self.norms.append(nn.BatchNorm1d(self.hidden_dim))

        self.dropout = nn.Dropout(self.dropout_p)

        # --- Предсказатель ---
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 3),  # Δx, Δy, Δangle
        )

    def forward(self, x, edge_index, edge_attr):
        # Encode features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # Message passing
        for conv, norm in zip(self.layers, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Node predictions
        return self.predictor(x)
