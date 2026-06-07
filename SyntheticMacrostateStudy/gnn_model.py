import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


class GNNModel(nn.Module):
    """Simple edge-aware GNN for one-step node-level delta prediction.

    Inputs:
        x: [num_nodes, 9]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, 11]
    Output:
        [num_nodes, 3] = delta_x, delta_y, delta_angle
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.gnn_layers
        self.dropout_p = getattr(config, "dropout", 0.1)

        self.node_encoder = nn.Linear(9, self.hidden_dim)
        self.edge_encoder = nn.Linear(11, self.hidden_dim)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.layers.append(GINEConv(mlp))
            self.norms.append(nn.BatchNorm1d(self.hidden_dim))

        self.dropout = nn.Dropout(self.dropout_p)
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 3),
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv, norm in zip(self.layers, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)

        return self.predictor(x)
