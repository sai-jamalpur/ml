import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SpatioTemporalGraphModel(nn.Module):
    def __init__(self, num_nodes=49, input_dim=5, temporal_hidden=128, spatial_hidden=64, gat_heads=4, dropout=0.2):
        super(SpatioTemporalGraphModel, self).__init__()
        self.num_nodes = num_nodes
        
        self.temporal_gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=temporal_hidden, 
            num_layers=2, 
            batch_first=True,
            dropout=dropout
        )
        self.temporal_norm = nn.LayerNorm(temporal_hidden)
        
        self.spatial_gat = GATConv(
            in_channels=temporal_hidden, 
            out_channels=spatial_hidden // gat_heads, 
            heads=gat_heads, 
            concat=True,
            dropout=dropout
        )
        self.spatial_norm = nn.LayerNorm(spatial_hidden)
        
        self.regressor = nn.Sequential(
            nn.Linear(spatial_hidden, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def _create_batched_edge_index(self, base_edge_index, batch_size, device):
        batched_edges = []
        for b in range(batch_size):
            offset = b * self.num_nodes
            batched_edges.append(base_edge_index + offset)
        return torch.cat(batched_edges, dim=1).to(device)

    def forward(self, x, base_edge_index):
        B, N, W, F_dim = x.shape
        device = x.device
        
        x_flat = x.view(B * N, W, F_dim) 
        _, hidden = self.temporal_gru(x_flat)
        
        temporal_emb = hidden[-1]
        temporal_emb = self.temporal_norm(temporal_emb)
        
        batched_edge_index = self._create_batched_edge_index(base_edge_index, B, device)
        
        spatial_emb = self.spatial_gat(temporal_emb, batched_edge_index)
        spatial_emb = F.leaky_relu(self.spatial_norm(spatial_emb), 0.2)
        
        predictions_flat = self.regressor(spatial_emb)
        return predictions_flat.view(B, N)