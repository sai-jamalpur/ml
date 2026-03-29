import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class NiftyDatasetCreator:
    def __init__(self, ohlcv_path, returns_path):
        self.ohlcv_path = os.path.abspath(ohlcv_path)
        self.returns_path = os.path.abspath(returns_path)
        
        if not os.path.exists(self.ohlcv_path):
            raise FileNotFoundError(f"OHLCV file not found: {self.ohlcv_path}")
        if not os.path.exists(self.returns_path):
            raise FileNotFoundError(f"Returns file not found: {self.returns_path}")
            
        self.load_data()
        
    def load_data(self):
        print("Parsing Returns Data...")
        ret_df = pd.read_csv(self.returns_path, index_col=0)
        ret_tickers = list(ret_df.columns)
        self.num_time_steps = len(ret_df)
        
        print("Mapping OHLCV Data Columns...")
        with open(self.ohlcv_path, 'r') as f:
            lines = f.readlines()
            
        features_line = lines[0].strip('\n').split(',')
        tickers_line = lines[1].strip('\n').split(',')
        
        # Build explicit column index map
        ticker_to_cols = {}
        for col_idx in range(1, min(len(features_line), len(tickers_line))):
            feat = features_line[col_idx].strip()
            tick = tickers_line[col_idx].strip()
            
            if tick and feat:
                if tick not in ticker_to_cols:
                    ticker_to_cols[tick] = {}
                ticker_to_cols[tick][feat] = col_idx
                
        expected_features = ['open', 'high', 'low', 'close', 'volume']
        
        # Structural Intersection: Keep only aligned tickers
        self.tickers = []
        for t in ret_tickers:
            if t in ticker_to_cols and all(f in ticker_to_cols[t] for f in expected_features):
                self.tickers.append(t)
                
        self.num_nodes = len(self.tickers)
        self.num_features = len(expected_features)
        
        print(f"Successfully aligned {self.num_nodes} tickers present in both datasets.")
        
        self.returns_tensor = ret_df[self.tickers].values 
        
        data_lines = [line for line in lines[3:] if line.strip()] 
        actual_time_steps = min(self.num_time_steps, len(data_lines))
        self.returns_tensor = self.returns_tensor[:actual_time_steps]
        self.num_time_steps = actual_time_steps
        
        self.ohlcv_tensor = np.zeros((self.num_time_steps, self.num_nodes, self.num_features))
        
        for t in range(self.num_time_steps):
            parts = data_lines[t].strip('\n').split(',')
            for node_idx, ticker in enumerate(self.tickers):
                for feat_idx, feat in enumerate(expected_features):
                    col_idx = ticker_to_cols[ticker][feat]
                    try:
                        val = float(parts[col_idx]) if col_idx < len(parts) and parts[col_idx] else 0.0
                    except ValueError:
                        val = 0.0
                    self.ohlcv_tensor[t, node_idx, feat_idx] = val
                    
        print("Applying Temporal Z-Score Normalization...")
        mean = np.mean(self.ohlcv_tensor, axis=0, keepdims=True)
        std = np.std(self.ohlcv_tensor, axis=0, keepdims=True) + 1e-8
        self.ohlcv_normalized = (self.ohlcv_tensor - mean) / std
        
    def get_correlation_graph(self, threshold=0.35):
        print(f"Building Spatial Graph Adjacency Matrix (Threshold > {threshold})...")
        corr_matrix = np.corrcoef(self.returns_tensor, rowvar=False)
        source_nodes, target_nodes = [], []
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and abs(corr_matrix[i, j]) > threshold:
                    source_nodes.append(i)
                    target_nodes.append(j)
                    
        return torch.tensor([source_nodes, target_nodes], dtype=torch.long)


class SpatioTemporalStockDataset(Dataset):
    def __init__(self, features, returns, window_size=5):
        self.features = features
        self.returns = returns
        self.window_size = window_size
        self.num_samples = len(features) - window_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_window = self.features[idx : idx + self.window_size] 
        x_window = np.transpose(x_window, (1, 0, 2))
        y_target = self.returns[idx + self.window_size]
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)