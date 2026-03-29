import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class NiftyDatasetCreator:
    """
    Parses the flattened multi-index CSV format using Explicit Column Mapping 
    to guarantee alignment between Returns and OHLCV tensors, ignoring extra/missing columns.
    """
    def __init__(self, ohlcv_path, returns_path):
        import os
        if not os.path.exists(ohlcv_path) or not os.path.exists(returns_path):
            raise FileNotFoundError("CSV files not found. Ensure they are in the same directory.")
            
        self.ohlcv_path = ohlcv_path
        self.returns_path = returns_path
        
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
        
        # Build explicit column index map: ticker -> {feature: column_index}
        ticker_to_cols = {}
        for col_idx in range(1, min(len(features_line), len(tickers_line))):
            feat = features_line[col_idx].strip()
            tick = tickers_line[col_idx].strip()
            
            if tick and feat: # Skip empty strings or trailing commas
                if tick not in ticker_to_cols:
                    ticker_to_cols[tick] = {}
                ticker_to_cols[tick][feat] = col_idx
                
        # Define expected features
        expected_features = ['open', 'high', 'low', 'close', 'volume']
        
        # Structural Intersection: Keep only tickers that exist in BOTH files 
        # AND have all 5 features present in the OHLCV file.
        self.tickers = []
        for t in ret_tickers:
            if t in ticker_to_cols and all(f in ticker_to_cols[t] for f in expected_features):
                self.tickers.append(t)
                
        self.num_nodes = len(self.tickers)
        self.num_features = len(expected_features)
        
        print(f"Successfully aligned {self.num_nodes} tickers present in both datasets.")
        
        # 1. Rebuild Returns Tensor strictly with aligned tickers
        self.returns_tensor = ret_df[self.tickers].values  # Shape: (Time, Nodes)
        
        # 2. Extract OHLCV Data safely using the mapped indices
        # Skip the first 3 lines (headers + empty timestamp row)
        data_lines = [line for line in lines[3:] if line.strip()] 
        
        # Ensure time steps match exactly between both files
        actual_time_steps = min(self.num_time_steps, len(data_lines))
        self.returns_tensor = self.returns_tensor[:actual_time_steps]
        self.num_time_steps = actual_time_steps
        
        # Initialize 3D Orthogonal Tensor: (Time, Nodes, Features)
        self.ohlcv_tensor = np.zeros((self.num_time_steps, self.num_nodes, self.num_features))
        
        for t in range(self.num_time_steps):
            parts = data_lines[t].strip('\n').split(',')
            
            for node_idx, ticker in enumerate(self.tickers):
                for feat_idx, feat in enumerate(expected_features):
                    col_idx = ticker_to_cols[ticker][feat]
                    
                    # Safely parse float to prevent crashes on bad data lines
                    try:
                        val = float(parts[col_idx]) if col_idx < len(parts) and parts[col_idx] else 0.0
                    except ValueError:
                        val = 0.0
                        
                    self.ohlcv_tensor[t, node_idx, feat_idx] = val
                    
        print("Applying Temporal Z-Score Normalization...")
        # 3. Z-Score Normalization across the Temporal axis
        mean = np.mean(self.ohlcv_tensor, axis=0, keepdims=True)
        std = np.std(self.ohlcv_tensor, axis=0, keepdims=True) + 1e-8
        self.ohlcv_normalized = (self.ohlcv_tensor - mean) / std
        print("Data Loading Complete.")
        
    def get_correlation_graph(self, threshold=0.6):
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
    """
    Constructs sliding windows over the multivariate time series.
    Yields (X_window, Y_target) pairs for PyTorch modeling.
    """
    def __init__(self, features, returns, window_size=5):
        self.features = features
        self.returns = returns
        self.window_size = window_size
        self.num_samples = len(features) - window_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Extract temporal window for all nodes simultaneously
        x_window = self.features[idx : idx + self.window_size] 
        # Convert (Window, Nodes, Features) -> (Nodes, Window, Features)
        x_window = np.transpose(x_window, (1, 0, 2))
        
        # Target is the return at the timestep immediately following the window
        y_target = self.returns[idx + self.window_size]
        
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)


if __name__ == "__main__":
    # Define file paths
    OHLCV_FILE = 'consolidated_daily_ohlcv.csv'
    RETURNS_FILE = 'consolidated_daily_returns.csv'
    
    # 1. Instantiate the Creator
    creator = NiftyDatasetCreator(OHLCV_FILE, RETURNS_FILE)

    # 2. Extract Spatial Graph Topology
    # Adjust threshold to tune graph sparsity (0.6 is a standard starting point for equities)
    edge_index = creator.get_correlation_graph(threshold=0.6)

    # 3. Initialize PyTorch Dataset & DataLoader
    # window_size = 5 implies using the past 5 days of OHLCV to predict the 6th day's return
    dataset = SpatioTemporalStockDataset(
        features=creator.ohlcv_normalized, 
        returns=creator.returns_tensor, 
        window_size=5
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 4. Verification Printout
    print("\n--- Structural Verification ---")
    print(f"Total Tickers (Nodes): {creator.num_nodes}")
    print(f"Total Timesteps: {creator.num_time_steps}")
    print(f"Total Edges in Graph: {edge_index.shape[1]}")
    
    for batch_x, batch_y in dataloader:
        print("\nFirst Batch Shapes:")
        print(f"X (Features) Shape : {batch_x.shape} -> (Batch, Nodes, Window, Features)") 
        print(f"Y (Targets) Shape  : {batch_y.shape} -> (Batch, Nodes)")
        break
    
    print("\nDataset generation pipeline is ready for model ingestion.")