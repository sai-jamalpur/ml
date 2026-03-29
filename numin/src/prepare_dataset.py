import pandas as pd
import numpy as np
import torch
import os
import argparse

def prepare_and_split_dataset(ohlcv_path, returns_path, output_dir="data", threshold=0.35, train_ratio=0.8):
    print(f"--- Starting Data Preparation & Splitting ---")
    
    # 1. Parse Returns Data
    print(f"Loading Returns: {returns_path}")
    ret_df = pd.read_csv(returns_path, index_col=0)
    ret_tickers = list(ret_df.columns)
    num_time_steps = len(ret_df)
    
    # 2. Parse OHLCV Data
    print(f"Loading OHLCV: {ohlcv_path}")
    with open(ohlcv_path, 'r') as f:
        lines = f.readlines()
        
    features_line = lines[0].strip('\n').split(',')
    tickers_line = lines[1].strip('\n').split(',')
    
    ticker_to_cols = {}
    for col_idx in range(1, min(len(features_line), len(tickers_line))):
        feat = features_line[col_idx].strip()
        tick = tickers_line[col_idx].strip()
        if tick and feat:
            if tick not in ticker_to_cols:
                ticker_to_cols[tick] = {}
            ticker_to_cols[tick][feat] = col_idx
            
    expected_features = ['open', 'high', 'low', 'close', 'volume']
    
    # Intersection of valid tickers
    tickers = [t for t in ret_tickers if t in ticker_to_cols and all(f in ticker_to_cols[t] for f in expected_features)]
    num_nodes = len(tickers)
    num_features = len(expected_features)
    
    print(f"Aligned {num_nodes} tickers perfectly.")
    
    returns_tensor = ret_df[tickers].values 
    data_lines = [line for line in lines[3:] if line.strip()] 
    actual_time_steps = min(num_time_steps, len(data_lines))
    returns_tensor = returns_tensor[:actual_time_steps]
    
    ohlcv_tensor = np.zeros((actual_time_steps, num_nodes, num_features))
    
    for t in range(actual_time_steps):
        parts = data_lines[t].strip('\n').split(',')
        for node_idx, ticker in enumerate(tickers):
            for feat_idx, feat in enumerate(expected_features):
                col_idx = ticker_to_cols[ticker][feat]
                try:
                    val = float(parts[col_idx]) if col_idx < len(parts) and parts[col_idx] else 0.0
                except ValueError:
                    val = 0.0
                ohlcv_tensor[t, node_idx, feat_idx] = val

    # ==========================================================
    # 3. TEMPORAL SPLIT (80% Train, 20% Eval)
    # ==========================================================
    split_idx = int(actual_time_steps * train_ratio)
    
    train_ohlcv = ohlcv_tensor[:split_idx]
    train_returns = returns_tensor[:split_idx]
    
    eval_ohlcv = ohlcv_tensor[split_idx:]
    eval_returns = returns_tensor[split_idx:]
    
    print(f"\nTemporal Split Executed (Ratio: {train_ratio}):")
    print(f" - Train samples: {len(train_ohlcv)} days")
    print(f" - Eval samples : {len(eval_ohlcv)} days")

    # ==========================================================
    # 4. LEAKAGE-FREE NORMALIZATION & GRAPH GENERATION
    # ==========================================================
    print("\nExecuting Anti-Leakage Protocols...")
    
    # A) Z-Score strictly on Training Data
    train_mean = np.mean(train_ohlcv, axis=0, keepdims=True)
    train_std = np.std(train_ohlcv, axis=0, keepdims=True) + 1e-8
    
    train_ohlcv_norm = (train_ohlcv - train_mean) / train_std
    # Apply the TRAIN mean/std to the EVAL set to represent true out-of-sample scaling
    eval_ohlcv_norm = (eval_ohlcv - train_mean) / train_std 
    
    # B) Graph Topology strictly on Training Returns
    corr_matrix = np.corrcoef(train_returns, rowvar=False)
    source_nodes, target_nodes = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and abs(corr_matrix[i, j]) > threshold:
                source_nodes.append(i)
                target_nodes.append(j)
    
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    print(f"Sparse Graph Generated: {edge_index.shape[1]} Edges (using only Train correlations).")

    # ==========================================================
    # 5. PHYSICAL FOLDER CREATION & SAVING
    # ==========================================================
    train_dir = os.path.join(output_dir, "train")
    eval_dir = os.path.join(output_dir, "eval")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Save Numpy Arrays (.npy is much faster to load than CSV)
    np.save(os.path.join(train_dir, "ohlcv.npy"), train_ohlcv_norm)
    np.save(os.path.join(train_dir, "returns.npy"), train_returns)
    
    np.save(os.path.join(eval_dir, "ohlcv.npy"), eval_ohlcv_norm)
    np.save(os.path.join(eval_dir, "returns.npy"), eval_returns)
    
    # Save Graph Topology
    torch.save(edge_index, os.path.join(output_dir, "edge_index.pt"))
    
    print(f"\n[Success] Data successfully saved to '{output_dir}/'")
    print(f"├─ {output_dir}/train/ohlcv.npy")
    print(f"├─ {output_dir}/train/returns.npy")
    print(f"├─ {output_dir}/eval/ohlcv.npy")
    print(f"├─ {output_dir}/eval/returns.npy")
    print(f"└─ {output_dir}/edge_index.pt")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and split ST-GAT dataset")
    parser.add_argument('--ohlcv', type=str, default='consolidated_daily_ohlcv.csv')
    parser.add_argument('--returns', type=str, default='consolidated_daily_returns.csv')
    parser.add_argument('--out_dir', type=str, default='data')
    parser.add_argument('--threshold', type=float, default=0.35)
    
    args = parser.parse_args()
    prepare_and_split_dataset(args.ohlcv, args.returns, args.out_dir, args.threshold)