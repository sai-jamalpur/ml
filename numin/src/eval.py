import torch
import numpy as np
import pandas as pd
import argparse
import os
import sys

# Force Python to check the current folder for local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from numin2 import Numin2API
except ImportError:
    print("[Error] The 'numin2' package is required for evaluation. Run: pip install numin2")
    sys.exit(1)

from model import SpatioTemporalGraphModel

class SpatioTemporalStockDataset(torch.utils.data.Dataset):
    """Reads directly from the pre-processed physical numpy arrays"""
    def __init__(self, data_folder, window_size=5):
        # Explicitly loading from the targeted folder (e.g., 'data/eval')
        self.features = np.load(os.path.join(data_folder, "ohlcv.npy"))
        self.returns = np.load(os.path.join(data_folder, "returns.npy"))
        self.window_size = window_size
        self.num_samples = len(self.features) - window_size
        
        # Extract topology dimensions dynamically (Time, Nodes, Features)
        self.num_nodes = self.features.shape[1]
        self.num_features = self.features.shape[2]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_window = self.features[idx : idx + self.window_size] 
        x_window = np.transpose(x_window, (1, 0, 2))
        y_target = self.returns[idx + self.window_size]
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)


def allocate_proportional_capital(signals):
    """
    Converts raw return signals into a strict (Time, N+1) portfolio weight matrix.
    The final column represents the Cash position.
    """
    time_steps, num_nodes = signals.shape
    positions = np.zeros((time_steps, num_nodes + 1))
    
    for t in range(time_steps):
        sig_t = signals[t]
        abs_sum = np.sum(np.abs(sig_t))
        
        if abs_sum > 1e-8:
            positions[t, :-1] = sig_t / abs_sum
            positions[t, -1] = 0.0 # Fully invested, 0% Cash
        else:
            positions[t, :-1] = 0.0
            positions[t, -1] = 1.0 # Flat signal, 100% Cash
            
    return positions


def run_evaluation(data_dir, weights_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Executing Out-of-Sample Backtest on: {device} ---")

    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Model weights '{weights_file}' not found. Run train.py first.")

    # 1. Target the Evaluation Folder strictly
    eval_dir = os.path.join(data_dir, "eval")
    edge_index_path = os.path.join(data_dir, "edge_index.pt")
    
    if not os.path.exists(eval_dir) or not os.path.exists(edge_index_path):
        raise FileNotFoundError(f"Processed eval data not found in '{eval_dir}'. Run prepare_data.py first.")

    print(f"Loading Out-of-Sample OHLCV and Returns from: {eval_dir}")
    edge_index = torch.load(edge_index_path, weights_only=True).to(device)
    
    # Instantiate dataset using ONLY the eval folder
    test_dataset = SpatioTemporalStockDataset(eval_dir, window_size=5)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 2. Reconstruct Model
    model = SpatioTemporalGraphModel(
        num_nodes=test_dataset.num_nodes,
        input_dim=test_dataset.num_features,
        temporal_hidden=128, 
        spatial_hidden=64,
        gat_heads=4
    ).to(device)

    model.load_state_dict(torch.load(weights_file, map_location=device, weights_only=True))
    model.eval()

    # 3. Inference Loop (Using eval OHLCV to predict eval Returns)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x, edge_index)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 4. Generate Position Matrices for Numin2 API
    print("Calculating Proportional Capital Allocations...")
    model_positions = allocate_proportional_capital(all_preds)
    ideal_positions = allocate_proportional_capital(all_targets)
    delta_positions = ideal_positions - model_positions

    # 5. Execute Backtests
    print(f"Routing {len(model_positions)} evaluation days through Numin2API...")
    api = Numin2API()
    
    try:
        model_results = api.backtest_positions(model_positions, all_targets)
        ideal_results = api.backtest_positions(ideal_positions, all_targets)
        delta_results = api.backtest_positions(delta_positions, all_targets)
        
        print("\n" + "="*65)
        print(f"{'METRIC':<20} | {'ST-GAT MODEL':<12} | {'IDEAL (MAX)':<12} | {'DELTA (ERROR)':<12}")
        print("-" * 65)
        
        m_tot = model_results.get('total_profit', 0)
        i_tot = ideal_results.get('total_profit', 0)
        d_tot = delta_results.get('total_profit', 0)
        
        m_shp = model_results.get('sharpe_ratio', 0)
        i_shp = ideal_results.get('sharpe_ratio', 0)
        d_shp = delta_results.get('sharpe_ratio', 0)
        
        print(f"{'Total PnL':<20} | {m_tot:>11.4f} | {i_tot:>11.4f} | {d_tot:>11.4f}")
        print(f"{'Sharpe Ratio':<20} | {m_shp:>11.4f} | {i_shp:>11.4f} | {d_shp:>11.4f}")
        print("="*65)
        
        # =====================================================================
        # 6. CSV EXPORT
        # =====================================================================
        # The API returns summary metrics. We calculate the daily timeline natively 
        # for the CSV by taking the dot product of weights (excluding cash) and actual returns.
        
        model_daily = np.sum(model_positions[:, :-1] * all_targets, axis=1)
        ideal_daily = np.sum(ideal_positions[:, :-1] * all_targets, axis=1)
        delta_daily = np.sum(delta_positions[:, :-1] * all_targets, axis=1)
        
        time_len = len(all_targets)
        timeline_df = pd.DataFrame({
            'Eval_Trading_Day': np.arange(time_len),
            'Model_Daily_PnL': model_daily,
            'Model_Cumulative_PnL': np.cumsum(model_daily),
            'Ideal_Daily_PnL': ideal_daily,
            'Ideal_Cumulative_PnL': np.cumsum(ideal_daily),
            'Delta_Daily_PnL': delta_daily,
            'Delta_Cumulative_PnL': np.cumsum(delta_daily)
        })
        
        csv_filename = "eval_backtest_timeline.csv"
        timeline_df.to_csv(csv_filename, index=False)
        print(f"\n[Export] Detailed day-by-day PnL timeline saved to '{csv_filename}'")
            
    except Exception as e:
        print(f"\n[API Execution Failed]: {e}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Now defaults strictly to the 'data' directory containing the 'eval' folder
    parser.add_argument('--data_dir', type=str, default='data', help="Base path to processed data folders")
    parser.add_argument('--weights', type=str, default='best_st_gat_model.pth')
    args = parser.parse_args()
    
    run_evaluation(args.data_dir, args.weights)