import torch
import numpy as np
import pandas as pd
import argparse
import os
import sys
import json
import logging
from datetime import datetime

import matplotlib.pyplot as plt

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


def setup_logger(log_path):
    logger = logging.getLogger("numin.eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_eval_plot(timeline_df, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(timeline_df["Eval_Trading_Day"], timeline_df["Model_Cumulative_PnL"], label="Model", linewidth=2)
    axes[0].plot(timeline_df["Eval_Trading_Day"], timeline_df["Ideal_Cumulative_PnL"], label="Ideal", linewidth=2)
    axes[0].plot(timeline_df["Eval_Trading_Day"], timeline_df["Delta_Cumulative_PnL"], label="Delta", linewidth=2)
    axes[0].set_ylabel("Cumulative PnL")
    axes[0].set_title("Cumulative Backtest PnL")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(timeline_df["Eval_Trading_Day"], timeline_df["Model_Daily_PnL"], label="Model Daily", linewidth=1.5)
    axes[1].plot(timeline_df["Eval_Trading_Day"], timeline_df["Ideal_Daily_PnL"], label="Ideal Daily", linewidth=1.5)
    axes[1].plot(timeline_df["Eval_Trading_Day"], timeline_df["Delta_Daily_PnL"], label="Delta Daily", linewidth=1.5)
    axes[1].set_xlabel("Evaluation Trading Day")
    axes[1].set_ylabel("Daily PnL")
    axes[1].set_title("Daily Backtest PnL")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_predictions_vs_actual_plot(pred_daily, actual_daily, output_path):
    days = np.arange(len(pred_daily))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(days, pred_daily, label="Predicted Mean Return", linewidth=2)
    ax.plot(days, actual_daily, label="Actual Mean Return", linewidth=2)
    ax.set_xlabel("Evaluation Trading Day")
    ax.set_ylabel("Mean Return")
    ax.set_title("Predicted vs Actual Daily Mean Returns")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_eval_returns_histogram(all_preds, all_targets, output_path):
    pred_flat = all_preds.reshape(-1)
    target_flat = all_targets.reshape(-1)

    fig, ax = plt.subplots(figsize=(11, 6))
    bins = 120
    ax.hist(target_flat, bins=bins, alpha=0.55, density=True, label="Ground Truth Returns")
    ax.hist(pred_flat, bins=bins, alpha=0.55, density=True, label="Predicted Returns")
    ax.set_title("Eval Set Return Distribution: Prediction vs Ground Truth")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def load_aligned_tickers_from_raw_data():
    project_root = os.path.dirname(current_dir)
    returns_csv = os.path.join(project_root, "data", "consolidated_daily_returns.csv")
    ohlcv_csv = os.path.join(project_root, "data", "consolidated_daily_ohlcv.csv")

    if not os.path.exists(returns_csv) or not os.path.exists(ohlcv_csv):
        return None

    ret_df = pd.read_csv(returns_csv, index_col=0)
    ret_tickers = list(ret_df.columns)

    with open(ohlcv_csv, "r", encoding="utf-8") as fp:
        lines = fp.readlines()

    if len(lines) < 2:
        return None

    features_line = lines[0].strip("\n").split(",")
    tickers_line = lines[1].strip("\n").split(",")

    ticker_to_cols = {}
    for col_idx in range(1, min(len(features_line), len(tickers_line))):
        feat = features_line[col_idx].strip()
        tick = tickers_line[col_idx].strip()
        if tick and feat:
            if tick not in ticker_to_cols:
                ticker_to_cols[tick] = {}
            ticker_to_cols[tick][feat] = col_idx

    expected_features = ["open", "high", "low", "close", "volume"]
    tickers = [
        t for t in ret_tickers
        if t in ticker_to_cols and all(f in ticker_to_cols[t] for f in expected_features)
    ]

    return tickers


def save_day_stockwise_plot(tickers, pred_day, target_day, day_idx, output_path):
    x = np.arange(len(tickers))
    width = 0.42

    fig_w = max(14, len(tickers) * 0.22)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    ax.bar(x - width / 2, target_day, width=width, label="Ground Truth", alpha=0.85)
    ax.bar(x + width / 2, pred_day, width=width, label="Prediction", alpha=0.85)
    ax.set_title(f"Stock-wise Prediction vs Ground Truth | Eval Day {day_idx}")
    ax.set_xlabel("Stocks")
    ax.set_ylabel("Return")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=75, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_evaluation(data_dir, weights_file, logs_dir="logs/evaluations", plot_day=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(logs_dir, run_id)
    os.makedirs(run_output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(run_output_dir, "eval.log"))

    logger.info("--- Executing Out-of-Sample Backtest on: %s ---", device)

    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Model weights '{weights_file}' not found. Run train.py first.")

    # 1. Target the Evaluation Folder strictly
    eval_dir = os.path.join(data_dir, "eval")
    edge_index_path = os.path.join(data_dir, "edge_index.pt")
    
    if not os.path.exists(eval_dir) or not os.path.exists(edge_index_path):
        raise FileNotFoundError(f"Processed eval data not found in '{eval_dir}'. Run prepare_data.py first.")

    logger.info("Loading Out-of-Sample OHLCV and Returns from: %s", eval_dir)
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
    logger.info("Calculating Proportional Capital Allocations...")
    model_positions = allocate_proportional_capital(all_preds)
    ideal_positions = allocate_proportional_capital(all_targets)
    delta_positions = ideal_positions - model_positions

    # 5. Execute Backtests
    logger.info("Routing %d evaluation days through Numin2API...", len(model_positions))
    api = Numin2API()
    
    try:
        model_results = api.backtest_positions(model_positions, all_targets)
        ideal_results = api.backtest_positions(ideal_positions, all_targets)
        delta_results = api.backtest_positions(delta_positions, all_targets)
        
        logger.info("=" * 65)
        logger.info("%-20s | %-12s | %-12s | %-12s", "METRIC", "ST-GAT MODEL", "IDEAL (MAX)", "DELTA (ERROR)")
        logger.info("-" * 65)
        
        m_tot = model_results.get('total_profit', 0)
        i_tot = ideal_results.get('total_profit', 0)
        d_tot = delta_results.get('total_profit', 0)
        
        m_shp = model_results.get('sharpe_ratio', 0)
        i_shp = ideal_results.get('sharpe_ratio', 0)
        d_shp = delta_results.get('sharpe_ratio', 0)
        
        logger.info("%-20s | %11.4f | %11.4f | %11.4f", "Total PnL", m_tot, i_tot, d_tot)
        logger.info("%-20s | %11.4f | %11.4f | %11.4f", "Sharpe Ratio", m_shp, i_shp, d_shp)
        logger.info("=" * 65)
        
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
        
        # Compare model predictions against actual returns at day granularity.
        pred_daily_mean = np.mean(all_preds, axis=1)
        actual_daily_mean = np.mean(all_targets, axis=1)
        daily_errors = pred_daily_mean - actual_daily_mean
        daily_mse = float(np.mean(daily_errors ** 2))
        daily_mae = float(np.mean(np.abs(daily_errors)))
        daily_rmse = float(np.sqrt(daily_mse))

        csv_filename = os.path.join(run_output_dir, "eval_backtest_timeline.csv")
        timeline_df.to_csv(csv_filename, index=False)

        pred_vs_actual_csv = os.path.join(run_output_dir, "predictions_vs_actual_day_returns.csv")
        pd.DataFrame({
            "Eval_Trading_Day": np.arange(time_len),
            "Predicted_Mean_Return": pred_daily_mean,
            "Actual_Mean_Return": actual_daily_mean,
            "Error": daily_errors,
            "Squared_Error": daily_errors ** 2,
        }).to_csv(pred_vs_actual_csv, index=False)

        plot_filename = os.path.join(run_output_dir, "eval_backtest_plot.png")
        save_eval_plot(timeline_df, plot_filename)

        pred_vs_actual_plot = os.path.join(run_output_dir, "predictions_vs_actual_day_returns.png")
        save_predictions_vs_actual_plot(pred_daily_mean, actual_daily_mean, pred_vs_actual_plot)

        returns_hist_plot = os.path.join(run_output_dir, "eval_returns_histogram_pred_vs_gt.png")
        save_eval_returns_histogram(all_preds, all_targets, returns_hist_plot)

        num_nodes = all_targets.shape[1]
        ticker_names = load_aligned_tickers_from_raw_data()
        if ticker_names is None or len(ticker_names) != num_nodes:
            ticker_names = [f"Stock_{i}" for i in range(num_nodes)]

        per_stock_actual_cum = np.sum(all_targets, axis=0)
        per_stock_pred_cum = np.sum(all_preds, axis=0)
        per_stock_mae = np.mean(np.abs(all_preds - all_targets), axis=0)
        per_stock_rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2, axis=0))
        per_stock_dir_acc = np.mean(np.sign(all_preds) == np.sign(all_targets), axis=0)

        per_stock_corr = []
        for i in range(num_nodes):
            pred_i = all_preds[:, i]
            target_i = all_targets[:, i]
            if np.std(pred_i) < 1e-12 or np.std(target_i) < 1e-12:
                per_stock_corr.append(0.0)
            else:
                per_stock_corr.append(float(np.corrcoef(pred_i, target_i)[0, 1]))
        per_stock_corr = np.array(per_stock_corr)

        top_k = min(12, num_nodes)
        top_idx = np.argsort(per_stock_actual_cum)[-top_k:][::-1]
        similarity_df = pd.DataFrame({
            "Ticker": [ticker_names[i] for i in top_idx],
            "Actual_Cumulative_Return": per_stock_actual_cum[top_idx],
            "Predicted_Cumulative_Return": per_stock_pred_cum[top_idx],
            "Correlation": per_stock_corr[top_idx],
            "MAE": per_stock_mae[top_idx],
            "RMSE": per_stock_rmse[top_idx],
            "Direction_Accuracy": per_stock_dir_acc[top_idx],
        })

        similarity_csv = os.path.join(run_output_dir, "top_stocks_similarity_metrics.csv")
        similarity_df.to_csv(similarity_csv, index=False)

        resolved_day = plot_day if plot_day >= 0 else (time_len - 1)
        if resolved_day < 0 or resolved_day >= time_len:
            raise ValueError(f"plot_day must be in range [0, {time_len - 1}], got {plot_day}")

        day_pred = all_preds[resolved_day]
        day_target = all_targets[resolved_day]
        day_abs_error = np.abs(day_pred - day_target)
        day_corr = 0.0
        if np.std(day_pred) > 1e-12 and np.std(day_target) > 1e-12:
            day_corr = float(np.corrcoef(day_pred, day_target)[0, 1])

        day_plot_path = os.path.join(run_output_dir, f"pred_vs_gt_stockwise_day_{resolved_day}.png")
        save_day_stockwise_plot(ticker_names, day_pred, day_target, resolved_day, day_plot_path)

        day_csv_path = os.path.join(run_output_dir, f"pred_vs_gt_stockwise_day_{resolved_day}.csv")
        pd.DataFrame({
            "Ticker": ticker_names,
            "Prediction": day_pred,
            "Ground_Truth": day_target,
            "Abs_Error": day_abs_error,
        }).to_csv(day_csv_path, index=False)

        metrics = {
            "model": {"total_profit": float(m_tot), "sharpe_ratio": float(m_shp)},
            "ideal": {"total_profit": float(i_tot), "sharpe_ratio": float(i_shp)},
            "delta": {"total_profit": float(d_tot), "sharpe_ratio": float(d_shp)},
            "prediction_daily_mean_loss": {
                "mse": daily_mse,
                "mae": daily_mae,
                "rmse": daily_rmse,
            },
            "num_eval_days": int(time_len),
            "weights_file": os.path.abspath(weights_file),
            "selected_day": {
                "day_index": int(resolved_day),
                "correlation": float(day_corr),
                "mae": float(np.mean(day_abs_error)),
                "plot": os.path.abspath(day_plot_path),
                "csv": os.path.abspath(day_csv_path),
            },
            "returns_histogram_plot": os.path.abspath(returns_hist_plot),
            "top_stock_similarity_csv": os.path.abspath(similarity_csv),
        }
        metrics_path = os.path.join(run_output_dir, "eval_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)

        logger.info("[Export] Timeline CSV: %s", os.path.abspath(csv_filename))
        logger.info("[Export] Backtest plot: %s", os.path.abspath(plot_filename))
        logger.info("[Export] Pred vs Actual CSV: %s", os.path.abspath(pred_vs_actual_csv))
        logger.info("[Export] Pred vs Actual plot: %s", os.path.abspath(pred_vs_actual_plot))
        logger.info("[Export] Eval returns histogram plot: %s", os.path.abspath(returns_hist_plot))
        logger.info("[Metrics] Daily Mean Return Loss | MSE: %.8f | MAE: %.8f | RMSE: %.8f", daily_mse, daily_mae, daily_rmse)
        logger.info("[Export] Selected day stock-wise CSV: %s", os.path.abspath(day_csv_path))
        logger.info("[Export] Selected day stock-wise plot: %s", os.path.abspath(day_plot_path))
        logger.info("[Metrics] Selected day %d | Corr: %.4f | MAE: %.6f", resolved_day, day_corr, float(np.mean(day_abs_error)))
        logger.info("[Export] Top stock similarity CSV: %s", os.path.abspath(similarity_csv))
        logger.info("[Metrics] Top-%d stock similarity (ranked by ground-truth cumulative return):", top_k)
        for row in similarity_df.itertuples(index=False):
            logger.info(
                "  %s | Corr: %.4f | MAE: %.4f | RMSE: %.4f | DirAcc: %.4f",
                row.Ticker,
                float(row.Correlation),
                float(row.MAE),
                float(row.RMSE),
                float(row.Direction_Accuracy),
            )
        logger.info("[Export] Metrics JSON: %s", os.path.abspath(metrics_path))
            
    except Exception as e:
        logger.exception("[API Execution Failed]: %s", e)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Now defaults strictly to the 'data' directory containing the 'eval' folder
    parser.add_argument('--data_dir', type=str, default='data', help="Base path to processed data folders")
    parser.add_argument('--weights', type=str, default='best_st_gat_model.pth')
    parser.add_argument('--logs_dir', type=str, default='logs/evaluations', help="Directory to save eval logs and plots")
    parser.add_argument('--plot_day', type=int, default=-1, help="Eval day index for stock-wise pred-vs-ground-truth plot (-1 for last day)")
    args = parser.parse_args()
    
    run_evaluation(args.data_dir, args.weights, args.logs_dir, args.plot_day)