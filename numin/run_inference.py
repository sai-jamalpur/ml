"""
Inference-only comparison: model.pth vs model_new.pth
Portfolio allocation via softmax of base-model scores (no TransformerAllocator training needed).
"""

import os, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_PATH   = "data/dataset_train.csv"
TEST_PATH    = "data/dataset_test.csv"
SEQ_LEN      = 30
NUM_FEATURES = 18
FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "ret_5", "ret_30", "rel_ret_5", "rank_ret_5",
    "vol_30", "risk_adj_ret", "vol_z", "pv_signal",
    "dist_high", "z_price", "residual", "market_ret_5", "corr_market_30",
]

print(f"Device: {DEVICE}")

# ── Model Architecture ─────────────────────────────────────────────────────────

class ResidualConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(ch)
        self.drop  = nn.Dropout(0.2)

    def forward(self, x):
        res = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.gelu(out + res)


class MultiWindowFeatureExtractor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.short_conv = nn.Conv1d(in_dim, out_dim, 3, padding=1)
        self.mid_conv   = nn.Conv1d(in_dim, out_dim, 5, padding=2)
        self.long_conv  = nn.Conv1d(in_dim, out_dim, 7, padding=3)
        self.batch_norm = nn.BatchNorm1d(out_dim * 3)
        self.do         = nn.Dropout(0.3)

    def forward(self, x):
        return self.do(F.gelu(self.batch_norm(
            torch.cat([self.short_conv(x), self.mid_conv(x), self.long_conv(x)], dim=1)
        )))


class StockPricePredictor(nn.Module):
    def __init__(self, num_features=NUM_FEATURES, hidden_size=128):
        super().__init__()
        self.initial_norm  = nn.BatchNorm1d(num_features)
        self.feature_block = MultiWindowFeatureExtractor(num_features, hidden_size)
        self.compression   = nn.Conv1d(hidden_size * 3, hidden_size, 1)
        self.res_blocks    = nn.Sequential(
            ResidualConvBlock(hidden_size),
            ResidualConvBlock(hidden_size),
            ResidualConvBlock(hidden_size),
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.GELU(), nn.BatchNorm1d(128), nn.Dropout(0.4),
            nn.Linear(128, 64),          nn.GELU(),                       nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.initial_norm(x)
        x = self.feature_block(x)
        x = F.gelu(self.compression(x))
        x = self.res_blocks(x)
        pooled = (x.mean(dim=-1) + x.max(dim=-1).values) / 2
        return self.output_head(pooled).squeeze(-1)


def load_model(path):
    m = StockPricePredictor().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    m.eval()
    return m


# ── Load Models ────────────────────────────────────────────────────────────────

print("\nLoading models...")
model_old = load_model("model.pth")
model_new = load_model("model_new.pth")
print("  model.pth     ✓")
print("  model_new.pth ✓")


# ── Build Portfolio Data ───────────────────────────────────────────────────────

print("\nLoading portfolio data from", TRAIN_PATH)
df = pd.read_csv(TRAIN_PATH)
df.sort_values(["timestamp", "ticker"], inplace=True)

tickers    = sorted(df["ticker"].unique())
num_stocks = len(tickers)
dates      = sorted(df["timestamp"].unique())
num_dates  = len(dates)

X_all = np.zeros((num_dates, num_stocks, NUM_FEATURES))
Y_all = np.zeros((num_dates, num_stocks))

t2i = {t: i for i, t in enumerate(tickers)}
d2i = {d: i for i, d in enumerate(dates)}

for _, row in df.iterrows():
    di = d2i[row["timestamp"]]
    si = t2i[row["ticker"]]
    X_all[di, si, :] = row[FEATURE_COLS].values
    if "return" in row:
        Y_all[di, si] = row["return"]

X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
Y_all = np.nan_to_num(Y_all, nan=0.0, posinf=0.0, neginf=0.0)

# Shift labels: index t predicts return at t+1
Y_all[:-1] = Y_all[1:]
Y_all[-1]  = 0.0

valid_samples = num_dates - SEQ_LEN
train_size    = int(valid_samples * 0.8)
test_indices  = list(range(train_size, valid_samples))

print(f"  {num_stocks} stocks | {num_dates} dates | {valid_samples} valid samples")
print(f"  Train: {train_size} | Test: {len(test_indices)}")


# ── Inference: Score all stocks per test date ──────────────────────────────────

def run_inference(model, label):
    print(f"\nRunning inference: {label}")
    dates_out, port_rets, ew_rets = [], [], []

    with torch.no_grad():
        for idx in sorted(test_indices):
            # seq_x shape: (num_stocks, SEQ_LEN, NUM_FEATURES)
            seq_x = X_all[idx : idx + SEQ_LEN]             # (T, S, F)
            seq_x = np.transpose(seq_x, (1, 0, 2))         # (S, T, F)
            target_y = Y_all[idx + SEQ_LEN - 1]            # (S,)

            x_tensor = torch.tensor(seq_x, dtype=torch.float32).to(DEVICE)  # (S, T, F)
            scores   = model(x_tensor)                      # (S,)
            scores   = torch.nan_to_num(scores, 0.0)
            weights  = torch.softmax(scores, dim=0).cpu().numpy()

            date = dates[idx + SEQ_LEN - 1]
            dates_out.append(date)
            port_rets.append(float(np.dot(weights, target_y)))
            ew_rets.append(float(target_y.mean()))

    df_out = pd.DataFrame({
        "date":         pd.to_datetime(dates_out),
        label:          port_rets,
        "equal_weight": ew_rets,
    }).set_index("date").sort_index()

    print(f"  Done — {len(df_out)} test dates")
    return df_out


results_old = run_inference(model_old, "model.pth")
results_new = run_inference(model_new, "model_new.pth")

results = (
    results_old[["model.pth"]]
    .join(results_new[["model_new.pth"]])
    .join(results_old[["equal_weight"]])
)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(series, ann_factor=252):
    r        = np.asarray(series, dtype=float)
    cum_pnl  = np.cumsum(r)                         # simple cumulative PnL
    total    = float(cum_pnl[-1])
    ann_r    = float(r.mean() * ann_factor)
    ann_v    = float(r.std(ddof=1) * np.sqrt(ann_factor))
    sharpe   = ann_r / ann_v if ann_v > 0 else float("nan")
    roll_max = np.maximum.accumulate(cum_pnl)
    max_dd   = float((cum_pnl - roll_max).min())
    win      = float((r > 0).mean())
    avg_d    = float(r.mean())
    return {
        "Total PnL (Σ ret)": f"{total:+.4f}",
        "Ann. Return":        f"{ann_r:+.4f}",
        "Ann. Volatility":    f"{ann_v:.4f}",
        "Sharpe Ratio":       f"{sharpe:.3f}",
        "Max Drawdown":       f"{max_dd:.4f}",
        "Win Rate":           f"{win * 100:.1f}%",
        "Avg Daily Return":   f"{avg_d:+.6f}",
    }

summary = pd.DataFrame({
    "model.pth":     compute_metrics(results["model.pth"]),
    "model_new.pth": compute_metrics(results["model_new.pth"]),
    "Equal Weight":  compute_metrics(results["equal_weight"]),
})

print("\n" + "═" * 66)
print("   OUT-OF-SAMPLE PERFORMANCE SUMMARY")
print("═" * 66)
print(summary.to_string())
print("═" * 66)


# ── Directional Accuracy ───────────────────────────────────────────────────────

def dir_acc(model):
    correct, total = 0, 0
    with torch.no_grad():
        for idx in sorted(test_indices):
            seq_x    = np.transpose(X_all[idx : idx + SEQ_LEN], (1, 0, 2))
            target_y = Y_all[idx + SEQ_LEN - 1]
            x_t      = torch.tensor(seq_x, dtype=torch.float32).to(DEVICE)
            scores   = model(x_t).cpu().numpy()
            mask     = target_y != 0
            correct += int((np.sign(scores[mask]) == np.sign(target_y[mask])).sum())
            total   += int(mask.sum())
    return correct / total * 100 if total > 0 else 0.0

print("\nComputing directional accuracy on test set...")
da_old = dir_acc(model_old)
da_new = dir_acc(model_new)
print(f"\n{'Model':<20} {'Dir Acc':>10}")
print("-" * 32)
print(f"{'model.pth':<20} {da_old:>9.2f}%")
print(f"{'model_new.pth':<20} {da_new:>9.2f}%")
print(f"{'Δ (new − old)':<20} {da_new - da_old:>+9.2f}%")


# ── Plots ──────────────────────────────────────────────────────────────────────

print("\nGenerating comparison plots...")
# Cumulative PnL via cumsum (returns can go < -1, so cumprod is not valid)
cum = results.cumsum()

fig = plt.figure(figsize=(16, 14))
fig.suptitle(
    "model.pth vs model_new.pth — Out-of-Sample Portfolio Comparison",
    fontsize=14, fontweight="bold", y=0.998,
)
gs = gridspec.GridSpec(3, 2, hspace=0.48, wspace=0.32)

# 1. Cumulative PnL
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(cum.index, cum["model.pth"],     lw=2.2, label="model.pth")
ax1.plot(cum.index, cum["model_new.pth"], lw=2.2, label="model_new.pth")
ax1.plot(cum.index, cum["equal_weight"],  lw=1.6, ls="--", alpha=0.8, label="Equal Weight")
ax1.axhline(0, color="grey", lw=0.8, ls=":")
ax1.set_title("Cumulative Out-of-Sample PnL (sum of daily returns)", fontsize=11)
ax1.set_xlabel("Date"); ax1.set_ylabel("Cumulative Return (sum)")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# 2. Daily Returns
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(results.index, results["model.pth"],     alpha=0.75, lw=1.0, label="model.pth")
ax2.plot(results.index, results["model_new.pth"], alpha=0.75, lw=1.0, label="model_new.pth")
ax2.axhline(0, color="grey", lw=0.8, ls="--")
ax2.set_title("Daily Portfolio Returns", fontsize=11)
ax2.set_xlabel("Date"); ax2.set_ylabel("Daily Return")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# 3. Rolling 30-Day Sharpe
ax3 = fig.add_subplot(gs[1, 1])
for col in ["model.pth", "model_new.pth"]:
    r  = results[col]
    rs = r.rolling(30).mean() / (r.rolling(30).std() + 1e-8) * np.sqrt(252)
    ax3.plot(rs.index, rs, lw=1.5, label=col)
ax3.axhline(0, color="grey", lw=0.8, ls="--")
ax3.set_title("Rolling 30-Day Annualised Sharpe Ratio", fontsize=11)
ax3.set_xlabel("Date"); ax3.set_ylabel("Sharpe Ratio")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

# 4. Return Distribution
ax4 = fig.add_subplot(gs[2, 0])
colors = {"model.pth": "steelblue", "model_new.pth": "darkorange", "equal_weight": "grey"}
for col, c in colors.items():
    ax4.hist(results[col], bins=40, alpha=0.45, density=True, label=col, color=c)
ax4.set_title("Daily Return Distribution", fontsize=11)
ax4.set_xlabel("Daily Return"); ax4.set_ylabel("Density")
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

# 5. Drawdown (on cumulative PnL)
ax5 = fig.add_subplot(gs[2, 1])
for col, lbl in [("model.pth", "model.pth"), ("model_new.pth", "model_new.pth"), ("equal_weight", "Equal Weight")]:
    cpnl = cum[col]
    dd   = cpnl - cpnl.cummax()
    ax5.fill_between(dd.index, dd, 0, alpha=0.30, label=lbl)
    ax5.plot(dd.index, dd, lw=0.8, alpha=0.7)
ax5.set_title("Portfolio Drawdown (absolute, on cumulative PnL)", fontsize=11)
ax5.set_xlabel("Date"); ax5.set_ylabel("Drawdown")
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

out_path = "model_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
print("\nDone.")
