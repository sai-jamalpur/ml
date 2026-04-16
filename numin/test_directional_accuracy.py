import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class StockDataset(Dataset):
    def __init__(self, data_path, ticker=None, seq_len=30, scaler=None):
        self.seq_len = seq_len
        self.ticker = ticker
        
        df = pd.read_csv(data_path)
        if ticker:
            df = df[df['ticker'] == ticker].copy()
            
        df.sort_values(['ticker', 'timestamp'], inplace=True)
        
        self.feature_cols = [
            'open', 'high', 'low', 'close', 'volume', 
            'ret_5', 'ret_30', 'rel_ret_5', 'rank_ret_5', 
            'vol_30', 'risk_adj_ret', 'vol_z', 'pv_signal', 
            'dist_high', 'z_price', 'residual', 'market_ret_5', 'corr_market_30'
        ]
        
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)
        
        df['target'] = df.groupby('ticker')['return'].shift(-1)
        df.dropna(subset=['target'], inplace=True)
        
        features = df[self.feature_cols].values
        targets = df['target'].values
        tickers_list = df['ticker'].values
        
        if scaler is None:
            self.mean = np.nanmean(features, axis=0)
            self.std = np.nanstd(features, axis=0)
            self.std = np.where(self.std == 0, 1.0, self.std)  
        else:
            self.mean, self.std = scaler
            
        features = (features - self.mean) / (self.std + 1e-8)
        
        self.valid_indices = []
        for i in range(len(features) - self.seq_len):
            if tickers_list[i] == tickers_list[i + self.seq_len - 1]:
                self.valid_indices.append(i)
                
        self.x_data = features
        self.y_data = targets
            
    def __len__(self):
        return len(self.valid_indices)
        
    def __getitem__(self, i):
        idx = self.valid_indices[i]
        x = self.x_data[idx:idx + self.seq_len]
        y = self.y_data[idx + self.seq_len - 1] 
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.gelu(out)

class MultiWindowFeatureExtractor(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.short_conv = nn.Conv1d(input_dims, output_dims, kernel_size=3, padding=1)
        self.mid_conv = nn.Conv1d(input_dims, output_dims, kernel_size=5, padding=2)
        self.long_conv = nn.Conv1d(input_dims, output_dims, kernel_size=7, padding=3)
        self.batch_norm = nn.BatchNorm1d(output_dims * 3)
        self.dropout_layer = nn.Dropout(0.3)

    def forward(self, x):
        out_short = self.short_conv(x)
        out_mid   = self.mid_conv(x)
        out_long  = self.long_conv(x)
        combined = torch.cat([out_short, out_mid, out_long], dim=1)
        return self.dropout_layer(F.gelu(self.batch_norm(combined)))

class StockPricePredictor(nn.Module):
    def __init__(self, num_features, hidden_size=128): 
        super().__init__()
        self.initial_norm = nn.BatchNorm1d(num_features)
        self.feature_extractor = MultiWindowFeatureExtractor(num_features, hidden_size)
        
        self.res_blocks = nn.Sequential(
            ResidualConvBlock(hidden_size * 3),
            ResidualConvBlock(hidden_size * 3)
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size * 3, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.2, 
            bidirectional=True
        )
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.initial_norm(x)
        x = self.feature_extractor(x)
        x = self.res_blocks(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        last_hidden_state = attn_out[:, -1, :]
        out = self.regressor(last_hidden_state)
        return out.squeeze(1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load train dataset to get the scaler
    print("Loading train dataset to extract scaler values...")
    train_dataset = StockDataset('data/dataset_train.csv', ticker=None, seq_len=30)
    scaler = (train_dataset.mean, train_dataset.std)

    # 2. Load test dataset using train scaler
    print("Loading test dataset...")
    test_dataset = StockDataset('data/dataset_test.csv', ticker=None, seq_len=30, scaler=scaler)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # 3. Initialize and load the model
    print("Loading model...")
    num_features = len(train_dataset.feature_cols)
    model = StockPricePredictor(num_features).to(device)
    model.load_state_dict(torch.load('model_new.pth', map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    print("Evaluating directional accuracy on test data...")
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    # Mask to ignore examples where true return is exactly 0
    mask = all_targets != 0.0
    filtered_preds = all_preds[mask]
    filtered_targets = all_targets[mask]

    correct_direction = (np.sign(filtered_preds) == np.sign(filtered_targets)).astype(float)
    dir_acc = correct_direction.mean() * 100

    print(f"\n==============================================")
    print(f"  Test Directional Accuracy   : {dir_acc:.2f}%")
    print(f"  Total Valid Evaluated Samples: {len(filtered_targets)}")
    print(f"==============================================\n")

if __name__ == '__main__':
    main()
