import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import sys
import csv
import logging
from datetime import datetime

import matplotlib.pyplot as plt

# Force Python to check the current folder for local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from model import SpatioTemporalGraphModel

class SpatioTemporalStockDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, window_size=5):
        self.features = np.load(os.path.join(data_folder, "ohlcv.npy"))
        self.returns = np.load(os.path.join(data_folder, "returns.npy"))
        self.window_size = window_size
        self.num_samples = len(self.features) - window_size
        
        self.num_nodes = self.features.shape[1]
        self.num_features = self.features.shape[2]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_window = self.features[idx : idx + self.window_size] 
        x_window = np.transpose(x_window, (1, 0, 2))
        y_target = self.returns[idx + self.window_size]
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)

class CrossSectionalHybridLoss(nn.Module):
    def __init__(self, alpha=0.3):
        super(CrossSectionalHybridLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        mse_loss = self.mse(preds, targets)
        
        preds_mean = preds.mean(dim=1, keepdim=True)
        targets_mean = targets.mean(dim=1, keepdim=True)
        preds_centered = preds - preds_mean
        targets_centered = targets - targets_mean
        
        covariance = (preds_centered * targets_centered).sum(dim=1)
        preds_std = torch.sqrt((preds_centered ** 2).sum(dim=1) + 1e-8)
        targets_std = torch.sqrt((targets_centered ** 2).sum(dim=1) + 1e-8)
        
        correlation = covariance / (preds_std * targets_std)
        ic_loss = -correlation.mean()
        
        return (self.alpha * mse_loss) + ((1 - self.alpha) * ic_loss), correlation.mean()


def setup_logger(log_path):
    logger = logging.getLogger("numin.train")
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


def save_training_artifacts(history, output_dir):
    csv_path = os.path.join(output_dir, "training_history.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["epoch", "train_loss", "train_ic", "val_loss", "val_ic", "lr"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["train_ic"][i],
                history["val_loss"][i],
                history["val_ic"][i],
                history["lr"][i],
            ])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_ic"], label="Train IC", linewidth=2)
    axes[1].plot(history["epoch"], history["val_ic"], label="Val IC", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IC")
    axes[1].set_title("Training and Validation Information Coefficient")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return csv_path, plot_path


def train_pipeline(data_dir, models_dir, resume, epochs=50, logs_dir="logs"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime("train_%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(logs_dir, run_id)
    os.makedirs(run_output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(run_output_dir, "train.log"))

    logger.info("--- Executing ST-GAT Training on: %s ---", device)

    # 1. Directory Setup
    train_dir = os.path.join(data_dir, "train")
    eval_dir = os.path.join(data_dir, "eval")
    edge_index_path = os.path.join(data_dir, "edge_index.pt")
    
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(train_dir) or not os.path.exists(edge_index_path):
        raise FileNotFoundError(f"Processed data not found in '{data_dir}'. Run prepare_data.py first.")

    # 2. Load Data
    logger.info("Loading data from %s...", data_dir)
    edge_index = torch.load(edge_index_path, weights_only=True).to(device)
    train_dataset = SpatioTemporalStockDataset(train_dir, window_size=5)
    val_dataset = SpatioTemporalStockDataset(eval_dir, window_size=5)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 3. Initialize Architecture
    model = SpatioTemporalGraphModel(
        num_nodes=train_dataset.num_nodes, 
        input_dim=train_dataset.num_features,
        temporal_hidden=128, 
        spatial_hidden=64,
        gat_heads=4
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params | total: %s | trainable: %s", f"{total_params:,}", f"{trainable_params:,}")

    criterion = CrossSectionalHybridLoss(alpha=0.3) 
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # 4. Resume State Management
    start_epoch = 0
    best_val_ic = -float('inf')
    latest_ckpt_path = os.path.join(models_dir, "latest_model.pth")
    history = {
        "epoch": [],
        "train_loss": [],
        "train_ic": [],
        "val_loss": [],
        "val_ic": [],
        "lr": [],
    }

    if resume:
        if os.path.exists(latest_ckpt_path):
            logger.info("[*] Resuming from checkpoint: %s", latest_ckpt_path)
            checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            best_val_ic = checkpoint.get('best_val_ic', -float('inf'))
            logger.info("[*] Resumed at Epoch %s | Previous Best IC: %.4f", start_epoch, best_val_ic)
        else:
            logger.warning("[!] No checkpoint found at %s. Starting from scratch.", latest_ckpt_path)

    # 5. Execution Loop
    logger.info("Starting Training Loop...")
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss, epoch_ic = 0.0, 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x, edge_index)
            loss, batch_ic = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_ic += batch_ic.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_ic = epoch_ic / len(train_loader)
        
        # Validation Auditing
        model.eval()
        val_loss, val_ic = 0.0, 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_preds = model(val_x, edge_index)
                v_loss, v_ic = criterion(val_preds, val_y)
                val_loss += v_loss.item()
                val_ic += v_ic.item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_ic = val_ic / len(val_loader)
        scheduler.step(avg_val_ic)
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["train_ic"].append(avg_train_ic)
        history["val_loss"].append(avg_val_loss)
        history["val_ic"].append(avg_val_ic)
        history["lr"].append(current_lr)
        
        # -------------------------------------------------------------
        # CHECKPOINTING LOGIC
        # -------------------------------------------------------------
        checkpoint_msg = ""
        
        # Standardize the state dictionary to save
        state_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_ic': best_val_ic
        }

        # 1. ALWAYS save the latest model (overwrites previous 'latest')
        torch.save(state_dict, latest_ckpt_path)

        # 2. Save a historical milestone every 10 epochs
        if (epoch + 1) % 10 == 0:
            milestone_path = os.path.join(models_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(state_dict, milestone_path)
            checkpoint_msg += " -> [Milestone Saved]"

        # 3. Save the Best Performing Model (Weights only for easy eval loading)
        if avg_val_ic > best_val_ic:
            best_val_ic = avg_val_ic
            # Update the state dict with the new best IC
            state_dict['best_val_ic'] = best_val_ic 
            best_path = os.path.join(models_dir, "best_st_gat_model.pth")
            # We save only the model weights here so eval.py remains completely unchanged
            torch.save(model.state_dict(), best_path) 
            checkpoint_msg += " -> [New Best Saved]"
        
        logger.info(
            "Epoch [%02d/%d] | Train Loss: %.6f | Val Loss: %.6f | Train IC: %.4f | Val IC: %.4f | LR: %.6f%s",
            epoch + 1,
            epochs,
            avg_train_loss,
            avg_val_loss,
            avg_train_ic,
            avg_val_ic,
            current_lr,
            checkpoint_msg,
        )

    history_csv, history_plot = save_training_artifacts(history, run_output_dir)

    logger.info("Training Complete. Best Validation IC: %.4f", best_val_ic)
    logger.info("Checkpoints saved to: %s/", os.path.abspath(models_dir))
    logger.info("Training history CSV: %s", os.path.abspath(history_csv))
    logger.info("Training curves plot: %s", os.path.abspath(history_plot))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help="Path to the processed data folder")
    parser.add_argument('--models_dir', type=str, default='models', help="Directory to save model checkpoints")
    parser.add_argument('--logs_dir', type=str, default='logs/training', help="Directory to save logs and plots")
    parser.add_argument('--epochs', type=int, default=50, help="Total number of epochs to train")
    parser.add_argument('--resume', action='store_true', help="Flag to resume training from latest_model.pth")
    args = parser.parse_args()
    
    train_pipeline(args.data_dir, args.models_dir, args.resume, args.epochs, args.logs_dir)