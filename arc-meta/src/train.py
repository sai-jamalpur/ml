import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from datetime import datetime
import os
import argparse

from src.arc_dataset import ARCTaskDataset
from src.arc_dataloader import get_arc_loader
from src.model import ARCModel


# =========================
# Args
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default='models/arc_epoch_200.pt', help="Checkpoint path")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size (optimal: 4)")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--entropy_w", type=float, default=0.001, help="Entropy regularization weight (optimal: 0.001)")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
args = parser.parse_args()


# =========================
# Setup
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

os.makedirs("models", exist_ok=True)


# =========================
# Data
# =========================
dataset = ARCTaskDataset("data/training")
loader = get_arc_loader(dataset, batch_size=args.batch_size)

logging.info(f"Loaded {len(dataset)} training tasks.")


# =========================
# Model
# =========================
model = ARCModel().to(device)

# Parameter logging
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

logging.info(f"Total parameters: {total_params:,}")
logging.info(f"Trainable parameters: {trainable_params:,}")

logging.info("Module breakdown:")
for name, module in model.named_children():
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    logging.info(f"{name:15s} | total: {total:,} | trainable: {trainable:,}")


# =========================
# Optimizer + AMP
# =========================
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scaler = torch.amp.GradScaler("cuda")

logging.info(f"Hyperparameters (from HPO):")
logging.info(f"  Learning rate: {args.lr}")
logging.info(f"  Batch size: {args.batch_size}")
logging.info(f"  Entropy weight: {args.entropy_w}")
logging.info(f"  Max epochs: {args.epochs}")


# =========================
# Resume
# =========================
start_epoch = 0
best_loss = float("inf")

if args.resume and os.path.exists(args.resume):
    logging.info(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("best_loss", float("inf"))
    else:
        model.load_state_dict(checkpoint)
        logging.warning("Loaded weights only (no optimizer state).")

    logging.info(f"Resumed from epoch {start_epoch}")
else:
    logging.info("Starting from scratch.")


# =========================
# Training
# =========================
num_epochs = args.epochs

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in pbar:

        if batch["query_y"] is None:
            continue

        support_x = batch["support_x"].to(device)
        support_y = batch["support_y"].to(device)
        query_x = batch["query_x"].to(device)
        query_y = batch["query_y"].to(device)

        support_mask = batch["support_mask"].to(device)
        query_mask = batch["query_mask"].to(device)

        B, S, _, H, W = support_x.shape
        Q = query_x.shape[1]

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):

            # =========================
            # Batched encoder
            # =========================
            sx = model.encoder(support_x.view(B * S, 1, H, W))
            sy = model.encoder(support_y.view(B * S, 1, H, W))
            qx = model.encoder(query_x.view(B * Q, 1, H, W))

            sx = sx.view(B, S, *sx.shape[1:])
            sy = sy.view(B, S, *sy.shape[1:])
            qx = qx.view(B, Q, *qx.shape[1:])

            losses = []

            # =========================
            # Task loop
            # =========================
            for b in range(B):
                s_len = int(support_mask[b].sum().item())
                q_len = int(query_mask[b].sum().item())

                sx_b = sx[b, :s_len]
                sy_b = sy[b, :s_len]
                qx_b = qx[b, :q_len]
                qy_b = query_y[b, :q_len]

                if sx_b.numel() == 0 or qx_b.numel() == 0:
                    continue

                task_tokens = model.task_encoder(sx_b, sy_b)
                pred = model.decoder(qx_b, task_tokens)

                loss = F.cross_entropy(pred, qy_b.squeeze(1).long())

                # entropy regularization
                probs = torch.softmax(pred, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

                loss = loss + args.entropy_w * entropy

                loss = torch.clamp(loss, max=10.0)
                losses.append(loss)

            if len(losses) == 0:
                continue

            total_batch_loss = torch.stack(losses).mean()

        if torch.isnan(total_batch_loss):
            logging.warning("NaN loss, skipping batch")
            continue

        scaler.scale(total_batch_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += total_batch_loss.item()
        pbar.set_postfix({"loss": f"{total_batch_loss.item():.4f}"})


    # =========================
    # Epoch summary
    # =========================
    avg_loss = total_loss / len(loader)
    logging.info(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")


    # =========================
    # Save latest
    # =========================
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss
    }, "models/arc_latest.pt")


    # =========================
    # Save every 50 epochs
    # =========================
    if (epoch + 1) % 50 == 0:
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss
        }, f"models/arc_epoch_{epoch+1}.pt")

        logging.info(f"Saved checkpoint at epoch {epoch+1}")


    # =========================
    # Save best
    # =========================
    if avg_loss < best_loss:
        best_loss = avg_loss

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss
        }, "models/arc_best.pt")

        logging.info("Saved BEST model")