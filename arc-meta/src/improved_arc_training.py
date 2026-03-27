
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from datetime import datetime
import os
import argparse
import copy

# Assuming the improved ARCModel is in a file named improved_arc_model.py
# If running in a single notebook cell, the model classes should be defined above this code.
from src.arc_dataset import ARCTaskDataset
from src.arc_dataloader import get_arc_loader
from improved_arc_model import ARCModel # Ensure this matches your model file name

# =========================
# Setup & Logging
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

# =========================
# Hyperparameters (Notebook style)
# =========================
class Args:
    resume = 'models/arc_best.pt'
    batch_size = 4
    lr = 0.0001
    entropy_w = 0.001
    epochs = 2000
    tta_steps = 10
    tta_lr = 0.001

args = Args()

# =========================
# Data Loading
# =========================
train_dataset = ARCTaskDataset("data/training")
eval_dataset = ARCTaskDataset("data/evaluation")

train_loader = get_arc_loader(train_dataset, batch_size=args.batch_size)
eval_loader = get_arc_loader(eval_dataset, batch_size=1)

logging.info(f"Loaded {len(train_dataset)} training tasks")
logging.info(f"Loaded {len(eval_dataset)} eval tasks")

# =========================
# Model Initialization
# =========================
model = ARCModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scaler = torch.amp.GradScaler("cuda")

start_epoch = 0
best_loss = float("inf")

# =========================
# Resume Checkpoint
# =========================
if args.resume and os.path.exists(args.resume):
    logging.info(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    
    # Handle potential state_dict key mismatches if architecture changed
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("best_loss", float("inf"))
        logging.info(f"Resumed from epoch {start_epoch}")
    except RuntimeError as e:
        logging.warning(f"Could not fully resume state_dict due to architecture changes: {e}")
        logging.info("Starting with fresh weights or partial load.")

# =========================
# Evaluation with TTA
# =========================
def evaluate(model):
    model.eval()
    pixel_correct = 0
    pixel_total = 0
    task_correct = 0
    task_total = 0

    for batch in eval_loader:
        if batch["query_y"] is None:
            continue

        sx = batch["support_x"].to(device)
        sy = batch["support_y"].to(device)
        qx = batch["query_x"].to(device)
        qy = batch["query_y"].to(device)

        sm = batch["support_mask"]
        qm = batch["query_mask"]

        B = sx.shape[0]

        for b in range(B):
            s_len = int(sm[b].sum())
            q_len = int(qm[b].sum())

            sx_b = sx[b, :s_len]
            sy_b = sy[b, :s_len]
            qx_b = qx[b, :q_len]
            qy_b = qy[b, :q_len]

            # Use the model's built-in adaptation method
            # We adapt only the decoder for faster evaluation
            pred_logits = model.forward_with_adaptation(
                sx_b, sy_b, qx_b, 
                steps=args.tta_steps, 
                lr=args.tta_lr,
                adapt_encoder=False
            )

            pred = pred_logits.argmax(dim=1)
            target = qy_b.squeeze(1).long()

            # Pixel accuracy
            pixel_correct += (pred == target).float().sum().item()
            pixel_total += target.numel()
            
            # Task accuracy (perfect match)
            if torch.equal(pred, target):
                task_correct += 1
            task_total += 1

    pixel_acc = pixel_correct / pixel_total if pixel_total > 0 else 0
    task_acc = task_correct / task_total if task_total > 0 else 0
    return pixel_acc, task_acc

# =========================
# Training Loop
# =========================
for epoch in range(start_epoch, args.epochs):
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

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
            # Efficiently encode all support and query grids in the batch
            # support_x: (B, S, 1, H, W) -> (B*S, 1, H, W)
            sx_feats = model.encoder(support_x.view(B * S, 1, H, W))
            sy_feats = model.encoder(support_y.view(B * S, 1, H, W))
            qx_feats = model.encoder(query_x.view(B * Q, 1, H, W))

            # Reshape back to batch dimensions
            sx_feats = sx_feats.view(B, S, -1, H, W)
            sy_feats = sy_feats.view(B, S, -1, H, W)
            qx_feats = qx_feats.view(B, Q, -1, H, W)

            batch_losses = []

            for b in range(B):
                s_len = int(support_mask[b].sum().item())
                q_len = int(query_mask[b].sum().item())

                if s_len == 0 or q_len == 0:
                    continue

                # Extract features for the current task in the batch
                sx_b = sx_feats[b, :s_len]
                sy_b = sy_feats[b, :s_len]
                qx_b = qx_feats[b, :q_len]
                qy_b = query_y[b, :q_len].squeeze(1).long()

                # Rule inference via Task Encoder
                task_tokens = model.task_encoder(sx_b, sy_b)
                
                # Predict query outputs
                pred = model.decoder(qx_b, task_tokens)

                # Cross-entropy loss
                ce_loss = F.cross_entropy(pred, qy_b)

                # Entropy regularization for confidence
                probs = torch.softmax(pred, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

                loss = ce_loss + args.entropy_w * entropy
                loss = torch.clamp(loss, max=10.0) # Gradient clipping for stability

                batch_losses.append(loss)

            if not batch_losses:
                continue

            total_batch_loss = torch.stack(batch_losses).mean()

        if torch.isnan(total_batch_loss):
            logging.warning("NaN loss detected, skipping batch")
            continue

        scaler.scale(total_batch_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += total_batch_loss.item()
        pbar.set_postfix({"loss": f"{total_batch_loss.item():.4f}"})

    # =========================
    # Epoch Summary & Evaluation
    # =========================
    avg_loss = total_loss / len(train_loader)
    logging.info(f"Epoch {epoch+1} - Train loss: {avg_loss:.4f}")

    # Periodic evaluation (e.g., every 5 epochs for speed)
    if (epoch + 1) % 5 == 0:
        pixel_acc, task_acc = evaluate(model)
        logging.info(f"Epoch {epoch+1} - Eval Pixel Acc: {pixel_acc:.4f}, Task Acc: {task_acc:.4f}")

    # =========================
    # Checkpoint Saving
    # =========================
    save_dict = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss
    }
    torch.save(save_dict, "models/arc_latest.pt")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(save_dict, "models/arc_best.pt")
        logging.info(f"New best model saved with loss {best_loss:.4f}")
