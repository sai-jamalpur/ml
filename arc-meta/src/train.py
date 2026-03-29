#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os
import argparse
import copy

from src.arc_dataset import ARCTaskDataset
from src.arc_dataloader import get_arc_loader
from src.model import ARCFewShotHRM 

parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default='')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--entropy_w", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=5000)
parser.add_argument("--t_steps", type=int, default=4)
parser.add_argument("--max_segments", type=int, default=8)
parser.add_argument("--tta_steps", type=int, default=20)
parser.add_argument("--tta_lr", type=float, default=0.001)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
os.makedirs("models", exist_ok=True)

train_dataset = ARCTaskDataset("data/training")
eval_dataset = ARCTaskDataset("data/evaluation")
train_loader = get_arc_loader(train_dataset, batch_size=args.batch_size)
eval_loader = get_arc_loader(eval_dataset, batch_size=1)

model = ARCFewShotHRM(dim=128, T_steps=args.t_steps, max_segments=args.max_segments).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scaler = torch.amp.GradScaler("cuda")

start_epoch, best_loss = 0, float("inf")
if args.resume and os.path.exists(args.resume):
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)

def safe_tta(model, sx, sy, qx, steps, lr):
    local_model = copy.deepcopy(model)
    local_model.train()
    opt = torch.optim.Adam(local_model.parameters(), lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        segment_logits = local_model(sx, sy, sx, return_all_segments=True)
        loss = sum(F.cross_entropy(seg, sy.squeeze(1).long()) for seg in segment_logits) / len(segment_logits)
        loss.backward()
        opt.step()

    local_model.eval()
    with torch.no_grad():
        return local_model(sx, sy, qx)

# def evaluate(model):
#     model.eval()
#     correct, total = 0, 0
#     for batch in eval_loader:
#         if batch["query_y"] is None: continue
#         sx, sy = batch["support_x"].to(device), batch["support_y"].to(device)
#         qx, qy = batch["query_x"].to(device), batch["query_y"].to(device)
        
#         for b in range(sx.shape[0]):
#             s_len, q_len = int(batch["support_mask"][b].sum()), int(batch["query_mask"][b].sum())
#             pred = safe_tta(model, sx[b, :s_len], sy[b, :s_len], qx[b, :q_len], args.tta_steps, args.tta_lr)
#             correct += (pred.argmax(dim=1) == qy[b, :q_len].squeeze(1)).float().sum().item()
#             total += qy[b, :q_len].numel()
#     return correct / total if total > 0 else 0
def evaluate(model):
    model.eval()
    tasks_correct = 0
    total_tasks = 0
    pixels_correct = 0
    total_pixels = 0
    
    # tqdm provides a visual progress bar in your terminal
    for batch in tqdm(eval_loader, desc="Evaluating Tasks"):
        if batch["query_y"] is None: continue
        
        sx, sy = batch["support_x"].to(device), batch["support_y"].to(device)
        qx, qy = batch["query_x"].to(device), batch["query_y"].to(device)
        
        for b in range(sx.shape[0]):
            s_len = int(batch["support_mask"][b].sum())
            q_len = int(batch["query_mask"][b].sum())
            
            # 1. Run TTA (Test-Time Augmentation/Fine-tuning)
            pred_logits = safe_tta(model, sx[b, :s_len], sy[b, :s_len], qx[b, :q_len], args.tta_steps, args.tta_lr)
            
            # 2. Convert to discrete color predictions
            pred_grid = pred_logits.argmax(dim=1) # Shape: (Batch, H, W)
            target_grid = qy[b, :q_len].squeeze(1).long() # Shape: (Batch, H, W)
            
            # --- PIXEL ACCURACY CALCULATION ---
            # Count how many individual pixels match
            correct_mask = (pred_grid == target_grid)
            pixels_correct += correct_mask.sum().item()
            total_pixels += target_grid.numel()
            
            # --- TASK ACCURACY CALCULATION ---
            # A task is only correct if EVERY pixel in the grid is correct
            is_task_perfect = torch.equal(pred_grid, target_grid)
            if is_task_perfect:
                tasks_correct += 1
            total_tasks += 1
            
    # Calculate Final Metrics
    pixel_acc = pixels_correct / total_pixels if total_pixels > 0 else 0
    task_acc = tasks_correct / total_tasks if total_tasks > 0 else 0
    
    logging.info(f"--- Eval Results ---")
    logging.info(f"Pixel Accuracy: {pixel_acc:.2%} ({pixels_correct}/{total_pixels})")
    logging.info(f"Task Accuracy:  {task_acc:.2%} ({tasks_correct}/{total_tasks})")
    
    return task_acc # Usually we return Task Acc as the primary metric for ARC

for epoch in range(start_epoch, args.epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

    for batch in pbar:
        if batch["query_y"] is None: continue
        
        support_x, support_y = batch["support_x"].to(device), batch["support_y"].to(device)
        query_x, query_y = batch["query_x"].to(device), batch["query_y"].to(device)
        actual_batch_size = batch["actual_batch_size"]

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            losses = []
            for b in range(actual_batch_size):
                s_len, q_len = int(batch["support_mask"][b].sum()), int(batch["query_mask"][b].sum())
                sx_b, sy_b = support_x[b, :s_len], support_y[b, :s_len]
                qx_b, qy_b = query_x[b, :q_len], query_y[b, :q_len]

                segment_logits = model(sx_b, sy_b, qx_b, return_all_segments=True)
                
                # Deep Supervision mechanism
                task_loss = sum(F.cross_entropy(seg, qy_b.squeeze(1).long(), ignore_index=10) for seg in segment_logits)
                task_loss /= len(segment_logits)

                probs = torch.softmax(segment_logits[-1], dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                losses.append(task_loss + args.entropy_w * entropy)

            if not losses: continue
            
            # Dynamic Gradient Scaling to neutralize exact bucketing variance
            batch_loss = torch.stack(losses).mean()
            scaled_loss = batch_loss * (actual_batch_size / args.batch_size)

        scaler.scale(scaled_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += batch_loss.item()
        pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

    avg_loss = total_loss / len(train_loader)
    logging.info(f"Train loss: {avg_loss:.4f}")
    if (epoch + 1) % 400 == 0:
        eval_acc = evaluate(model)
        logging.info(f"Eval accuracy: {eval_acc:.4f}")
    # logging.info(f"Eval accuracy: {evaluate(model):.4f}")

    # 1. Continuous State Overwrite
    torch.save({
        "epoch": epoch + 1, 
        "model_state_dict": model.state_dict(), 
        "optimizer_state_dict": optimizer.state_dict()
    }, "models/hrm_latest.pt")
    
    # 2. Performance-Conditional Preservation
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            "epoch": epoch + 1, 
            "model_state_dict": model.state_dict(), 
            "optimizer_state_dict": optimizer.state_dict()
        }, "models/hrm_best.pt")

    # 3. Temporal-Conditional Preservation (20-Epoch Checkpoint)
    if (epoch + 1) % 400 == 0:
        checkpoint_path = f"models/hrm_{epoch + 1}.pt"
        torch.save({
            "epoch": epoch + 1, 
            "model_state_dict": model.state_dict(), 
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }, checkpoint_path)
        logging.info(f"State artifact preserved: {checkpoint_path}")