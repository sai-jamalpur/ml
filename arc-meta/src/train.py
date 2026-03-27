#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from datetime import datetime
import os
import argparse
import copy

from src.arc_dataset import ARCTaskDataset
from src.arc_dataloader import get_arc_loader
from src.model import ARCModel


# =========================
# Args
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default='models/arc_best.pt')
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--entropy_w", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=2000)

# TTA params
parser.add_argument("--tta_steps", type=int, default=10)
parser.add_argument("--tta_lr", type=float, default= 0.001)

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
train_dataset = ARCTaskDataset("data/training")
eval_dataset = ARCTaskDataset("data/evaluation")

train_loader = get_arc_loader(train_dataset, batch_size=args.batch_size)
eval_loader = get_arc_loader(eval_dataset, batch_size=1)

logging.info(f"Loaded {len(train_dataset)} training tasks")
logging.info(f"Loaded {len(eval_dataset)} eval tasks")


# =========================
# Model
# =========================
model = ARCModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scaler = torch.amp.GradScaler("cuda")

start_epoch = 0
best_loss = float("inf")

# =========================
# Resume
# =========================
if args.resume and os.path.exists(args.resume):
    logging.info(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint.get("epoch", 0)
    best_loss = checkpoint.get("best_loss", float("inf"))

    logging.info(f"Resumed from epoch {start_epoch}")


# =========================
# TTA
# =========================
def safe_tta(model, sx, sy, qx, steps, lr):
    local_model = copy.deepcopy(model)
    local_model.train()

    optimizer = torch.optim.Adam(local_model.decoder.parameters(), lr=lr)

    with torch.no_grad():
        sx_f = local_model.encoder(sx)
        sy_f = local_model.encoder(sy)

    for _ in range(steps):
        optimizer.zero_grad()
        tokens = local_model.task_encoder(sx_f, sy_f)
        pred = local_model.decoder(sx_f, tokens)
        loss = F.cross_entropy(pred, sy.squeeze(1).long())
        loss.backward()
        optimizer.step()

    local_model.eval()
    with torch.no_grad():
        tokens = local_model.task_encoder(sx_f, sy_f)
        qx_f = local_model.encoder(qx)
        return local_model.decoder(qx_f, tokens)


# =========================
# Evaluate
# =========================
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

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

            pred = safe_tta(
                model,
                sx_b,
                sy_b,
                qx_b,
                args.tta_steps,
                args.tta_lr
            )

            pred = pred.argmax(dim=1)
            target = qy_b.squeeze(1)

            correct += (pred == target).float().sum().item()
            total += target.numel()

    return correct / total if total > 0 else 0


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

            sx = model.encoder(support_x.view(B * S, 1, H, W))
            sy = model.encoder(support_y.view(B * S, 1, H, W))
            qx = model.encoder(query_x.view(B * Q, 1, H, W))

            sx = sx.view(B, S, *sx.shape[1:])
            sy = sy.view(B, S, *sy.shape[1:])
            qx = qx.view(B, Q, *qx.shape[1:])

            losses = []

            for b in range(B):
                s_len = int(support_mask[b].sum().item())
                q_len = int(query_mask[b].sum().item())

                sx_b = sx[b, :s_len]
                sy_b = sy[b, :s_len]
                qx_b = qx[b, :q_len]
                qy_b = query_y[b, :q_len]

                if sx_b.numel() == 0 or qx_b.numel() == 0:
                    continue

                tokens = model.task_encoder(sx_b, sy_b)
                pred = model.decoder(qx_b, tokens)

                loss = F.cross_entropy(pred, qy_b.squeeze(1).long())

                probs = torch.softmax(pred, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

                loss = loss + args.entropy_w * entropy
                loss = torch.clamp(loss, max=10.0)

                losses.append(loss)

            if len(losses) == 0:
                continue

            total_batch_loss = torch.stack(losses).mean()

        if torch.isnan(total_batch_loss):
            continue

        scaler.scale(total_batch_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += total_batch_loss.item()
        pbar.set_postfix({"loss": f"{total_batch_loss.item():.4f}"})


    # =========================
    # Epoch Summary
    # =========================
    avg_loss = total_loss / len(train_loader)
    logging.info(f"Train loss: {avg_loss:.4f}")

    # =========================
    # Eval with TTA
    # =========================
    acc = evaluate(model)
    logging.info(f"Eval accuracy: {acc:.4f}")


    # =========================
    # Save
    # =========================
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss
    }, "models/arc_latest.pt")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss
        }, "models/arc_best.pt")
        logging.info("Saved BEST model")