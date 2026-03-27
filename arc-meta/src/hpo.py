import torch
import torch.nn.functional as F
import copy


# =========================
# Train one epoch
# =========================
def train_one_epoch(model, loader, optimizer, device, entropy_w):
    model.train()
    total_loss = 0

    for batch in loader:
        if batch["query_y"] is None:
            continue

        sx = batch["support_x"].to(device)
        sy = batch["support_y"].to(device)
        qx = batch["query_x"].to(device)
        qy = batch["query_y"].to(device)

        sm = batch["support_mask"].to(device)
        qm = batch["query_mask"].to(device)

        B, S, _, H, W = sx.shape
        Q = qx.shape[1]

        optimizer.zero_grad()

        sx_f = model.encoder(sx.view(B*S, 1, H, W)).view(B, S, -1, H, W)
        sy_f = model.encoder(sy.view(B*S, 1, H, W)).view(B, S, -1, H, W)
        qx_f = model.encoder(qx.view(B*Q, 1, H, W)).view(B, Q, -1, H, W)

        losses = []

        for b in range(B):
            s_len = int(sm[b].sum())
            q_len = int(qm[b].sum())

            sx_b = sx_f[b, :s_len]
            sy_b = sy_f[b, :s_len]
            qx_b = qx_f[b, :q_len]
            qy_b = qy[b, :q_len]

            if sx_b.numel() == 0 or qx_b.numel() == 0:
                continue

            tokens = model.task_encoder(sx_b, sy_b)
            pred = model.decoder(qx_b, tokens)

            loss = F.cross_entropy(pred, qy_b.squeeze(1).long())

            probs = torch.softmax(pred, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

            loss = loss + entropy_w * entropy
            losses.append(loss)

        if len(losses) == 0:
            continue

        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss


# =========================
# Safe TTA
# =========================
def safe_tta(model, sx, sy, qx, steps, lr):
    local_model = copy.deepcopy(model)
    local_model.train()

    params = list(local_model.decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

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
def evaluate(model, loader, device, tta_steps, tta_lr):
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
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

            sx_b = sx[b, :s_len].long()
            sy_b = sy[b, :s_len].long()
            qx_b = qx[b, :q_len].long()
            qy_b = qy[b, :q_len].long()

            pred = safe_tta(model, sx_b, sy_b, qx_b, tta_steps, tta_lr)

            h, w = qy_b.shape[-2:]
            pred = pred[:, :, :h, :w]

            pred = pred.argmax(dim=1)
            target = qy_b.squeeze(1)

            correct += (pred == target).float().sum().item()
            total += target.numel()

    return correct / total if total > 0 else 0


# =========================
# Main: Hyperparameter Optimization
# =========================
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    import argparse
    from src.model import ARCModel
    from src.arc_dataset import ARCTaskDataset
    from src.arc_dataloader import get_arc_loader
    import logging
    from datetime import datetime

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

    # Arguments
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization")
    parser.add_argument("--entropy_w", type=float, default=0.01, help="Entropy weight")
    parser.add_argument("--tta_steps", type=int, default=5, help="TTA steps")
    parser.add_argument("--tta_lr", type=float, default=1e-3, help="TTA learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load data
    logging.info("Loading datasets...")
    train_dataset = ARCTaskDataset("data/training")
    eval_dataset = ARCTaskDataset("data/evaluation")
    
    train_loader = get_arc_loader(train_dataset, batch_size=args.batch_size)
    eval_loader = get_arc_loader(eval_dataset, batch_size=1)
    
    logging.info(f"Loaded {len(train_dataset)} training tasks and {len(eval_dataset)} eval tasks")

    # Create model
    logging.info("Creating model...")
    model = ARCModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    logging.info(f"Starting HPO with entropy_w={args.entropy_w}, tta_steps={args.tta_steps}, tta_lr={args.tta_lr}")
    for epoch in range(args.epochs):
        logging.info(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # Train
        total_loss = train_one_epoch(model, train_loader, optimizer, device, args.entropy_w)
        logging.info(f"Train loss: {total_loss:.4f}")
        
        # Evaluate
        accuracy = evaluate(model, eval_loader, device, args.tta_steps, args.tta_lr)
        logging.info(f"Eval accuracy: {accuracy:.4f}")

    logging.info("HPO completed!")