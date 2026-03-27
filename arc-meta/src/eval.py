import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from datetime import datetime
import copy

from src.arc_dataset import ARCTaskDataset
from src.arc_dataloader import get_arc_loader
from src.model import ARCModel   # match your training


# =========================
# Metrics
# =========================
def pixel_accuracy(pred, target):
    pred = pred.argmax(dim=1)
    target = target.squeeze(1)
    return (pred == target).float().mean().item()


def task_accuracy(pred, target):
    pred = pred.argmax(dim=1)
    target = target.squeeze(1)
    return (pred == target).all(dim=(1, 2)).float().mean().item()


# =========================
# SAFE TTA (FIXED)
# =========================
def safe_tta(model, sx, sy, qx, steps=10, lr=5e-4):
    local_model = copy.deepcopy(model)
    local_model.train()

    params = list(local_model.decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    # detach encoder features (CRITICAL FIX)
    with torch.no_grad():
        sx_feat = local_model.encoder(sx)
        sy_feat = local_model.encoder(sy)

    for _ in range(steps):
        optimizer.zero_grad()

        task_tokens = local_model.task_encoder(sx_feat, sy_feat)
        pred = local_model.decoder(sx_feat, task_tokens)

        loss = F.cross_entropy(pred, sy.squeeze(1).long())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

    local_model.eval()
    with torch.no_grad():
        sx_feat = local_model.encoder(sx)
        sy_feat = local_model.encoder(sy)
        task_tokens = local_model.task_encoder(sx_feat, sy_feat)

        qx_feat = local_model.encoder(qx)
        return local_model.decoder(qx_feat, task_tokens)


# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/evaluation")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"eval_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")

    dataset = ARCTaskDataset(args.data_dir)
    loader = get_arc_loader(dataset, batch_size=args.batch_size)

    model = ARCModel().to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    total_pixel_acc = 0
    total_task_acc = 0
    count = 0

    logging.info("Starting evaluation...")

    for batch in tqdm(loader):

        if batch["query_y"] is None:
            continue

        support_x = batch["support_x"].to(device)
        support_y = batch["support_y"].to(device)
        query_x = batch["query_x"].to(device)
        query_y = batch["query_y"].to(device)

        support_mask = batch["support_mask"]
        query_mask = batch["query_mask"]

        B = support_x.shape[0]

        for b in range(B):

            # ✅ SAFE slicing (NO boolean indexing)
            s_len = int(support_mask[b].sum().item())
            q_len = int(query_mask[b].sum().item())

            sx = support_x[b, :s_len].long()
            sy = support_y[b, :s_len].long()
            qx = query_x[b, :q_len].long()
            qy = query_y[b, :q_len].long()

            if sx.numel() == 0 or qx.numel() == 0:
                continue

            # =========================
            # SAFE TTA
            # =========================
            pred = safe_tta(model, sx, sy, qx)

            # =========================
            # CRITICAL: crop output
            # =========================
            h, w = qy.shape[-2:]
            pred = pred[:, :, :h, :w]

            # =========================
            # Metrics
            # =========================
            p_acc = pixel_accuracy(pred, qy)
            t_acc = task_accuracy(pred, qy)

            total_pixel_acc += p_acc
            total_task_acc += t_acc
            count += 1

            logging.info(f"Task {count}: Pixel={p_acc:.4f}, Task={t_acc:.4f}")

    if count > 0:
        logging.info("=" * 50)
        logging.info(f"Pixel Accuracy: {total_pixel_acc / count:.4f}")
        logging.info(f"Task Accuracy:  {total_task_acc / count:.4f}")
        logging.info("=" * 50)
    else:
        logging.info("No tasks evaluated.")