#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib import colors

from src.arc_dataset import ARCTaskDataset
from src.arc_dataloader import get_arc_loader
from src.model import ARCFewShotHRM

# =========================
# Visual Artifact Dependency
# =========================
ARC_HEX_PALETTE = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]
ARC_CMAP = colors.ListedColormap(ARC_HEX_PALETTE)
ARC_NORM = colors.Normalize(vmin=0, vmax=9)

# Generate a unique run ID for this evaluation session
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EVAL_DIR = f"eval_run_{RUN_ID}"
VISUAL_DIR = os.path.join(EVAL_DIR, "visuals")

def preserve_visual_state(qx, qy, pred, task_id):
    """
    Reconstructs the spatial mapping of the task and preserves it as a visual artifact.
    """
    os.makedirs(VISUAL_DIR, exist_ok=True)
    
    # Isolate spatial grids: [C, H, W] -> [H, W]
    input_grid = qx.squeeze().cpu().numpy()
    truth_grid = qy.squeeze().cpu().numpy()
    
    # Convert logits to class indices: [C, H, W] -> [H, W]
    pred_grid = pred.argmax(dim=1).squeeze().cpu().numpy()

    # Handle structural edge cases (ensure 2D)
    if input_grid.ndim > 2: input_grid = input_grid[-1]
    if truth_grid.ndim > 2: truth_grid = truth_grid[-1]
    if pred_grid.ndim > 2: pred_grid = pred_grid[-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(input_grid, cmap=ARC_CMAP, norm=ARC_NORM)
    axes[0].set_title(f"Input (Query X)\nTask Index: {task_id}", fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(truth_grid, cmap=ARC_CMAP, norm=ARC_NORM)
    axes[1].set_title("Ground Truth (Query Y)", fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(pred_grid, cmap=ARC_CMAP, norm=ARC_NORM)
    axes[2].set_title("Model Prediction", fontsize=10)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, f"task_{task_id:04d}.png"), dpi=150)
    plt.close(fig)

# =========================
# Metrics
# =========================
def get_metrics(pred, target):
    """
    Calculates pixel-wise and full-grid accuracy.
    pred: [Q_len, C, H, W], target: [Q_len, H, W]
    """
    pred_indices = pred.argmax(dim=1)
    pixel_acc = (pred_indices == target).float().mean().item()
    
    # Grid Accuracy: Every pixel in a single grid must match perfectly
    correct_grids = (pred_indices == target).all(dim=-1).all(dim=-1)
    task_acc = correct_grids.float().mean().item()
    
    return pixel_acc, task_acc

# =========================
# Main Execution Sequence
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/evaluation")
    parser.add_argument("--batch_size", type=int, default=1) 
    parser.add_argument("--t_steps", type=int, default=4)
    parser.add_argument("--max_segments", type=int, default=8)
    args = parser.parse_args()

    # Create evaluation root folder
    os.makedirs(EVAL_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(EVAL_DIR, "evaluation.log")),
            logging.StreamHandler()
        ]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Run Directory: {EVAL_DIR}")
    logging.info(f"Zero-Shot Evaluation Mode | Device: {device}")

    # Dataset and Model setup
    dataset = ARCTaskDataset(args.data_dir)
    loader = get_arc_loader(dataset, batch_size=args.batch_size)
    model = ARCFewShotHRM(dim=128, T_steps=args.t_steps, max_segments=args.max_segments).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"HRM params | total: {total_params:,} | trainable: {trainable_params:,}")

    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) 
    model.load_state_dict(state_dict)
    model.eval()

    total_pixel_acc = 0
    total_task_acc = 0
    task_count = 0

    logging.info("Starting inference loop...")

    with torch.no_grad():
        for batch in tqdm(loader):
            if batch["query_y"] is None:
                continue

            # Standard ARC Tensors: [Batch, Num_Examples, Channels, H, W]
            support_x = batch["support_x"].to(device)
            support_y = batch["support_y"].to(device)
            query_x = batch["query_x"].to(device)
            query_y = batch["query_y"].to(device)

            for b in range(support_x.shape[0]):
                # Retrieve actual lengths from masks
                s_len = int(batch["support_mask"][b].sum().item())
                q_len = int(batch["query_mask"][b].sum().item())

                sx = support_x[b, :s_len].long()
                sy = support_y[b, :s_len].long()
                qx = query_x[b, :q_len].long()
                qy = query_y[b, :q_len].long()

                if sx.numel() == 0 or qx.numel() == 0:
                    continue

                # Inference
                logits = model(sx, sy, qx)

                # Alignment: Crop output to the actual GT dimensions
                h, w = qy.shape[-2:]
                logits = logits[:, :, :h, :w]

                # Metrics
                p_acc, t_acc = get_metrics(logits, qy)

                # Visualize and save to run-specific folder
                preserve_visual_state(qx, qy, logits, task_count)

                total_pixel_acc += p_acc
                total_task_acc += t_acc
                task_count += 1

                if task_count % 10 == 0:
                    logging.info(f"Task {task_count:03d}: P-Acc {p_acc:.4f} | T-Acc {t_acc:.4f}")

    if task_count > 0:
        avg_pixel = total_pixel_acc / task_count
        avg_task = total_task_acc / task_count
        logging.info("-" * 30)
        logging.info(f"FINAL RESULTS FOR RUN: {RUN_ID}")
        logging.info(f"Mean Pixel Accuracy: {avg_pixel:.4f}")
        logging.info(f"Mean Task Accuracy:  {avg_task:.4f}")
        logging.info("-" * 30)
    else:
        logging.error("No valid tasks found for evaluation.")