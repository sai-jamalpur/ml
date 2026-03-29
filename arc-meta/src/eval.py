#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from datetime import datetime
import copy
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
EVAL_DIR = f"eval_tta_run_{RUN_ID}"
VISUAL_DIR = os.path.join(EVAL_DIR, "visuals")

def preserve_visual_state(qx, qy, pred, task_id):
    """
    Reconstructs the spatial mapping of the task and preserves it as a visual artifact.
    """
    os.makedirs(VISUAL_DIR, exist_ok=True)
    
    input_grid = qx.squeeze().cpu().numpy()
    truth_grid = qy.squeeze().cpu().numpy()
    pred_grid = pred.argmax(dim=1).squeeze().cpu().numpy()

    if input_grid.ndim > 2: input_grid = input_grid[-1]
    if truth_grid.ndim > 2: truth_grid = truth_grid[-1]
    if pred_grid.ndim > 2: pred_grid = pred_grid[-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_grid, cmap=ARC_CMAP, norm=ARC_NORM)
    axes[0].set_title(f"Input (Query X)\nTask: {task_id}", fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(truth_grid, cmap=ARC_CMAP, norm=ARC_NORM)
    axes[1].set_title("Ground Truth (Query Y)", fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(pred_grid, cmap=ARC_CMAP, norm=ARC_NORM)
    axes[2].set_title("TTA Prediction", fontsize=10)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, f"task_{task_id:04d}.png"), dpi=150)
    plt.close(fig)

# =========================
# Metrics
# =========================
def get_metrics(pred, target):
    pred_indices = pred.argmax(dim=1)
    pixel_acc = (pred_indices == target).float().mean().item()
    correct_grids = (pred_indices == target).all(dim=-1).all(dim=-1)
    task_acc = correct_grids.float().mean().item()
    return pixel_acc, task_acc

# =========================
# Architecturally Aligned TTA
# =========================
def execute_tta(model, sx, sy, qx, steps, lr):
    """
    Performs Test-Time Augmentation by adapting a local copy of the model
    to the specific support examples of the current task.
    """
    # Clone model to avoid bleeding weights across different tasks
    local_model = copy.deepcopy(model)
    local_model.train()
    
    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)

    for i in range(steps):
        optimizer.zero_grad()
        # The model tries to reconstruct Support Y from Support X given its own context
        # return_all_segments=True enables deep supervision if supported by your HRM
        outputs = local_model(sx, sy, sx, return_all_segments=True)
        
        # Loss is average cross entropy across all reasoning segments
        loss = sum(F.cross_entropy(seg, sy.squeeze(1).long()) for seg in outputs) / len(outputs)
        
        loss.backward()
        optimizer.step()

    local_model.eval()
    with torch.no_grad():
        # Final inference on the actual Query X
        return local_model(sx, sy, qx)

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
    parser.add_argument("--tta_steps", type=int, default=10)
    parser.add_argument("--tta_lr", type=float, default=5e-4) # Slightly lower LR for stability
    args = parser.parse_args()

    os.makedirs(EVAL_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(EVAL_DIR, "evaluation_tta.log")),
            logging.StreamHandler()
        ]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"TTA Evaluation Run: {EVAL_DIR}")
    
    # Setup
    dataset = ARCTaskDataset(args.data_dir)
    loader = get_arc_loader(dataset, batch_size=args.batch_size)
    model = ARCFewShotHRM(dim=128, T_steps=args.t_steps, max_segments=args.max_segments).to(device)

    # Load Weights
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) 
    model.load_state_dict(state_dict)
    model.eval()

    total_pixel_acc = 0
    total_task_acc = 0
    task_count = 0

    logging.info(f"Starting TTA loop ({args.tta_steps} steps per task)...")

    for batch in tqdm(loader):
        if batch["query_y"] is None:
            continue

        support_x = batch["support_x"].to(device)
        support_y = batch["support_y"].to(device)
        query_x = batch["query_x"].to(device)
        query_y = batch["query_y"].to(device)

        for b in range(support_x.shape[0]):
            s_len = int(batch["support_mask"][b].sum().item())
            q_len = int(batch["query_mask"][b].sum().item())

            sx = support_x[b, :s_len].long()
            sy = support_y[b, :s_len].long()
            qx = query_x[b, :q_len].long()
            qy = query_y[b, :q_len].long()

            if sx.numel() == 0 or qx.numel() == 0:
                continue

            # Execute the TTA Adaptation
            logits = execute_tta(model, sx, sy, qx, steps=args.tta_steps, lr=args.tta_lr)

            # Alignment
            h, w = qy.shape[-2:]
            logits = logits[:, :, :h, :w]

            # Metrics
            p_acc, t_acc = get_metrics(logits, qy)

            # Visualize
            preserve_visual_state(qx, qy, logits, task_count)

            total_pixel_acc += p_acc
            total_task_acc += t_acc
            task_count += 1

            if task_count % 5 == 0:
                logging.info(f"Task {task_count:03d}: P-Acc {p_acc:.4f} | T-Acc {t_acc:.4f}")

    if task_count > 0:
        logging.info("-" * 40)
        logging.info(f"FINAL AGGREGATE RESULTS (TTA RUN)")
        logging.info(f"Mean Pixel Accuracy: {total_pixel_acc / task_count:.4f}")
        logging.info(f"Mean Task Accuracy:  {total_task_acc / task_count:.4f}")
        logging.info("-" * 40)