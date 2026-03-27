
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from datetime import datetime
import copy
import os

# Assuming the improved ARCModel is defined in the same notebook or imported
# from improved_arc_model import ARCModel

# =========================
# Metrics
# =========================
def pixel_accuracy(pred_logits, target):
    # pred_logits: (B, C, H, W), target: (B, 1, H, W)
    pred = pred_logits.argmax(dim=1)
    target = target.squeeze(1).long()
    return (pred == target).float().mean().item()


def task_accuracy(pred_logits, target):
    # pred_logits: (B, C, H, W), target: (B, 1, H, W)
    pred = pred_logits.argmax(dim=1)
    target = target.squeeze(1).long()
    # Check if the entire grid matches for each item in the batch
    # For ARC evaluation, we usually have B=1 here after TTA
    return (pred == target).all(dim=(1, 2)).float().mean().item()


# =========================
# Configuration (Notebook style)
# =========================
class EvalConfig:
    model_path = "models/arc_best.pt"
    data_dir = "data/evaluation"
    batch_size = 4
    tta_steps = 15 # Slightly more steps for evaluation
    tta_lr = 5e-4
    adapt_encoder = False # Set to True if you want to adapt the full model

config = EvalConfig()

# =========================
# Setup Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"eval_improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Device: {device}")

# =========================
# Data & Model Loading
# =========================
from src.arc_dataset import ARCTaskDataset
from src.arc_dataloader import get_arc_loader

dataset = ARCTaskDataset(config.data_dir)
loader = get_arc_loader(dataset, batch_size=config.batch_size)

# Initialize the improved model
# Make sure ARCModel is defined in your environment
model = ARCModel().to(device)

if os.path.exists(config.model_path):
    logging.info(f"Loading model from {config.model_path}")
    checkpoint = torch.load(config.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
else:
    logging.warning(f"Model path {config.model_path} not found! Using random weights.")

model.eval()

# =========================
# Evaluation Loop
# =========================
total_pixel_acc = 0
total_task_acc = 0
count = 0

logging.info("Starting evaluation with Improved TTA...")

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
        # Extract individual tasks from the batch
        s_len = int(support_mask[b].sum().item())
        q_len = int(query_mask[b].sum().item())

        if s_len == 0 or q_len == 0:
            continue

        sx = support_x[b, :s_len]
        sy = support_y[b, :s_len]
        qx = query_x[b, :q_len]
        qy = query_y[b, :q_len]

        # Use the model's built-in forward_with_adaptation
        # This is more robust as it's designed specifically for this architecture
        pred_logits = model.forward_with_adaptation(
            sx, sy, qx,
            steps=config.tta_steps,
            lr=config.tta_lr,
            adapt_encoder=config.adapt_encoder
        )

        # Ensure the prediction matches the target spatial dimensions
        # (Though they should match if the architecture is consistent)
        h, w = qy.shape[-2:]
        pred_logits = pred_logits[:, :, :h, :w]

        # Calculate metrics
        p_acc = pixel_accuracy(pred_logits, qy)
        t_acc = task_accuracy(pred_logits, qy)

        total_pixel_acc += p_acc
        total_task_acc += t_acc
        count += 1

        # Optional: log every task for detailed tracking
        # logging.info(f"Task {count}: Pixel={p_acc:.4f}, Task={t_acc:.4f}")

# =========================
# Final Results
# =========================
if count > 0:
    final_pixel = total_pixel_acc / count
    final_task = total_task_acc / count
    logging.info("=" * 50)
    logging.info(f"Final Evaluation Results ({count} tasks):")
    logging.info(f"Average Pixel Accuracy: {final_pixel:.4f}")
    logging.info(f"Average Task Accuracy:  {final_task:.4f}")
    logging.info("=" * 50)
else:
    logging.info("No tasks were evaluated.")
