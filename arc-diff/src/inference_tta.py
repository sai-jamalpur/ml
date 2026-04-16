"""
inference_tta.py — Test-Time Adaptation (TTA) inference for ARC Diffusion Model.

Meta-learning style TTA: for each test task file, briefly fine-tune the model on
the file's training examples (augmented with D4 symmetries and color shifts), then
run inference with the adapted model, and restore original weights before moving
to the next file.

Usage:
    python src/inference_tta.py \\
        --checkpoint models/model.pt \\
        --data-dir aug_data \\
        --tta-steps 20 \\
        --tta-lr 1e-4 \\
        --output-dir predictions_tta
"""

import argparse
import copy
import importlib.util
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


# ---------------------------------------------------------------------------
# Module loading helpers (mirrors inference.py)
# ---------------------------------------------------------------------------

def _load_local_module(module_filename: str, module_name: str):
    module_path = CURRENT_DIR / module_filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_filename} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_dataloader_module = _load_local_module("dataloader.py", "arc_diff_dataloader")
_model_module = _load_local_module("model.py", "arc_diff_model")

ARCTestTorchDataset = _dataloader_module.ARCTestTorchDataset
arc_collate_fn = _dataloader_module.arc_collate_fn
ARCDiffusionModel = _model_module.ARCDiffusionModel


# ---------------------------------------------------------------------------
# Noise scheduler (same as inference.py / train.py)
# ---------------------------------------------------------------------------

class DiscreteNoiseScheduler:
    def __init__(self, num_timesteps: int = 50, vocab_size: int = 10):
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bsz, h, w = x0.shape
        alpha_bar = self.alpha_bars[t].view(bsz, 1, 1)
        noise_mask = torch.rand(bsz, h, w, device=x0.device) > alpha_bar
        random_tokens = torch.randint(0, self.vocab_size, (bsz, h, w), device=x0.device)
        return torch.where(noise_mask, random_tokens, x0)


# ---------------------------------------------------------------------------
# Augmentation utilities
# ---------------------------------------------------------------------------

def _apply_d4(grid: np.ndarray, d4_idx: int) -> np.ndarray:
    if d4_idx == 0:
        return grid
    elif d4_idx == 1:
        return np.rot90(grid, 1)
    elif d4_idx == 2:
        return np.rot90(grid, 2)
    elif d4_idx == 3:
        return np.rot90(grid, 3)
    elif d4_idx == 4:
        return np.fliplr(grid)
    elif d4_idx == 5:
        return np.flipud(grid)
    elif d4_idx == 6:
        return np.transpose(grid)
    elif d4_idx == 7:
        return np.fliplr(np.transpose(grid))
    raise ValueError(f"Invalid d4_idx: {d4_idx}")


def _apply_color_shift(grid: np.ndarray, shift: int) -> np.ndarray:
    return (grid + shift) % 10


# ---------------------------------------------------------------------------
# Build TTA training examples from a JSON file
# ---------------------------------------------------------------------------

def build_tta_examples(
    json_path: Path,
    task_idx: int,
    max_size: int = 30,
    d4_range: int = 8,
    color_range: int = 10,
    max_examples: Optional[int] = None,
) -> List[Dict]:
    """Load the 'train' split from a JSON file and augment with D4 × color shifts.

    Each example in aug_data already contains the original + augmented
    training examples (written by ARCDatasetBuilder). For TTA we directly
    use those; optionally cap with *max_examples* to limit GPU memory.

    If you want extra on-the-fly augmentation (e.g., for small datasets),
    set d4_range > 1 or color_range > 1 — the function will re-augment the
    raw examples from the file's train split by applying all D4 × color combos.
    When d4_range=1 and color_range=1 the raw training examples are returned
    as-is (identity only), which is appropriate when loading from aug_data
    that already contains augmented versions.
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    raw_train = data.get("train", [])
    examples: List[Dict] = []

    for entry in raw_train:
        inp = np.array(entry["input"], dtype=np.int64)
        out = np.array(entry["output"], dtype=np.int64)
        if inp.ndim != 2 or out.ndim != 2:
            continue
        if inp.shape[0] > max_size or inp.shape[1] > max_size:
            continue
        if out.shape[0] > max_size or out.shape[1] > max_size:
            continue

        for d4_idx in range(d4_range):
            for color_shift in range(color_range):
                aug_inp = _apply_d4(inp.copy(), d4_idx)
                aug_inp = _apply_color_shift(aug_inp, color_shift)
                aug_out = _apply_d4(out.copy(), d4_idx)
                aug_out = _apply_color_shift(aug_out, color_shift)
                examples.append({
                    "input": aug_inp,
                    "output": aug_out,
                    "task_idx": task_idx,
                })

        if max_examples is not None and len(examples) >= max_examples:
            break

    if max_examples is not None:
        examples = examples[:max_examples]

    return examples


# ---------------------------------------------------------------------------
# Collate a list of raw example dicts into a padded tensor batch
# ---------------------------------------------------------------------------

def _collate_tta_batch(
    examples: List[Dict],
    model_size: int,
    device: torch.device,
    pad_value: int = 10,
) -> Dict[str, torch.Tensor]:
    """Pad a list of example dicts to model_size × model_size and stack."""
    inputs, outputs, in_masks, out_masks, task_ids = [], [], [], [], []

    for e in examples:
        inp: np.ndarray = e["input"]
        out: np.ndarray = e["output"]

        ih, iw = inp.shape[0], inp.shape[1]
        oh, ow = out.shape[0], out.shape[1]

        padded_inp = np.full((model_size, model_size), pad_value, dtype=np.int64)
        padded_inp[:ih, :iw] = inp[:ih, :iw]

        padded_out = np.full((model_size, model_size), pad_value, dtype=np.int64)
        padded_out[:oh, :ow] = out[:oh, :ow]

        in_mask = np.zeros((model_size, model_size), dtype=bool)
        in_mask[:ih, :iw] = True

        out_mask = np.zeros((model_size, model_size), dtype=bool)
        out_mask[:oh, :ow] = True

        inputs.append(padded_inp)
        outputs.append(padded_out)
        in_masks.append(in_mask)
        out_masks.append(out_mask)
        task_ids.append(e["task_idx"])

    return {
        "input_grid": torch.tensor(np.stack(inputs), dtype=torch.long, device=device),
        "output_grid": torch.tensor(np.stack(outputs), dtype=torch.long, device=device),
        "input_mask": torch.tensor(np.stack(in_masks), dtype=torch.bool, device=device),
        "output_mask": torch.tensor(np.stack(out_masks), dtype=torch.bool, device=device),
        "task_idx": torch.tensor(task_ids, dtype=torch.long, device=device),
    }


# ---------------------------------------------------------------------------
# TTA adaptation step
# ---------------------------------------------------------------------------

def tta_adapt(
    model: torch.nn.Module,
    scheduler: DiscreteNoiseScheduler,
    train_examples: List[Dict],
    steps: int,
    lr: float,
    tta_batch_size: int,
    model_size: int,
    device: torch.device,
    pad_value: int = 10,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Fine-tune *model* in-place on *train_examples* for *steps* gradient steps.

    Uses the same two-pass self-conditioning training objective as train.py.
    Model is set to train mode during adaptation and eval mode when done.
    """
    if not train_examples or steps <= 0:
        return

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(train_examples)

    for step in range(steps):
        idx = np.random.choice(n, min(tta_batch_size, n), replace=False).tolist()
        batch = _collate_tta_batch([train_examples[i] for i in idx], model_size, device, pad_value)

        input_grid = batch["input_grid"]
        output_grid = batch["output_grid"]
        output_mask = batch["output_mask"]
        task_ids = batch["task_idx"]
        bsz = input_grid.shape[0]

        t = torch.randint(0, scheduler.num_timesteps, (bsz,), device=device)
        alpha_bar = scheduler.alpha_bars[t].clamp(1e-5, 1 - 1e-5)
        logsnr = torch.log(alpha_bar) - torch.log(1 - alpha_bar)
        xt = scheduler.add_noise(output_grid, t)

        optimizer.zero_grad(set_to_none=True)

        # First pass: get soft self-conditioning labels (no grad)
        with torch.no_grad():
            logits_prev = model(
                xt=xt,
                input_grid=input_grid,
                task_ids=task_ids,
                logsnr=logsnr,
                d4_idx=torch.zeros(bsz, dtype=torch.long, device=device),
                color_shift=torch.zeros(bsz, dtype=torch.long, device=device),
                masks=output_mask,
                sc_p0=None,
            )
            sc = torch.log_softmax(logits_prev, dim=-1)
            sc = sc * output_mask.unsqueeze(-1).float()

        # Second pass: conditioned on soft labels (with grad)
        logits = model(
            xt=xt,
            input_grid=input_grid,
            task_ids=task_ids,
            logsnr=logsnr,
            d4_idx=torch.zeros(bsz, dtype=torch.long, device=device),
            color_shift=torch.zeros(bsz, dtype=torch.long, device=device),
            masks=output_mask,
            sc_p0=sc,
        )

        valid_mask = output_mask.view(-1).float()
        per_cell_loss = F.cross_entropy(
            logits.view(-1, 10),
            output_grid.view(-1),
            reduction="none",
            ignore_index=10,
        )
        loss = (per_cell_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if logger is not None and (step == 0 or (step + 1) % 10 == 0):
            logger.debug("  TTA step %d/%d  loss=%.4f", step + 1, steps, loss.item())

    model.eval()


# ---------------------------------------------------------------------------
# Single-batch inference (no_grad, mirrors inference.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _infer_batch(
    model: torch.nn.Module,
    scheduler: DiscreteNoiseScheduler,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run two-pass diffusion inference on a padded batch.

    Returns (predictions, pred_h, pred_w) all on CPU.
    """
    input_grid = batch["input_grid"].to(device)
    output_mask = batch["output_mask"].to(device)
    input_mask = batch["input_mask"].to(device)
    task_ids = batch["task_idx"].to(device)
    has_output = batch["has_output"].to(device)

    bsz = input_grid.shape[0]
    timesteps = torch.full((bsz,), scheduler.num_timesteps - 1, dtype=torch.long, device=device)
    alpha_bar = scheduler.alpha_bars[timesteps].clamp(1e-5, 1 - 1e-5)
    logsnr = torch.log(alpha_bar) - torch.log(1 - alpha_bar)

    xt = torch.randint(0, scheduler.vocab_size, (bsz, model.max_size, model.max_size), device=device)
    effective_mask = torch.where(has_output[:, None, None], output_mask, input_mask)

    d4 = batch.get("d4_idx", torch.zeros(bsz, dtype=torch.long)).to(device)
    color = batch.get("color_shift", torch.zeros(bsz, dtype=torch.long)).to(device)

    logits_prev = model(
        xt=xt, input_grid=input_grid, task_ids=task_ids,
        logsnr=logsnr, d4_idx=d4, color_shift=color,
        masks=effective_mask, sc_p0=None,
    )
    sc = torch.log_softmax(logits_prev, dim=-1)
    sc = sc * effective_mask.unsqueeze(-1).float()

    logits = model(
        xt=xt, input_grid=input_grid, task_ids=task_ids,
        logsnr=logsnr, d4_idx=d4, color_shift=color,
        masks=effective_mask, sc_p0=sc,
    )
    predictions = logits.argmax(dim=-1).cpu()

    pred_h, pred_w = model.predict_sizes(
        input_grid=input_grid,
        task_ids=task_ids,
        d4_idx=d4,
        color_shift=color,
    )
    return predictions, pred_h.cpu(), pred_w.cpu()


# ---------------------------------------------------------------------------
# Utilities shared with inference.py
# ---------------------------------------------------------------------------

def _crop_grid(grid: torch.Tensor, height: int, width: int) -> List[List[int]]:
    return grid[:height, :width].tolist()


def _infer_input_size(mask_2d: torch.Tensor) -> Tuple[int, int]:
    rows = int(mask_2d.any(dim=1).sum().item())
    cols = int(mask_2d.any(dim=0).sum().item())
    return rows, cols


def _new_group_stats() -> Dict:
    return {
        "total_examples": 0,
        "labeled_examples": 0,
        "cell_correct": 0.0,
        "cell_total": 0.0,
        "task_correct": 0,
        "task_total": 0,
    }


def _update_group_stats(stats, has_label, correct_cells, total_cells, exact_match):
    stats["total_examples"] += 1
    if not has_label:
        return
    stats["labeled_examples"] += 1
    stats["cell_correct"] += float(correct_cells)
    stats["cell_total"] += float(total_cells)
    stats["task_total"] += 1
    if exact_match:
        stats["task_correct"] += 1


def _finalize_group_stats(stats_map):
    finalized = {}
    for key in sorted(stats_map.keys()):
        item = stats_map[key]
        ct = item["cell_total"]
        tt = item["task_total"]
        finalized[key] = {
            "total_examples": item["total_examples"],
            "labeled_examples": item["labeled_examples"],
            "unlabeled_examples": item["total_examples"] - item["labeled_examples"],
            "cell_accuracy": (item["cell_correct"] / ct) if ct > 0 else None,
            "task_accuracy": (item["task_correct"] / tt) if tt > 0 else None,
            "cell_correct": item["cell_correct"],
            "cell_total": ct,
            "task_correct": item["task_correct"],
            "task_total": tt,
        }
    return finalized


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("arc_diff_tta")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _append_jsonl(path: Path, payload: Dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def _model_param_stats(model: torch.nn.Module) -> Dict:
    return {
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def _pad_batch_to_model_size(batch: Dict, model_size: int, pad_value: int = 10) -> Dict:
    """Pad collated batch tensors to model_size × model_size."""
    def _pad(t: torch.Tensor, fill):
        out = torch.full((t.shape[0], model_size, model_size), fill, dtype=t.dtype, device=t.device)
        out[:, :t.shape[-2], :t.shape[-1]] = t
        return out

    return {
        "input_grid": _pad(batch["input_grid"], pad_value),
        "output_grid": _pad(batch["output_grid"], pad_value),
        "input_mask": _pad(batch["input_mask"], False),
        "output_mask": _pad(batch["output_mask"], False),
        "task_idx": batch["task_idx"],
        "height": batch["height"],
        "width": batch["width"],
        "d4_idx": batch["d4_idx"],
        "color_shift": batch["color_shift"],
        "task_ids": batch["task_ids"],
        "file_name": batch["file_name"],
        "has_output": batch["has_output"],
    }


# ---------------------------------------------------------------------------
# Main TTA inference loop
# ---------------------------------------------------------------------------

def run_inference_tta(args) -> None:
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"tta_{Path(args.checkpoint).stem}_{run_tag}"
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(log_dir / f"{run_name}.log")

    tracker_path = Path(args.tracker_file)
    tracker_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Starting TTA inference run %s on %s", run_name, device)
    logger.info("TTA steps=%d  lr=%g  tta_batch_size=%d  tta_max_examples=%s",
                args.tta_steps, args.tta_lr, args.tta_batch_size,
                str(args.tta_max_examples) if args.tta_max_examples else "all")

    # ---- Load model --------------------------------------------------------
    model = ARCDiffusionModel(max_size=args.max_size).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    param_stats = _model_param_stats(model)
    ckpt_size = Path(args.checkpoint).stat().st_size

    logger.info("Loaded checkpoint %s (%d bytes, %d params)",
                args.checkpoint, ckpt_size, param_stats["total_params"])

    # Keep a clean copy to restore after each file's adaptation
    original_state: Dict = copy.deepcopy(model.state_dict())

    scheduler = DiscreteNoiseScheduler(num_timesteps=args.num_timesteps).to(device)

    # ---- Load test dataset -------------------------------------------------
    test_dataset = ARCTestTorchDataset(
        root_dir=args.data_dir,
        max_size=args.max_size,
        task_types=args.task_types,
        task_ids=args.task_ids,
    )
    logger.info("Loaded test dataset: %d examples from %s", len(test_dataset), args.data_dir)

    # Group example indices by (task_type, file_name)
    file_groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for idx in range(len(test_dataset)):
        ex = test_dataset.examples[idx]
        key = (ex["task_id"], ex["file_name"])
        file_groups[key].append(idx)

    logger.info("Found %d unique task files", len(file_groups))

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    _append_jsonl(tracker_path, {
        "mode": "tta_inference",
        "event": "run_started",
        "run_tag": run_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "tta_steps": args.tta_steps,
        "tta_lr": args.tta_lr,
        "tta_batch_size": args.tta_batch_size,
        "output_dir": str(output_root),
        "data_dir": args.data_dir,
        **param_stats,
    })

    # ---- Metrics accumulators ----------------------------------------------
    total_examples = 0
    labeled_examples = 0
    cell_correct = 0.0
    cell_total = 0.0
    task_correct = 0
    task_total = 0
    per_file_counter: Dict = defaultdict(int)
    pattern_metrics: Dict = defaultdict(_new_group_stats)
    task_metrics: Dict = defaultdict(_new_group_stats)

    run_status = "ok"
    error_message = None

    try:
        sorted_files = sorted(file_groups.keys())
        for file_idx, (task_type, file_name) in enumerate(sorted_files):
            if args.max_batches is not None and file_idx >= args.max_batches:
                break

            example_indices = file_groups[(task_type, file_name)]
            json_path = Path(args.data_dir) / task_type / file_name
            logger.info("[%d/%d] File: %s/%s  (%d test examples)",
                        file_idx + 1, len(sorted_files), task_type, file_name,
                        len(example_indices))

            # ---- TTA: load and augment training examples -------------------
            task_idx_int = test_dataset.task_id_to_idx[task_type]

            # When loading from aug_data, the train split already contains
            # augmented examples. We load them as-is (d4_range=1, color_range=1)
            # to avoid double-augmentation. Switch to d4_range=8, color_range=10
            # if running against the raw (non-augmented) dataset.
            d4_range = args.tta_aug_d4 if args.tta_augment else 1
            color_range = args.tta_aug_colors if args.tta_augment else 1

            tta_examples: List[Dict] = []
            if json_path.exists():
                tta_examples = build_tta_examples(
                    json_path=json_path,
                    task_idx=task_idx_int,
                    max_size=args.max_size,
                    d4_range=d4_range,
                    color_range=color_range,
                    max_examples=args.tta_max_examples,
                )
            else:
                logger.warning("JSON file not found, skipping TTA for this file: %s", json_path)

            logger.info("  TTA pool: %d examples  adapting for %d steps ...",
                        len(tta_examples), args.tta_steps)

            # ---- Adapt model -----------------------------------------------
            tta_adapt(
                model=model,
                scheduler=scheduler,
                train_examples=tta_examples,
                steps=args.tta_steps,
                lr=args.tta_lr,
                tta_batch_size=args.tta_batch_size,
                model_size=args.max_size,
                device=device,
                pad_value=args.pad_value,
                logger=logger,
            )

            # ---- Inference on this file's test examples --------------------
            for batch_start in range(0, len(example_indices), args.batch_size):
                batch_indices = example_indices[batch_start: batch_start + args.batch_size]
                raw_items = [test_dataset[i] for i in batch_indices]
                batch = arc_collate_fn(raw_items, pad_value=args.pad_value)
                batch = _pad_batch_to_model_size(batch, args.max_size, pad_value=args.pad_value)

                predictions, pred_h, pred_w = _infer_batch(model, scheduler, batch, device)

                input_grid = batch["input_grid"]
                output_grid = batch["output_grid"]
                input_mask = batch["input_mask"]
                output_mask = batch["output_mask"]
                has_output = batch["has_output"]

                for i in range(len(batch_indices)):
                    total_examples += 1
                    t_name = str(batch["task_ids"][i])
                    f_name = str(batch["file_name"][i])
                    file_stem = Path(f_name).stem
                    pattern_id = t_name
                    task_id = f"{t_name}/{file_stem}"

                    per_file_counter[(t_name, f_name)] += 1
                    example_idx = per_file_counter[(t_name, f_name)] - 1

                    in_h, in_w = _infer_input_size(input_mask[i])
                    input_list = _crop_grid(input_grid[i], in_h, in_w)

                    if bool(has_output[i].item()):
                        out_h = int(batch["height"][i].item())
                        out_w = int(batch["width"][i].item())
                    else:
                        out_h = int(pred_h[i].item())
                        out_w = int(pred_w[i].item())

                    out_h = max(1, min(out_h, args.max_size))
                    out_w = max(1, min(out_w, args.max_size))

                    pred_list = _crop_grid(predictions[i], out_h, out_w)

                    groundtruth_list = None
                    has_label = bool(has_output[i].item())
                    correct_cells_f = 0.0
                    total_cells_f = 0.0
                    exact_match = False

                    if has_label:
                        labeled_examples += 1
                        groundtruth_list = _crop_grid(output_grid[i], out_h, out_w)
                        mask_i = output_mask[i, :out_h, :out_w]
                        pred_i = predictions[i, :out_h, :out_w]
                        target_i = output_grid[i, :out_h, :out_w]
                        correct_cells_f = float(((pred_i == target_i) & mask_i).float().sum().item())
                        total_cells_f = float(mask_i.float().sum().item())
                        cell_correct += correct_cells_f
                        cell_total += total_cells_f
                        task_total += 1
                        exact_match = (
                            bool(torch.equal(pred_i[mask_i], target_i[mask_i]))
                            if mask_i.any() else True
                        )
                        if exact_match:
                            task_correct += 1

                    _update_group_stats(pattern_metrics[pattern_id], has_label,
                                        correct_cells_f, total_cells_f, exact_match)
                    _update_group_stats(task_metrics[task_id], has_label,
                                        correct_cells_f, total_cells_f, exact_match)

                    task_dir = output_root / t_name
                    task_dir.mkdir(parents=True, exist_ok=True)
                    out_path = task_dir / f"{file_stem}_test_{example_idx:03d}.json"
                    out_path.write_text(json.dumps({
                        "task_id": t_name,
                        "file_name": f_name,
                        "example_index": example_idx,
                        "input": input_list,
                        "groundtruth": groundtruth_list,
                        "prediction": pred_list,
                    }), encoding="utf-8")

            logger.info("  cell_acc_so_far=%.4f  task_acc_so_far=%.4f",
                        cell_correct / max(cell_total, 1.0),
                        task_correct / max(task_total, 1))

            # ---- Restore model weights for the next file -------------------
            model.load_state_dict(original_state)
            model.eval()

    except Exception as exc:
        run_status = "error"
        error_message = str(exc)
        logger.exception("TTA inference failed: %s", exc)
        _append_jsonl(tracker_path, {
            "mode": "tta_inference",
            "event": "run_finished",
            "run_tag": run_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": run_status,
            "error": error_message,
            "total_examples": total_examples,
            **param_stats,
        })
        raise

    # ---- Save metrics ------------------------------------------------------
    cell_acc = (cell_correct / cell_total) if cell_total > 0 else None
    task_acc = (task_correct / task_total) if task_total > 0 else None

    summary = {
        "mode": "tta",
        "tta_steps": args.tta_steps,
        "tta_lr": args.tta_lr,
        "total_examples": total_examples,
        "labeled_examples": labeled_examples,
        "unlabeled_examples": total_examples - labeled_examples,
        "cell_accuracy": cell_acc,
        "task_accuracy": task_acc,
        "cell_correct": cell_correct,
        "cell_total": cell_total,
        "task_correct": task_correct,
        "task_total": task_total,
    }

    (output_root / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "metrics_pattern_wise.json").write_text(
        json.dumps(_finalize_group_stats(pattern_metrics), indent=2), encoding="utf-8")
    (output_root / "metrics_task_wise.json").write_text(
        json.dumps(_finalize_group_stats(task_metrics), indent=2), encoding="utf-8")

    logger.info("Saved predictions to %s", output_root)
    logger.info("Summary: %s", json.dumps(summary))

    _append_jsonl(tracker_path, {
        "mode": "tta_inference",
        "event": "run_finished",
        "run_tag": run_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(args.checkpoint),
        "status": run_status,
        "error": error_message,
        "total_examples": total_examples,
        "labeled_examples": labeled_examples,
        "cell_accuracy": cell_acc,
        "task_accuracy": task_acc,
        "tta_steps": args.tta_steps,
        "tta_lr": args.tta_lr,
        **param_stats,
    })


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [i.strip() for i in value.split(",") if i.strip()]
    return items if items else None


def _load_yaml_config(path: str) -> Dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for --config support.") from exc
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg if isinstance(cfg, dict) else {}


def _apply_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args
    cfg = _load_yaml_config(args.config)
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    infer = cfg.get("inference", {})
    tta = cfg.get("tta", {})

    def _set(key, val):
        if val is not None and hasattr(args, key):
            setattr(args, key, val)

    _set("data_dir", data.get("augmented_dataset_path"))
    _set("task_types", data.get("task_types"))
    _set("task_ids", data.get("task_ids"))
    _set("num_workers", data.get("num_workers"))
    _set("max_size", model.get("max_size"))
    _set("num_timesteps", model.get("num_timesteps"))
    _set("checkpoint", infer.get("checkpoint"))
    _set("batch_size", infer.get("batch_size"))
    _set("output_dir", infer.get("output_dir"))
    _set("tta_steps", tta.get("steps"))
    _set("tta_lr", tta.get("lr"))
    _set("tta_batch_size", tta.get("batch_size"))
    _set("tta_max_examples", tta.get("max_examples"))
    _set("tta_augment", tta.get("augment"))
    _set("tta_aug_d4", tta.get("aug_d4"))
    _set("tta_aug_colors", tta.get("aug_colors"))

    if not args.checkpoint:
        raise ValueError("Checkpoint is required. Set --checkpoint or inference.checkpoint in --config.")
    return args


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ARC inference with Test-Time Adaptation (meta-learning style)."
    )
    # --- Standard inference args ---
    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pt)")
    p.add_argument("--data-dir", default="aug_data", help="Root folder with ARC task JSON files")
    p.add_argument("--task-types", type=_parse_csv_list, default=None,
                   help="Comma-separated task type folders to include")
    p.add_argument("--task-ids", type=_parse_csv_list, default=None,
                   help="Comma-separated task file stems to include")
    p.add_argument("--output-dir", default="predictions_tta", help="Where to save predictions")
    p.add_argument("--batch-size", type=int, default=8, help="Inference batch size per file")
    p.add_argument("--max-size", type=int, default=30, help="Maximum grid size")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    p.add_argument("--pad-value", type=int, default=10, help="Pad token")
    p.add_argument("--num-timesteps", type=int, default=50, help="Diffusion timesteps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-batches", type=int, default=None,
                   help="Cap on number of files to process (for quick tests)")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--tracker-file", default="models/run_tracker.jsonl")

    # --- TTA-specific args ---
    p.add_argument("--tta-steps", type=int, default=20,
                   help="Gradient steps for TTA adaptation per task file (default: 20)")
    p.add_argument("--tta-lr", type=float, default=1e-4,
                   help="Learning rate for TTA (default: 1e-4, lower than training to avoid forgetting)")
    p.add_argument("--tta-batch-size", type=int, default=8,
                   help="Mini-batch size drawn from TTA pool each step (default: 8)")
    p.add_argument("--tta-max-examples", type=int, default=None,
                   help="Cap on TTA training pool size per file (default: use all)")
    p.add_argument("--tta-augment", action="store_true", default=False,
                   help="Apply extra D4×color augmentation on top of aug_data train examples. "
                        "Useful when running against the raw (non-augmented) dataset.")
    p.add_argument("--tta-aug-d4", type=int, default=8,
                   help="Number of D4 transforms to apply when --tta-augment is set (default: 8)")
    p.add_argument("--tta-aug-colors", type=int, default=10,
                   help="Number of color shifts to apply when --tta-augment is set (default: 10)")
    return p


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed = _apply_config(parsed)
    run_inference_tta(parsed)
