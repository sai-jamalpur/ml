import os
import json
from collections import defaultdict

def compute_metrics(base_dir):
    pattern_metrics = defaultdict(lambda: {"correct_tasks": 0, "total_tasks": 0, "correct_cells": 0, "total_cells": 0})
    total_metrics = {"correct_tasks": 0, "total_tasks": 0, "correct_cells": 0, "total_cells": 0}
    
    for pattern in os.listdir(base_dir):
        pattern_dir = os.path.join(base_dir, pattern)
        if not os.path.isdir(pattern_dir):
            continue
            
        for file in os.listdir(pattern_dir):
            if not file.endswith(".json"):
                continue
            
            file_path = os.path.join(pattern_dir, file)
            with open(file_path, "r") as f:
                data = json.load(f)
                
            pred = data.get("prediction", [])
            gt = data.get("groundtruth", [])
            
            # For 1D tasks, prediction and gt should be 2D lists like [[1, 2, ...]]
            # Ensure they are valid
            if not pred or not gt:
                continue
                
            pred_flat = [cell for row in pred for cell in row]
            gt_flat = [cell for row in gt for cell in row]
            
            task_correct = int(pred_flat == gt_flat)
            
            total_cells = max(len(gt_flat), len(pred_flat))
            correct_cells = sum(1 for p, g in zip(pred_flat, gt_flat) if p == g)
            
            pattern_metrics[pattern]["correct_tasks"] += task_correct
            pattern_metrics[pattern]["total_tasks"] += 1
            pattern_metrics[pattern]["correct_cells"] += correct_cells
            pattern_metrics[pattern]["total_cells"] += len(gt_flat)
            
            total_metrics["correct_tasks"] += task_correct
            total_metrics["total_tasks"] += 1
            total_metrics["correct_cells"] += correct_cells
            total_metrics["total_cells"] += len(gt_flat)
            
    # Calculate percentages
    pattern_wise = {}
    for pattern, metrics in pattern_metrics.items():
        pattern_wise[pattern] = {
            "task_accuracy": metrics["correct_tasks"] / metrics["total_tasks"] if metrics["total_tasks"] > 0 else 0,
            "cell_accuracy": metrics["correct_cells"] / metrics["total_cells"] if metrics["total_cells"] > 0 else 0,
            "total_tasks": metrics["total_tasks"]
        }
        
    summary = {
        "task_accuracy": total_metrics["correct_tasks"] / total_metrics["total_tasks"] if total_metrics["total_tasks"] > 0 else 0,
        "cell_accuracy": total_metrics["correct_cells"] / total_metrics["total_cells"] if total_metrics["total_cells"] > 0 else 0,
        "total_tasks": total_metrics["total_tasks"]
    }
    
    with open(os.path.join(base_dir, "metrics_pattern_wise.json"), "w") as f:
        json.dump(pattern_wise, f, indent=2)
        
    with open(os.path.join(base_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Metrics saved to {base_dir}")

if __name__ == "__main__":
    compute_metrics("arc-diff/outputs_new/predictions_new")
