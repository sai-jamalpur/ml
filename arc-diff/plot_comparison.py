import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(predictions_dir):
    pattern_wise_path = os.path.join(predictions_dir, "metrics_pattern_wise.json")
    summary_path = os.path.join(predictions_dir, "metrics_summary.json")

    with open(pattern_wise_path, "r") as f:
        pattern_data = json.load(f)
    
    with open(summary_path, "r") as f:
        summary_data = json.load(f)
        
    return pattern_data, summary_data

def main():
    old_dir = "/home/saij/ml/arc-diff/predictions"
    new_dir = "/home/saij/ml/arc-diff/outputs_new/predictions_new"
    
    old_pattern, old_summary = load_metrics(old_dir)
    new_pattern, new_summary = load_metrics(new_dir)

    all_patterns = sorted(list(set(old_pattern.keys()).union(set(new_pattern.keys()))))

    # Old metrics
    old_task_acc = [old_pattern.get(p, {}).get("task_accuracy", 0) * 100 for p in all_patterns]
    old_cell_acc = [old_pattern.get(p, {}).get("cell_accuracy", 0) * 100 for p in all_patterns]

    # New metrics
    new_task_acc = [new_pattern.get(p, {}).get("task_accuracy", 0) * 100 for p in all_patterns]
    new_cell_acc = [new_pattern.get(p, {}).get("cell_accuracy", 0) * 100 for p in all_patterns]

    # Sort by old task accuracy for better visualization
    sorted_indices = np.argsort(old_task_acc)
    patterns_sorted = [all_patterns[i] for i in sorted_indices]
    
    old_task_sorted = [old_task_acc[i] for i in sorted_indices]
    new_task_sorted = [new_task_acc[i] for i in sorted_indices]
    
    old_cell_sorted = [old_cell_acc[i] for i in sorted_indices]
    new_cell_sorted = [new_cell_acc[i] for i in sorted_indices]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})

    # Plot 1: Pattern-wise history (Grouped Bar chart)
    y_pos = np.arange(len(patterns_sorted))
    height = 0.35

    rects1 = ax1.barh(y_pos - height/2, old_task_sorted, height, label='Old Task Accuracy', color='lightskyblue')
    rects2 = ax1.barh(y_pos + height/2, new_task_sorted, height, label='New Task Accuracy', color='dodgerblue')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(patterns_sorted)
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title('Old vs New Pattern-wise Task Accuracy')
    ax1.legend()
    ax1.set_xlim(0, 100)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)

    # Plot 2: Summary visualization
    summary_labels = ['Task Accuracy', 'Cell Accuracy']
    old_values = [old_summary["task_accuracy"] * 100, old_summary["cell_accuracy"] * 100]
    new_values = [new_summary["task_accuracy"] * 100, new_summary["cell_accuracy"] * 100]
    
    x_pos = np.arange(len(summary_labels))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, old_values, width, label='Old Results', color='lightcoral')
    bars2 = ax2.bar(x_pos + width/2, new_values, width, label='New Results', color='crimson')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(summary_labels)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Overall Metrics Comparison')
    ax2.set_ylim(0, 100)
    ax2.legend()
    
    for bar in bars1 + bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    
    output_path = "/home/saij/ml/arc-diff/outputs_new/metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison visualization saved to {output_path}")

if __name__ == "__main__":
    main()
