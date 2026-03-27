#!/usr/bin/env python3
import subprocess
import json
import re
from datetime import datetime
import os

# Hyperparameter grid to search
GRID = {
    "entropy_w": [0.001, 0.01, 0.05],
    "tta_steps": [3, 5, 10],
    "tta_lr": [0.001, 0.005],
    "lr": [1e-4, 3e-4, 1e-3],
}

# Fixed parameters
FIXED_PARAMS = {
    "epochs": 3,
    "batch_size": 4,
}

results = []
total_configs = 1
for v in GRID.values():
    total_configs *= len(v)

print(f"{'='*70}")
print(f"HYPERPARAMETER GRID SEARCH")
print(f"{'='*70}")
print(f"Total configurations to test: {total_configs}")
print(f"Fixed params: {FIXED_PARAMS}")
print(f"Grid: {GRID}")
print(f"{'='*70}\n")

config_num = 0

# Generate all combinations
def generate_combinations(grid, keys=None, current=None):
    if keys is None:
        keys = list(grid.keys())
        current = {}
    
    if not keys:
        yield dict(current)
        return
    
    key = keys[0]
    for value in grid[key]:
        current[key] = value
        yield from generate_combinations(grid, keys[1:], current)

# Run each configuration
for config in generate_combinations(GRID):
    config_num += 1
    
    # Merge with fixed params
    params = {**FIXED_PARAMS, **config}
    
    print(f"[{config_num}/{total_configs}] Testing: {config}")
    
    # Build command
    cmd = [".venv/bin/python", "src/hpo.py"]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    
    # Run
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        # Parse final accuracy from stderr
        accuracy = None
        train_loss = None
        
        for line in result.stderr.split('\n'):
            if "Eval accuracy:" in line:
                match = re.search(r'Eval accuracy: ([\d.]+)', line)
                if match:
                    accuracy = float(match.group(1))
            if "Train loss:" in line:
                match = re.search(r'Train loss: ([\d.]+)', line)
                if match:
                    train_loss = float(match.group(1))
        
        if accuracy is not None:
            result_entry = {
                "config_num": config_num,
                "entropy_w": config["entropy_w"],
                "tta_steps": config["tta_steps"],
                "tta_lr": config["tta_lr"],
                "lr": config["lr"],
                "epochs": params["epochs"],
                "batch_size": params["batch_size"],
                "accuracy": accuracy,
                "train_loss": train_loss,
            }
            results.append(result_entry)
            print(f"  ✓ Accuracy: {accuracy:.4f}, Loss: {train_loss:.4f}\n")
        else:
            print(f"  ✗ Failed to parse accuracy\n")
    
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout (>600s)\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")

# Sort by accuracy descending
results.sort(key=lambda x: x['accuracy'], reverse=True)

# Print results
print(f"\n{'='*70}")
print("GRID SEARCH RESULTS (sorted by accuracy)")
print(f"{'='*70}\n")

for i, r in enumerate(results, 1):
    print(f"{i}. Accuracy: {r['accuracy']:.4f} | Loss: {r['train_loss']:.4f}")
    print(f"   entropy_w={r['entropy_w']}, tta_steps={r['tta_steps']}, tta_lr={r['tta_lr']}, lr={r['lr']}")

# Save results to JSON
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f"hpo_grid_results_{timestamp}.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print(f"Results saved to: {output_file}")
print(f"{'='*70}\n")

# Print best configuration
if results:
    best = results[0]
    print("BEST CONFIGURATION:")
    print(f".venv/bin/python src/hpo.py \\")
    print(f"  --entropy_w {best['entropy_w']} \\")
    print(f"  --tta_steps {best['tta_steps']} \\")
    print(f"  --tta_lr {best['tta_lr']} \\")
    print(f"  --lr {best['lr']} \\")
    print(f"  --epochs {best['epochs']} \\")
    print(f"  --batch_size {best['batch_size']}")
    print(f"\nExpected accuracy: {best['accuracy']:.4f}")
