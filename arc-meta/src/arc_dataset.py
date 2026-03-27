import json
import numpy as np
import torch
from pathlib import Path


def pad_grid(grid, size=30):
    h, w = grid.shape
    padded = np.zeros((size, size), dtype=np.int64)
    padded[:h, :w] = grid
    return padded


class ARCTaskDataset:
    def __init__(self, data_dir):
        self.files = list(Path(data_dir).glob("*.json"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx]) as f:
            data = json.load(f)

        train = data["train"]
        test = data["test"]

        support_x, support_y = [], []
        query_x, query_y = [], []

        for t in train:
            x = pad_grid(np.array(t["input"]))
            y = pad_grid(np.array(t["output"]))
            support_x.append(x)
            support_y.append(y)

        for t in test:
            x = pad_grid(np.array(t["input"]))
            query_x.append(x)

            if "output" in t:
                y = pad_grid(np.array(t["output"]))
                query_y.append(y)

        support_x = torch.from_numpy(np.array(support_x)).unsqueeze(1).float()
        support_y = torch.from_numpy(np.array(support_y)).unsqueeze(1).float()

        query_x = torch.from_numpy(np.array(query_x)).unsqueeze(1).float()
        query_y = torch.from_numpy(np.array(query_y)).unsqueeze(1).long() if query_y else None

        return {
            "support_x": support_x,
            "support_y": support_y,
            "query_x": query_x,
            "query_y": query_y
        }