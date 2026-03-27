import torch
from torch.utils.data import DataLoader, Sampler
import random


# =========================
# Bucket Sampler (KEY)
# =========================
class BucketSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size

        # group indices by grid size
        self.buckets = {}
        for i, item in enumerate(dataset):
            h = item["support_x"].shape[-2]
            w = item["support_x"].shape[-1]

            key = (h // 5, w // 5)  # coarse bucketing
            if key not in self.buckets:
                self.buckets[key] = []
            self.buckets[key].append(i)

        self.batches = []
        for bucket in self.buckets.values():
            random.shuffle(bucket)
            for i in range(0, len(bucket), batch_size):
                self.batches.append(bucket[i:i + batch_size])

        random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


# =========================
# Padding
# =========================
def pad_to_max(tensors, max_h, max_w):
    out = []
    for t in tensors:
        _, h, w = t.shape
        pad_h = max_h - h
        pad_w = max_w - w
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h))
        out.append(t)
    return torch.stack(out)


# =========================
# Collate
# =========================
def collate_fn(batch):
    B = len(batch)

    max_s = max(b["support_x"].shape[0] for b in batch)
    max_q = max(b["query_x"].shape[0] for b in batch)

    max_h = max(
        max(b["support_x"].shape[-2], b["query_x"].shape[-2])
        for b in batch
    )
    max_w = max(
        max(b["support_x"].shape[-1], b["query_x"].shape[-1])
        for b in batch
    )

    support_x = torch.zeros(B, max_s, 1, max_h, max_w)
    support_y = torch.zeros(B, max_s, 1, max_h, max_w)

    query_x = torch.zeros(B, max_q, 1, max_h, max_w)
    query_y = torch.zeros(B, max_q, 1, max_h, max_w)

    support_mask = torch.zeros(B, max_s)
    query_mask = torch.zeros(B, max_q)

    for i, b in enumerate(batch):
        s = b["support_x"].shape[0]
        q = b["query_x"].shape[0]

        support_x[i, :s] = pad_to_max(b["support_x"], max_h, max_w)
        support_y[i, :s] = pad_to_max(b["support_y"], max_h, max_w)

        query_x[i, :q] = pad_to_max(b["query_x"], max_h, max_w)
        query_y[i, :q] = pad_to_max(b["query_y"], max_h, max_w)

        support_mask[i, :s] = 1
        query_mask[i, :q] = 1

    return {
        "support_x": support_x,
        "support_y": support_y,
        "query_x": query_x,
        "query_y": query_y,
        "support_mask": support_mask,
        "query_mask": query_mask
    }


def get_arc_loader(dataset, batch_size=8):
    sampler = BucketSampler(dataset, batch_size)

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )