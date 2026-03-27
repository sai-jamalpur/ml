import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Positional Encoding (Improved)
# =========================
def get_2d_positional_encoding(H, W, C, device):
    y = torch.linspace(0, 1, H, device=device)
    x = torch.linspace(0, 1, W, device=device)

    yy, xx = torch.meshgrid(y, x, indexing='ij')

    pos = torch.stack([
        torch.sin(2 * 3.1415 * xx),
        torch.cos(2 * 3.1415 * xx),
        torch.sin(2 * 3.1415 * yy),
        torch.cos(2 * 3.1415 * yy),
    ], dim=0)

    pos = pos.unsqueeze(0).repeat(1, C // 4, 1, 1)
    return pos


# =========================
# Encoder
# =========================
class GridEncoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()

        self.embed = nn.Embedding(10, dim)

        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B,1,H,W) → (B,H,W)
        x = x.squeeze(1).long()
        x = self.embed(x)  # (B,H,W,C)
        x = x.permute(0, 3, 1, 2)

        feat = self.conv(x)

        B, C, H, W = feat.shape
        pos = get_2d_positional_encoding(H, W, C, x.device)
        feat = feat + pos

        return feat


# =========================
# Task Encoder (Token Reduced)
# =========================
class TaskEncoder(nn.Module):
    def __init__(self, dim=128, pool_size=8):
        super().__init__()
        self.pool_size = pool_size

        self.proj = nn.Conv2d(dim * 2, dim, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, sx, sy):
        delta = sy - sx
        importance = torch.abs(delta).mean(dim=1, keepdim=True)

        combined = torch.cat([sx, delta], dim=1)
        tokens = self.proj(combined)

        # importance weighting
        tokens = tokens * importance

        # spatial reduction
        tokens = F.adaptive_avg_pool2d(tokens, (self.pool_size, self.pool_size))

        S, C, H, W = tokens.shape
        tokens = tokens.view(S, C, H * W).permute(0, 2, 1)

        tokens = tokens.reshape(-1, C)
        tokens = self.norm(tokens)

        return tokens


# =========================
# Cross Attention
# =========================
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, kv):
        attn_out, _ = self.attn(q, kv, kv)
        return self.norm(q + attn_out)


# =========================
# Decoder
# =========================
class Decoder(nn.Module):
    def __init__(self, dim=128, num_classes=10):
        super().__init__()

        self.pre_conv = nn.Conv2d(dim, dim, 3, padding=1)

        self.cross_attn = CrossAttentionBlock(dim)

        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, num_classes, 1)
        )

    def forward(self, query_feat, task_tokens):
        B, C, H, W = query_feat.shape

        # locality bias
        query_feat = self.pre_conv(query_feat)

        # flatten
        q = query_feat.view(B, C, H * W).permute(0, 2, 1)

        kv = task_tokens.unsqueeze(0).repeat(B, 1, 1)

        # cross attention
        q = self.cross_attn(q, kv)

        # reshape
        q = q.permute(0, 2, 1).view(B, C, H, W)

        # skip connection
        q = q + query_feat

        return self.conv(q)


# =========================
# ARC Model
# =========================
class ARCModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = GridEncoder()
        self.task_encoder = TaskEncoder()
        self.decoder = Decoder()

    def forward(self, support_x, support_y, query_x):
        sx = self.encoder(support_x)
        sy = self.encoder(support_y)

        task_tokens = self.task_encoder(sx, sy)

        qx = self.encoder(query_x)

        return self.decoder(qx, task_tokens)

    # =========================
    # Efficient Adaptation
    # =========================
    def forward_with_adaptation(
        self,
        support_x,
        support_y,
        query_x,
        steps=20,
        lr=1e-3,
        adapt_encoder=False
    ):
        model = self

        # cache features (BIG speedup)
        sx = model.encoder(support_x)
        sy = model.encoder(support_y)

        if adapt_encoder:
            params = model.parameters()
        else:
            params = model.decoder.parameters()

        optimizer = torch.optim.Adam(params, lr=lr)

        model.train()

        for _ in range(steps):
            optimizer.zero_grad()

            task_tokens = model.task_encoder(sx, sy)
            pred = model.decoder(sx, task_tokens)

            loss = F.cross_entropy(pred, support_y.squeeze(1).long())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            sx = model.encoder(support_x)
            sy = model.encoder(support_y)
            task_tokens = model.task_encoder(sx, sy)

            qx = model.encoder(query_x)
            return model.decoder(qx, task_tokens)