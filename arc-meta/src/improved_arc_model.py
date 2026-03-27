
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# =========================
# Positional Encoding (Improved)
# =========================
def get_2d_positional_encoding(H, W, C, device):
    y = torch.linspace(0, 1, H, device=device)
    x = torch.linspace(0, 1, W, device=device)

    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # Using more frequency components for richer positional information
    # This creates 8 channels for positional encoding
    pos = torch.stack([
        torch.sin(2 * 3.1415 * xx),
        torch.cos(2 * 3.1415 * xx),
        torch.sin(4 * 3.1415 * xx),
        torch.cos(4 * 3.1415 * xx),
        torch.sin(2 * 3.1415 * yy),
        torch.cos(2 * 3.1415 * yy),
        torch.sin(4 * 3.1415 * yy),
        torch.cos(4 * 3.1415 * yy),
    ], dim=0)

    # Adjusting for C // 8 as we now have 8 components per position
    # This repeats the 8-channel positional encoding to match the feature dimension C
    pos = pos.unsqueeze(0).repeat(1, C // 8, 1, 1)
    return pos


# =========================
# Encoder (Multi-scale features)
# =========================
class GridEncoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()

        self.embed = nn.Embedding(10, dim)

        # Multi-scale convolutional blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2), # Larger kernel for broader context
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 5, padding=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        # To fuse features from different scales, ensuring output dimension is 'dim'
        self.fusion_conv = nn.Conv2d(dim * 2, dim, 1) 

    def forward(self, x):
        # x: (B,1,H,W) → (B,H,W)
        x = x.squeeze(1).long()
        x = self.embed(x)  # (B,H,W,C)
        x = x.permute(0, 3, 1, 2) # (B,C,H,W)

        feat1 = self.conv_block1(x)
        feat2 = self.conv_block2(x)

        # Fuse multi-scale features
        feat = self.fusion_conv(torch.cat([feat1, feat2], dim=1))

        B, C, H, W = feat.shape
        # Positional encoding is added to the features
        pos = get_2d_positional_encoding(H, W, C, x.device)
        feat = feat + pos

        return feat


# =========================
# Task Encoder (with Self-Attention for better rule inference)
# =========================
class TaskEncoder(nn.Module):
    def __init__(self, dim=128, pool_size=8, heads=4):
        super().__init__()
        self.pool_size = pool_size
        self.dim = dim

        self.proj = nn.Conv2d(dim * 2, dim, 1)
        self.norm = nn.LayerNorm(dim)
        
        # Self-attention to capture relationships within task features
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(dim)

    def forward(self, sx, sy):
        delta = sy - sx
        # More nuanced importance weighting: combine delta and absolute difference
        # This aims to highlight areas of change and their magnitude
        importance = (torch.abs(delta) + torch.abs(sy - sx)).mean(dim=1, keepdim=True)

        combined = torch.cat([sx, delta], dim=1)
        tokens = self.proj(combined)

        # importance weighting
        tokens = tokens * importance

        # spatial reduction to a fixed pool_size
        tokens = F.adaptive_avg_pool2d(tokens, (self.pool_size, self.pool_size))

        S, C, H, W = tokens.shape
        tokens = tokens.view(S, C, H * W).permute(0, 2, 1) # (S, H*W, C) for attention
        
        # Apply self-attention to task tokens to capture inter-token relationships
        attn_output, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_output)

        tokens = tokens.reshape(-1, C) # Flatten for LayerNorm
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

        # flatten for attention
        q = query_feat.view(B, C, H * W).permute(0, 2, 1)

        # Expand task_tokens to match batch size for cross-attention
        kv = task_tokens.unsqueeze(0).repeat(B, 1, 1)

        # cross attention
        q = self.cross_attn(q, kv)

        # reshape back to spatial dimensions
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
    # Efficient Adaptation (Refined)
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

        # Cache features if encoder is not being adapted
        if not adapt_encoder:
            model.encoder.eval() # Ensure encoder is in eval mode
            with torch.no_grad():
                sx_cached = model.encoder(support_x)
                sy_cached = model.encoder(support_y)
        
        # Determine parameters to adapt
        if adapt_encoder:
            params = model.parameters()
        else:
            params = model.decoder.parameters()

        optimizer = torch.optim.Adam(params, lr=lr)

        model.train() # Set model to train mode for adaptation

        for _ in range(steps):
            optimizer.zero_grad()

            # Re-encode support features if encoder is being adapted
            if adapt_encoder:
                sx_current = model.encoder(support_x)
                sy_current = model.encoder(support_y)
            else:
                sx_current = sx_cached
                sy_current = sy_cached

            task_tokens = model.task_encoder(sx_current, sy_current)
            pred = model.decoder(sx_current, task_tokens)

            loss = F.cross_entropy(pred, support_y.squeeze(1).long())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

        model.eval() # Set model back to eval mode after adaptation
        with torch.no_grad():
            # Re-encode query_x and support features with potentially adapted encoder
            if adapt_encoder:
                qx_final = model.encoder(query_x)
                sx_final = model.encoder(support_x)
                sy_final = model.encoder(support_y)
                task_tokens_final = model.task_encoder(sx_final, sy_final)
            else:
                qx_final = model.encoder(query_x)
                task_tokens_final = model.task_encoder(sx_cached, sy_cached) # Use cached features

            return model.decoder(qx_final, task_tokens_final)


# =========================
# Example Usage (for testing purposes, not part of the model definition)
# =========================
if __name__ == '__main__':
    # Dummy data for testing
    B, C_in, H, W = 1, 1, 10, 10
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assuming input values are integers from 0-9
    support_x = torch.randint(0, num_classes, (B, C_in, H, W), dtype=torch.float32).to(device)
    support_y = torch.randint(0, num_classes, (B, C_in, H, W), dtype=torch.float32).to(device)
    query_x = torch.randint(0, num_classes, (B, C_in, H, W), dtype=torch.float32).to(device)

    model = ARCModel().to(device)

    print("Testing forward pass...")
    output = model(support_x, support_y, query_x)
    print(f"Output shape (forward pass): {output.shape}")

    print("Testing forward_with_adaptation (adapt_encoder=True)...")
    adapted_output = model.forward_with_adaptation(support_x, support_y, query_x, adapt_encoder=True)
    print(f"Output shape (adapted forward pass): {adapted_output.shape}")

    print("Testing forward_with_adaptation (adapt_encoder=False, decoder only)...")
    adapted_output_decoder_only = model.forward_with_adaptation(support_x, support_y, query_x, adapt_encoder=False)
    print(f"Output shape (adapted forward pass, decoder only): {adapted_output_decoder_only.shape}")

    print("Model architecture and forward passes tested successfully!")
