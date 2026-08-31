import torch
import torch.nn as nn


class BEVProjector(nn.Module):
    """Pools the 80x180 BEV grid down to 20x45 (900 tokens, same 4:9
    aspect ratio, no distortion) and projects each 512-dim spatial
    cell into the LLM's embedding dimension. LayerNorm after the
    projection mirrors LiDAR-LLM's own vision_proj_norm.

    `gate` mirrors LiDAR-LLM's LLaMA-Adapter-style gate, but with a
    small non-zero init (0.01) rather than 0.0 — see the note in
    __init__. Training moves it further as the projector learns."""
    def __init__(self, bev_channels=512, llm_hidden_size=None, pool_size=(20, 45)):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.proj = nn.Linear(bev_channels, llm_hidden_size)
        self.norm = nn.LayerNorm(llm_hidden_size)
        # Small NON-ZERO init, not 0.0: an exactly-zero gate makes every
        # BEV token an all-zero vector, and Qwen3's RMSNorm backward is
        # 0/0 on a zero-magnitude row -> NaN grads (forward stays finite,
        # which is why this only ever broke training). Bisect confirmed:
        # zero-valued prefix NaNs at any length, random prefix is fine.
        # 0.01 keeps the initial BEV contribution negligible, as intended.
        self.gate = nn.Parameter(torch.full((1,), 0.01))

    def forward(self, bev_feat):
        # Defend against non-finite values from the untrained SECOND
        # backbone (NaN/Inf can appear from an uncalibrated random
        # network) — must happen before anything else, since mean/std
        # of a tensor containing even one NaN is itself NaN, which
        # would make the standardization below propagate the problem
        # instead of fixing it.
        bev_feat = torch.nan_to_num(bev_feat, nan=0.0, posinf=0.0, neginf=0.0)

        pooled = self.pool(bev_feat)                    # [B, 512, 20, 45]
        B, C, H, W = pooled.shape
        tokens = pooled.flatten(2).transpose(1, 2)       # [B, 900, 512]
        # Standardize each token to zero mean / unit variance before
        # projecting — via F.layer_norm, whose eps sits INSIDE the
        # sqrt (sqrt(var+eps)): backward-safe even for all-zero tokens
        # from empty BEV regions. A manual (x-mean)/(std+eps) is NOT
        # safe there: torch.std's backward divides by std, and at
        # std=0 that's 0/0 = NaN gradients (forward stays fine, which
        # is exactly why only backward failed, only on scenes with a
        # fully-empty pooled cell).
        tokens = torch.nn.functional.layer_norm(tokens, tokens.shape[-1:])
        return self.gate * self.norm(self.proj(tokens))  # [B, 900, hidden_size]
