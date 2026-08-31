"""
Quick diagnostic: checks every saved BEV feature tensor for NaN, Inf,
or extreme magnitude values. Run from the K-Radar repo root (same
folder as bev_features/), after llm_exp.py has produced the files.
"""
import torch
from sample_config import SAMPLE_INDICES

for idx in SAMPLE_INDICES:
    t = torch.load(f'bev_features/sample_{idx}.pt')
    print(
        idx,
        "min:", round(t.min().item(), 3),
        "max:", round(t.max().item(), 3),
        "has_nan:", torch.isnan(t).any().item(),
        "has_inf:", torch.isinf(t).any().item(),
    )
