"""
K-Radar -> LiDAR BEV -> Qwen3 fusion experiment
=================================================
Structured in stages matching the reproduction plan:
  STAGE 0  - environment shim (must run before anything else)
  STAGE 1  - Phase 1: LiDAR -> BEV encoder (SECOND, K-Radar)
  STAGE 2  - Phase 1 sanity check: visualize raw point cloud vs. learned BEV feature
  STAGE 3  - Phase 2: load frozen Qwen3, ungrounded baseline (no LiDAR data reaches it)
  STAGE 4  - Phase 3: BEV projector + fused generation (LiDAR tokens prefixed to the prompt)

Each stage prints/saves something so you can confirm it worked before moving to the next.
"""

# ============================================================
# STAGE 0 — environment shim
# ============================================================
# Fixes a known transformers/torch<2.2 bug (AttributeError on
# torch.utils._pytree.register_pytree_node). Must run before any
# import that could transitively pull in `transformers` — including
# K-Radar's own imports below, which drag it in via torchvision ->
# torch.onnx internals. See conversation history for the full trace.
import torch.utils._pytree as _pytree
if not hasattr(_pytree, 'register_pytree_node'):
    def _register_pytree_node_shim(cls, flatten_fn, unflatten_fn, **kwargs):
        kwargs.pop('serialized_type_name', None)   # newer-API-only kwarg, safe to drop
        return _pytree._register_pytree_node(cls, flatten_fn, unflatten_fn)
    _pytree.register_pytree_node = _register_pytree_node_shim

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from sample_config import SAMPLE_INDICES


# ============================================================
# STAGE 1 — Phase 1: LiDAR -> BEV encoder
# ============================================================
# What this proves: raw K-Radar LiDAR point clouds can be pushed through
# SECOND's backbone (VoxelBackBone8x -> HeightCompression -> BaseBEVBackbone)
# to produce a spatially-grounded [1, 512, 80, 180] feature tensor.
# Network uses K-Radar's pretrained LODN_SECOND weights, so the BEV
# features carry real object semantics (earlier runs used random init).
from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

PATH_CONFIG = './configs/cfg_SECOND.yml'
PATH_LODN_CKPT = './LODN_model_log/LODN_SECOND.pt'   # trained SECOND weights

pline = PipelineDetection_v1_0(PATH_CONFIG, mode='vis')
# Load K-Radar's pretrained LiDAR detector (from the auto-labeling
# project's LODN_model_log). Without this, SECOND is randomly
# initialized and its BEV features carry no object semantics —
# which is why the projector had nothing meaningful to learn from.
pline.load_dict_model(PATH_LODN_CKPT)
pline.network.eval()

os.makedirs('bev_features', exist_ok=True)
subset = Subset(pline.dataset_test, SAMPLE_INDICES)
data_loader = torch.utils.data.DataLoader(
    subset, batch_size=1, shuffle=False,
    collate_fn=pline.dataset_test.collate_fn, num_workers=0,
)

print("\n=== STAGE 1: extracting BEV features ===")
last_batch_dict = None   # keep an explicit handle instead of relying on loop-variable leakage
with torch.no_grad():
    for i, batch_dict in enumerate(data_loader):
        real_idx = SAMPLE_INDICES[i]   # loop position != dataset index — save by the real one
        net = pline.network
        batch_dict = net.pre_processor(batch_dict)
        batch_dict = net.vfe(batch_dict)
        batch_dict = net.backbone_3d(batch_dict)
        batch_dict = net.map_to_bev_module(batch_dict)
        batch_dict = net.backbone_2d(batch_dict)
        torch.save(batch_dict['spatial_features_2d'].cpu(), f'bev_features/sample_{real_idx}.pt')
        print(f"  dataset idx {real_idx}: {batch_dict['spatial_features_2d'].shape}")
        last_batch_dict = batch_dict

print("Stage 1 done: BEV feature tensors saved to bev_features/*.pt\n")


# ============================================================
# STAGE 2 — Phase 1 sanity check: visualize raw point cloud vs. BEV feature
# ============================================================
# What this proves: the learned BEV feature is spatially consistent with
# the real LiDAR geometry it came from (even though weights are random) —
# not a wiring bug like a swapped axis or wrong tensor.
print("=== STAGE 2: visual sanity check ===")

pc = pline.dataset_test.get_ldr64_from_path(last_batch_dict['meta'][0]['path']['ldr64'])  # Nx(x,y,z,...)

plt.figure(figsize=(8, 8))
plt.hist2d(pc[:, 0], pc[:, 1], bins=300, range=[[0, 72], [-16, 16]], cmap='viridis')
plt.xlabel('x (forward, m)'); plt.ylabel('y (lateral, m)')
plt.title('Raw LiDAR point cloud (BEV density)')
plt.savefig('raw_lidar_bev.png', dpi=150)
plt.close()

feat = last_batch_dict['spatial_features_2d'][0].detach().cpu().numpy()  # [512, 80, 180]
heatmap = feat.mean(axis=0)  # collapse 512 channels -> [80, 180], for visualization only
plt.figure(figsize=(8, 4))
plt.imshow(heatmap, cmap='viridis', origin='lower')
plt.colorbar()
plt.title('Learned BEV feature (channel-mean, LODN_SECOND weights)')
plt.savefig('bev_feature_heatmap.png', dpi=150)
plt.close()

print("Stage 2 done: raw_lidar_bev.png + bev_feature_heatmap.png saved\n")


# ============================================================
# STAGE 3 — Phase 2: load frozen Qwen3, ungrounded baseline
# ============================================================
# What this proves: the LLM loads and generates correctly in this same
# environment. The output here is a *baseline* — Qwen3 has no access to
# any LiDAR data at this point, it's answering purely from pretraining.
# Keep this output around; it's the control you compare Stage 4 against.
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=== STAGE 3: loading Qwen3 (ungrounded baseline) ===")

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).cuda()
model.eval()

baseline_messages = [{"role": "user", "content": "Describe what a LiDAR point cloud looks like from above."}]
baseline_text = tokenizer.apply_chat_template(
    baseline_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
baseline_inputs = tokenizer(baseline_text, return_tensors="pt").to("cuda")

with torch.no_grad():
    baseline_out = model.generate(**baseline_inputs, max_new_tokens=100)

baseline_response = tokenizer.decode(
    baseline_out[0][baseline_inputs['input_ids'].shape[1]:], skip_special_tokens=True
)
print("Baseline (ungrounded) response:")
print(baseline_response)
print(f"GPU memory after Stage 3: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print("Stage 3 done\n")


# ============================================================
# STAGE 4 — Phase 3: BEV projector + fused generation
# ============================================================
# What this proves (mechanically): the BEV feature tensor can be pooled,
# projected into Qwen3's embedding space, and prefixed onto a real prompt
# without shape/dtype errors, producing *some* generated text.
# What this does NOT prove yet: the projector is randomly initialized,
# so the model has not learned to use the LiDAR tokens meaningfully.
# The real test comes later: compare this output against the same run
# with zeroed/shuffled BEV tokens (Phase 5's sanity check) — if they're
# indistinguishable, the fusion isn't doing anything yet.
import torch.nn as nn

print("=== STAGE 4: BEV projector + fused generation ===")

from bev_projector import BEVProjector
from fusion_utils import embed_text, fuse_bev_and_text_embeds


hidden_size = model.config.hidden_size
projector = BEVProjector(512, hidden_size).to('cuda', dtype=model.dtype)

# reuse the BEV feature already extracted in Stage 1 — no need to recompute
bev_feat = last_batch_dict['spatial_features_2d'].to('cuda', dtype=model.dtype)  # [1, 512, 80, 180]
bev_tokens = projector(bev_feat)   # [1, 900, hidden_size]

fused_messages = [{"role": "user", "content": "Describe what you observe in this LiDAR scene."}]
fused_text = tokenizer.apply_chat_template(
    fused_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
_, text_embeds = embed_text(tokenizer, model, fused_text)
combined_embeds, combined_mask = fuse_bev_and_text_embeds(bev_tokens, text_embeds)

with torch.no_grad():
    fused_out = model.generate(inputs_embeds=combined_embeds, attention_mask=combined_mask, max_new_tokens=100)

# no slicing here: inputs_embeds has no discrete input_ids to measure prompt
# length against, so generate() already returns only the new tokens
fused_response = tokenizer.decode(fused_out[0], skip_special_tokens=True)
print("Fused (BEV-conditioned) response:")
print(fused_response)
print(f"GPU memory after Stage 4: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print("Stage 4 done\n")

print("=== Summary ===")
print("Baseline (no LiDAR):", baseline_response[:120].replace("\n", " "), "...")
print("Fused (with LiDAR): ", fused_response[:120].replace("\n", " "), "...")


# ============================================================
# STAGE 5 — sanity check: real vs. zeroed vs. shuffled BEV tokens
# ============================================================
# This is the actual test for whether the fusion is doing anything
# LiDAR-specific, as opposed to just reacting to "some prepended
# vectors" in general (which is all Stage 4 alone can show).
# Same idea as Radar4D-VLM's own control: if real/zeroed/shuffled
# all produce equally degenerate or equally coherent output, the
# fusion isn't extracting anything from the geometry yet — expected
# at this stage (projector is untrained), but important to confirm
# directly rather than assume.
print("=== STAGE 5: real vs. zeroed vs. shuffled BEV tokens ===")

def generate_with_bev_tokens(bev_tokens, label):
    _, text_embeds_ = embed_text(tokenizer, model, fused_text)
    combined_embeds_, combined_mask_ = fuse_bev_and_text_embeds(bev_tokens, text_embeds_)
    with torch.no_grad():
        out_ = model.generate(inputs_embeds=combined_embeds_, attention_mask=combined_mask_, max_new_tokens=100)
    response_ = tokenizer.decode(out_[0], skip_special_tokens=True)
    print(f"\n[{label}]")
    print(response_)
    return response_

# real tokens (same as Stage 4, recomputed here for clarity)
real_tokens = projector(bev_feat)

# zeroed: same shape, no information at all
zeroed_tokens = torch.zeros_like(real_tokens)

# shuffled: same values, spatial positions permuted — destroys geometric
# structure but keeps the same per-token statistics as the real feature
perm = torch.randperm(real_tokens.shape[1])
shuffled_tokens = real_tokens[:, perm, :]

resp_real = generate_with_bev_tokens(real_tokens, "REAL BEV tokens")
resp_zero = generate_with_bev_tokens(zeroed_tokens, "ZEROED BEV tokens")
resp_shuf = generate_with_bev_tokens(shuffled_tokens, "SHUFFLED BEV tokens")

print("\n=== Stage 5 comparison ===")
print("Real:     ", resp_real[:100].replace("\n", " "))
print("Zeroed:   ", resp_zero[:100].replace("\n", " "))
print("Shuffled: ", resp_shuf[:100].replace("\n", " "))
print("\nIf all three look similarly (in)coherent, that's expected right now —")
print("the projector hasn't been trained to extract anything geometry-specific yet.")
print("This becomes the meaningful check once you actually train the projector.")
