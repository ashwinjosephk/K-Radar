"""Does the TRAINED projector actually use the LiDAR?

Loads projector_trained.pt and answers each test-scene question three
ways: with the real BEV tokens, with zeroed tokens, and with spatially
shuffled tokens. If all three give the same answer, the model is
ignoring the LiDAR and answering from learned priors alone.

Uses the same test scenes as train_projector.py (same seed/split), so
these are genuinely held-out.
"""
import torch.utils._pytree as _pytree
if not hasattr(_pytree, 'register_pytree_node'):
    def _shim(cls, fl, ufl, **kw):
        kw.pop('serialized_type_name', None)
        return _pytree._register_pytree_node(cls, fl, ufl)
    _pytree.register_pytree_node = _shim

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bev_projector import BEVProjector
from fusion_utils import embed_text, fuse_bev_and_text_embeds
from train_projector import (MODEL_NAME, MODEL_DTYPE, SAVE_PATH, task_of,
                             split_scenes, load_triples)


def answer(tokenizer, model, projector, bev_feat, question, mode):
    with torch.no_grad():
        tokens = projector(bev_feat.to("cuda"))
        if mode == "zeroed":
            tokens = torch.zeros_like(tokens)
        elif mode == "shuffled":
            tokens = tokens[:, torch.randperm(tokens.shape[1]), :]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        _, text_embeds = embed_text(tokenizer, model, prompt)
        emb, mask = fuse_bev_and_text_embeds(tokens.to(model.dtype), text_embeds)
        out = model.generate(inputs_embeds=emb, attention_mask=mask,
                             max_new_tokens=10, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=MODEL_DTYPE, attn_implementation="eager").cuda()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    projector = BEVProjector(512, model.config.hidden_size).to("cuda")
    projector.load_state_dict(torch.load(SAVE_PATH))
    projector.eval()
    print(f"Loaded {SAVE_PATH}, gate = {projector.gate.item():+.5f}\n")

    _, _, test_idx = split_scenes()
    triples = load_triples(test_idx)

    from collections import defaultdict
    stats = defaultdict(lambda: {"real": 0, "shuf": 0, "same": 0, "n": 0})

    def hit(pred, gt):
        g = gt.strip().rstrip(".").lower()
        p = pred.lower()
        return p.startswith(g) or g in p[:60]

    for idx, bev_feat, question, gt in triples:
        real = answer(tokenizer, model, projector, bev_feat, question, "real")
        zero = answer(tokenizer, model, projector, bev_feat, question, "zeroed")
        shuf = answer(tokenizer, model, projector, bev_feat, question, "shuffled")
        t = task_of(question)
        stats[t]["n"] += 1
        stats[t]["real"] += hit(real, gt)
        stats[t]["shuf"] += hit(shuf, gt)
        stats[t]["same"] += (real == shuf)
        print(f"idx {idx:3d} | {t:<13} | GT: {gt:<12} | real: {real:<12} | shuffled: {shuf:<12} | "
              f"{'same' if real == shuf else 'DIFFERS'}")

    print("\n--- real vs. spatially shuffled BEV tokens, per task ---")
    print("Shuffling keeps the same 900 token vectors but permutes their grid")
    print("positions. If accuracy holds up when shuffled, the model is using the")
    print("tokens as an unordered feature bag, NOT as a spatial map.")
    print("side_LR and front_back are the decisive ones: they are unanswerable")
    print("without knowing where the activation sits.\n")
    for t, v in stats.items():
        n = v["n"]
        print(f"{t:<14} real {v['real']:2d}/{n:2d} = {v['real']/n:.2f}   "
              f"shuffled {v['shuf']:2d}/{n:2d} = {v['shuf']/n:.2f}   "
              f"identical answer {v['same']}/{n}")
        drop = (v["real"] - v["shuf"]) / n
        if t in ("side_LR", "front_back"):
            if drop > 0.15:
                print(f"               ^ shuffling COSTS {drop:.2f} — real spatial grounding")
            else:
                print(f"               ^ shuffling costs only {drop:.2f} — no spatial grounding")

if __name__ == "__main__":
    main()
