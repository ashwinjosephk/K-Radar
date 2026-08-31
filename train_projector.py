"""
Phase 5: train the BEV projector against K-Radar Q&A pairs.
Frozen: SECOND (features pre-saved), Qwen3 (verified 0 trainable).
Trainable: BEVProjector only (fp32; output cast to bf16 at fusion).

Hardening in this version:
- Loss computed MANUALLY in fp32 (not the model's internal bf16 path).
- Non-finite loss or grads => step SKIPPED and logged, so one bad step
  can no longer corrupt the parameters permanently (the old NaN death).
- 3-way split by scene: train/val/test. Test eval = real generation
  (exact-match), not teacher-forced accuracy.
"""

import torch.utils._pytree as _pytree
if not hasattr(_pytree, 'register_pytree_node'):
    def _shim(cls, fl, ufl, **kw):
        kw.pop('serialized_type_name', None)
        return _pytree._register_pytree_node(cls, fl, ufl)
    _pytree.register_pytree_node = _shim

import json
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sample_config import SAMPLE_INDICES
from bev_projector import BEVProjector
from fusion_utils import embed_text, fuse_bev_and_text_embeds

MODEL_NAME = "Qwen/Qwen3-1.7B"
# fp32 on purpose: bf16 backward through the frozen stack on
# torch 2.1.2 + transformers 4.57 is the prime suspect for the
# universal non-finite grads. ~6.8GB weights — fits 10GB now that
# the sequence is only ~260 tokens. If this OOMs, set back to
# torch.bfloat16 and send me the [grad-diagnosis] output instead.
MODEL_DTYPE = torch.bfloat16
# Gradient checkpointing rewrites backward (recompute instead of store)
# and on torch 2.1 defaults to the reentrant implementation, which is
# the last remaining non-standard thing in the model's backward path —
# and grad-diagnosis proved the NaN is born there. Off by default now.
# If this OOMs: set MODEL_DTYPE=torch.bfloat16 first (keep this False).
USE_GRAD_CHECKPOINTING = True
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4   # lowered from 2e-4: loss spikes mid-training (e.g. 0.26 -> 0.86)
                       # indicate steps too large for this small, unstable setup
MAX_GRAD_NORM = 1.0
SEED = 42
BEV_FEATURES_DIR = "bev_features"
QA_PATH = "kradar_qa.json"
SAVE_PATH = "projector_trained.pt"


GAP = 3   # scenes discarded either side of held-out blocks (adjacency buffer)


def split_scenes(seed=SEED):
    """Middle-block split with gap buffers.

    Problem with a random split: at STEP=3 frames are ~0.3s apart, so a
    randomly held-out frame sits between two training frames of nearly
    the same scene — the model scores well by recognizing near-duplicates.

    Problem with an end-block split (train 0-207, test 255-297): this
    sequence has two traffic regimes — sparse early (mostly "No",
    counts 0-1) and busy late ("Yes", counts 2-3). Training only on the
    early block and testing on the late one asks the model about a
    situation it never saw, so it falls back on priors.

    This split takes val and test from the MIDDLE, so training covers
    both the early and late phases, while the held-out blocks stay
    contiguous and non-adjacent (GAP scenes dropped either side).

    Layout: [train ... | gap | VAL | gap | TEST | gap | ... train]
    `seed` unused, kept so callers don't break.
    """
    idxs = sorted(SAMPLE_INDICES)
    n = len(idxs)
    n_val = n_test = max(1, int(n * 0.15))

    val_start = int(n * 0.40)
    val_end = val_start + n_val
    test_start = val_end + GAP
    test_end = test_start + n_test

    val = set(idxs[val_start:val_end])
    test = set(idxs[test_start:test_end])
    # training excludes the held-out blocks AND the gap frames around them
    train = set(idxs[:val_start - GAP]) | set(idxs[test_end + GAP:])
    return train, val, test


def load_triples(idx_set):
    with open(QA_PATH) as f:
        qa_data = json.load(f)
    triples = []
    for entry in qa_data:
        idx = entry["idx"]
        if idx not in idx_set:
            continue
        bev_feat = torch.load(f"{BEV_FEATURES_DIR}/sample_{idx}.pt")
        for qa in entry["qa_pairs"]:
            triples.append((idx, bev_feat, qa["question"], qa["answer"]))
    return triples


def build_labeled_sequence(tokenizer, model, bev_tokens, question, answer):
    """inputs_embeds + labels with loss masked to answer tokens only."""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False)

    prompt_ids, _ = embed_text(tokenizer, model, prompt_text)
    full_ids, full_embeds = embed_text(tokenizer, model, full_text)
    prompt_len = prompt_ids.shape[1]
    assert torch.equal(full_ids[:, :prompt_len], prompt_ids), \
        "chat template prefix mismatch — inspect manually"

    combined_embeds, attention_mask = fuse_bev_and_text_embeds(bev_tokens, full_embeds)
    n_bev = bev_tokens.shape[1]
    labels = torch.cat([
        torch.full((1, n_bev + prompt_len), -100, device="cuda"),
        full_ids[:, prompt_len:],
    ], dim=1)
    return combined_embeds, attention_mask, labels


def fp32_loss(logits, labels):
    """Manual causal-LM loss in fp32, computed ONLY on the answer span
    (answer tokens are contiguous at the sequence tail) — upcasting the
    full [1, seq, vocab] logits to fp32 was a ~570MB allocation per
    step and the direct cause of the OOM. Returns None if no valid
    label tokens (cross-entropy would be 0/0 = NaN)."""
    valid_pos = (labels[0] != -100).nonzero()
    if valid_pos.numel() == 0:
        return None
    start = valid_pos[0].item()
    shift_logits = logits[:, start - 1:-1, :].float()   # tiny slice, then upcast
    shift_labels = labels[:, start:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1), ignore_index=-100)


def token_accuracy(logits, labels):
    """Teacher-forced token accuracy on answer tokens (optimistic proxy)."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100
    if valid.sum() == 0:
        return None
    correct = (shift_logits.argmax(dim=-1) == shift_labels) & valid
    return correct.sum().item() / valid.sum().item()


def run_epoch(triples, tokenizer, model, projector, optimizer=None):
    is_train = optimizer is not None
    total_loss, total_acc, n_loss, n_acc, n_skipped = 0.0, 0.0, 0, 0, 0

    for idx, bev_feat, question, answer in tqdm(triples, desc="train" if is_train else "val  ", leave=False):
        bev_tokens = projector(bev_feat.to("cuda"))
        bev_tokens_cast = bev_tokens.to(model.dtype)
        if is_train:
            bev_tokens_cast.retain_grad()   # lets us see the gradient arriving FROM the frozen model
        combined_embeds, attention_mask, labels = build_labeled_sequence(
            tokenizer, model, bev_tokens_cast, question, answer)

        with torch.set_grad_enabled(is_train):
            out = model(inputs_embeds=combined_embeds, attention_mask=attention_mask)
            loss = fp32_loss(out.logits, labels)

        if loss is None or not torch.isfinite(loss):
            n_skipped += 1
            tqdm.write(f"[skip] idx={idx} loss={'None' if loss is None else loss.item()} q={question[:40]!r}")
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            continue

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grads_ok = all(torch.isfinite(p.grad).all() for p in projector.parameters() if p.grad is not None)
            if not grads_ok:
                # one-shot diagnosis: pinpoint where the non-finite grad first exists
                incoming_ok = bev_tokens_cast.grad is not None and torch.isfinite(bev_tokens_cast.grad).all().item()
                per_param = {n: torch.isfinite(p.grad).all().item()
                             for n, p in projector.named_parameters() if p.grad is not None}
                raise RuntimeError(
                    f"[grad-diagnosis] idx={idx}\n"
                    f"  gradient arriving from frozen Qwen3 into BEV tokens finite: {incoming_ok}\n"
                    f"  projector param grads finite: {per_param}\n"
                    f"  -> if incoming is False: the NaN is born INSIDE the frozen model's backward "
                    f"(model-side: dtype/kernel issue). If incoming is True but params False: "
                    f"the projector's own backward is at fault.")
            torch.nn.utils.clip_grad_norm_(projector.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            # hard stop if params ever go non-finite despite the guards
            for p in projector.parameters():
                if not torch.isfinite(p).all():
                    raise RuntimeError(f"projector params went non-finite after step at idx={idx}")

        total_loss += loss.item(); n_loss += 1
        acc = token_accuracy(out.logits, labels)
        if acc is not None:
            total_acc += acc; n_acc += 1

    avg_loss = total_loss / n_loss if n_loss else float("nan")
    avg_acc = total_acc / n_acc if n_acc else float("nan")
    return avg_loss, avg_acc, n_skipped


def task_of(question):
    """Which template type a question belongs to."""
    if question.startswith("Is there"):
        return "existence"
    if question.startswith("How many"):
        return "counting"
    if "left or the right" in question:
        return "side_LR"
    if "ahead of or behind" in question:
        return "front_back"
    return "nearest_class"


def evaluate_generation(triples, tokenizer, model, projector, max_new_tokens=20):
    """Test eval: real autoregressive generation, exact-answer matching.

    Reports PER-TASK accuracy and compares each task against its
    constant-answer baseline (always predicting that task's most common
    ground-truth answer). This matters: if a test block's ground truth
    never varies, a model that answers the same thing every time scores
    100% on that task while demonstrating nothing. Overall accuracy
    alone hides that completely.
    """
    from collections import Counter, defaultdict
    projector.eval()

    per_task = defaultdict(lambda: {"correct": 0, "total": 0})
    gt_by_task = defaultdict(list)
    pred_by_task = defaultdict(list)

    for idx, bev_feat, question, answer in triples:
        with torch.no_grad():
            bev_tokens = projector(bev_feat.to("cuda")).to(model.dtype)
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            _, text_embeds = embed_text(tokenizer, model, prompt_text)
            emb, mask = fuse_bev_and_text_embeds(bev_tokens, text_embeds)
            out = model.generate(inputs_embeds=emb, attention_mask=mask,
                                 max_new_tokens=max_new_tokens, use_cache=True, do_sample=False)
        pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        gt = answer.strip().rstrip(".")
        ok = pred.lower().startswith(gt.lower()) or gt.lower() in pred.lower()[:60]

        t = task_of(question)
        per_task[t]["correct"] += ok
        per_task[t]["total"] += 1
        gt_by_task[t].append(gt)
        pred_by_task[t].append(pred[:20])

        print(f"  idx {idx:3d} | {t:<13} | GT: {answer:<16} | Pred: {pred[:40]!r} | {'OK' if ok else 'X '}")

    total_correct = sum(v["correct"] for v in per_task.values())
    total_n = sum(v["total"] for v in per_task.values())

    print("\n--- per-task breakdown ---")
    for t in ("existence", "counting", "nearest_class", "side_LR", "front_back"):
        if per_task[t]["total"] == 0:
            continue
        c, n = per_task[t]["correct"], per_task[t]["total"]
        gts = gt_by_task[t]
        # constant-answer baseline: always predict the most common GT
        majority, maj_count = Counter(gts).most_common(1)[0]
        n_distinct_gt = len(set(gts))
        n_distinct_pred = len(set(pred_by_task[t]))

        flags = []
        if n_distinct_gt == 1:
            flags.append(f"DEGENERATE TEST: every GT is '{majority}' — a constant answer scores 100%")
        if n_distinct_pred == 1:
            flags.append("model gave the SAME answer every time")
        if c < maj_count:
            flags.append("BELOW the constant-answer baseline")

        print(f"{t:<14} {c:2d}/{n:2d} = {c/n:.2f}   "
              f"baseline (always '{majority}'): {maj_count}/{n} = {maj_count/n:.2f}   "
              f"[{n_distinct_gt} distinct GT, {n_distinct_pred} distinct pred]")
        for f in flags:
            print(f"               ^ {f}")

    print(f"\nOverall exact-match: {total_correct}/{total_n} = {total_correct/total_n:.2f}")
    print("Read the per-task rows above, not this number — tasks with a")
    print("single repeated ground truth inflate it without showing grounding.")
    projector.train()


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=MODEL_DTYPE, attn_implementation="eager").cuda()
    model.eval()
    if USE_GRAD_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False          # through all frozen layers
    for p in model.parameters():
        p.requires_grad = False
    print(f"Qwen3 trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)} (should be 0)")

    projector = BEVProjector(512, model.config.hidden_size).to("cuda")  # fp32 on purpose
    print(f"Projector trainable params: {sum(p.numel() for p in projector.parameters())}")
    optimizer = torch.optim.AdamW(projector.parameters(), lr=LEARNING_RATE)

    train_idx, val_idx, test_idx = split_scenes()
    train_triples, val_triples, test_triples = load_triples(train_idx), load_triples(val_idx), load_triples(test_idx)
    print(f"Train: {len(train_triples)} triples, scenes {sorted(train_idx)}")
    print(f"Val:   {len(val_triples)} triples, scenes {sorted(val_idx)}")
    print(f"Test:  {len(test_triples)} triples, scenes {sorted(test_idx)}\n")

    best_val_loss = float("inf")
    best_epoch = -1
    for epoch in range(NUM_EPOCHS):
        tr_loss, tr_acc, tr_skip = run_epoch(train_triples, tokenizer, model, projector, optimizer)
        va_loss, va_acc, va_skip = run_epoch(val_triples, tokenizer, model, projector)
        torch.cuda.empty_cache()

        # Save the BEST checkpoint by val loss, not the last epoch's.
        # Training here is unstable (small dataset, single scalar gate),
        # so whichever epoch we happen to stop on is close to arbitrary —
        # epoch 20 can easily be much worse than epoch 18.
        is_best = va_loss < best_val_loss
        if is_best:
            best_val_loss, best_epoch = va_loss, epoch + 1
            torch.save(projector.state_dict(), SAVE_PATH)

        print(f"epoch {epoch+1:2d}/{NUM_EPOCHS}  train_loss: {tr_loss:.4f}  train_acc: {tr_acc:.3f}  "
              f"val_loss: {va_loss:.4f}  val_acc: {va_acc:.3f}  gate: {projector.gate.item():+.5f}"
              + ("  <- best, saved" if is_best else "")
              + (f"  [skipped: {tr_skip}+{va_skip}]" if (tr_skip or va_skip) else ""))

    # reload the best checkpoint before the test eval, so we evaluate
    # the model we actually saved rather than whatever the last epoch left
    projector.load_state_dict(torch.load(SAVE_PATH))
    print(f"\nSaved {SAVE_PATH} from epoch {best_epoch} (val_loss {best_val_loss:.4f}). "
          f"Gate: {projector.gate.item():+.5f}")

    print("\n=== TEST: generation-based eval on held-out scenes ===")
    evaluate_generation(test_triples, tokenizer, model, projector)


if __name__ == "__main__":
    main()
