"""
Shared sample indices for the K-Radar experiment.

Both llm_exp.py (BEV feature extraction) and generate_qa.py (Q&A
generation) import SAMPLE_INDICES from here, so the BEV feature saved
for a given frame and the Q&A pair generated for that same frame are
guaranteed to refer to the identical scene — required once these get
paired up for Phase 5's training loop.

Spread across the sequence (every 15th frame) rather than consecutive
indices — Sequence 1's consecutive frames are near-duplicates (same
drive, fractions of a second apart), which gave near-identical Q&A
answers when we first tried indices [0..49].

Test split has 299 samples (seq 1); adjust MAX_INDEX if you point at
a different config/sequence with a different split size.
"""

MAX_INDEX = 299
STEP = 3

SAMPLE_INDICES = list(range(0, MAX_INDEX, STEP))  # ~100 frames
