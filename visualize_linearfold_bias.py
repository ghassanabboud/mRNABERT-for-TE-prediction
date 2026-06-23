"""
Visualizes the LinearFold secondary-structure bias for sequences in a CSV.

Loads the pre-computed .npz (produced by generate_linearfold_bias.py) and shows:
  1. The raw (K, 3) token-pair array for a chosen transcript
  2. The (L, L) bias matrix as produced by LinearFoldDataCollator
  3. A summary of the token sequence with pair annotations

Run from the repo root:
    python visualize_linearfold_bias.py
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from train_biased_head import LinearFoldDataCollator, TxIdSupervisedDataset

MODEL_NAME = "YYLY66/mRNABERT"
CSV_PATH   = "processed_data_RiboNN/test_data/test.csv"
NPZ_PATH   = "processed_data_RiboNN/test_data/test.npz"

# How many sequences to display (set to None for all)
MAX_DISPLAY = 2

# ---------------------------------------------------------------------------
# Load tokenizer and NPZ archive (for raw pair inspection)
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, model_max_length=1024, padding_side="right",
    use_fast=True, trust_remote_code=True,
)
vocab = tokenizer.get_vocab()
id_to_tok = {v: k for k, v in vocab.items()}

pairs_lookup = dict(np.load(NPZ_PATH, allow_pickle=False))

df = pd.read_csv(CSV_PATH)
print(f"CSV rows: {len(df)}  |  NPZ keys: {len(pairs_lookup)}")

# ---------------------------------------------------------------------------
# Build dataset + collator exactly as the training script would
# ---------------------------------------------------------------------------

dataset   = TxIdSupervisedDataset(data_path=CSV_PATH, tokenizer=tokenizer)
collator  = LinearFoldDataCollator(tokenizer=tokenizer, bias_npz_path=NPZ_PATH)

# ---------------------------------------------------------------------------
# For each sequence, show the pair array and the bias matrix
# ---------------------------------------------------------------------------

shown = 0
for idx in range(len(dataset)):
    if MAX_DISPLAY is not None and shown >= MAX_DISPLAY:
        break

    tx_id     = str(df.iloc[idx]["tx_id"])
    seq_field = df.iloc[idx]["sequence"]
    tokens    = seq_field.split()

    # Tokenize (adds CLS and SEP)
    enc        = tokenizer(seq_field, return_tensors="pt", padding=False, truncation=True)
    input_ids  = enc["input_ids"][0]
    token_strs = [id_to_tok[i.item()] for i in input_ids]
    L          = len(token_strs)

    pairs = pairs_lookup.get(tx_id)

    print("\n" + "=" * 70)
    print(f"tx_id : {tx_id}")
    print(f"tokens (excl. CLS/SEP): {tokens}")
    print(f"token sequence (incl. CLS/SEP, L={L}):")
    for pos, (tid, tok) in enumerate(zip(input_ids.tolist(), token_strs)):
        print(f"  pos {pos:2d}  id {tid:3d}  '{tok}'")

    # --- Raw pair array from NPZ ---
    print(f"\nRaw pair array from NPZ (shape {None if pairs is None else pairs.shape}):")
    if pairs is None:
        print("  (no entry in NPZ for this tx_id)")
    elif len(pairs) == 0:
        print("  (empty — no base pairs found)")
    else:
        print(f"  {'t_i':>4}  {'t_j':>4}  {'count':>5}  {'token_i (pos t_i+1)':>22}  {'token_j (pos t_j+1)':>22}")
        for ti, tj, cnt in pairs:
            tok_i = token_strs[ti + 1] if (ti + 1) < L else "?"
            tok_j = token_strs[tj + 1] if (tj + 1) < L else "?"
            print(f"  {ti:4d}  {tj:4d}  {cnt:5d}  {tok_i:>22}  {tok_j:>22}")

    # --- Run through collator (the same path as training) ---
    # dataset[idx] returns a dict with tx_id included
    instance = dataset[idx]
    batch = collator([instance])   # collator pops tx_id and returns bio_prior

    bio_prior = batch["bio_prior"]   # (1, 1, L, L)
    bias      = bio_prior[0, 0]      # (L, L)

    print(f"\nbio_prior from collator shape: {bio_prior.shape}  (B=1, heads=1, L={L}, L={L})")

    # --- Print bias matrix ---
    print(f"\nBias matrix  (rows=query pos, cols=key pos):")
    print(f"  {'':8s}", end="")
    for j, tok in enumerate(token_strs):
        print(f"  {tok:>5s}", end="")
    print()
    for i, row_tok in enumerate(token_strs):
        print(f"  {i:2d} {row_tok:>4s} ", end="")
        for j in range(L):
            v = bias[i, j].item()
            print(f"  {v:5.0f}", end="")
        print()

    # --- Nonzero entries summary ---
    print("\nNonzero entries (query pos ↔ key pos : count):")
    found_any = False
    for i in range(L):
        for j in range(i + 1, L):
            v = bias[i, j].item()
            if v != 0:
                print(f"  pos {i:2d} '{token_strs[i]}' ↔ pos {j:2d} '{token_strs[j]}' : {v:.0f}")
                found_any = True
    if not found_any:
        print("  (none)")

    shown += 1

print("\nDone.")