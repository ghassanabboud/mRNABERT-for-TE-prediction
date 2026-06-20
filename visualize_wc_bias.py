"""
Visualizes the Watson-Crick lookup matrix and shows the bias produced
by BiasedDataCollator for a small example sequence.

Run from the repo root:
    python visualize_wc_bias.py
"""

import torch
import pandas as pd
from transformers import AutoTokenizer

from train_biased_head import BiasedDataCollator, build_wc_lookup

MODEL_NAME = "YYLY66/mRNABERT"

# ---------------------------------------------------------------------------
# 1. Tokenizer vocabulary
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, model_max_length=1024, padding_side="right",
    use_fast=True, trust_remote_code=True,
)

vocab = tokenizer.get_vocab()
id_to_tok = {v: k for k, v in vocab.items()}

print("=" * 60)
print("1. TOKENIZER VOCABULARY")
print("=" * 60)
for tok_id in sorted(id_to_tok):
    print(f"  {tok_id:3d}  {id_to_tok[tok_id]}")

# ---------------------------------------------------------------------------
# 2. Full WC lookup matrix
# ---------------------------------------------------------------------------

wc = build_wc_lookup(tokenizer, utr_only=True)
V = tokenizer.vocab_size
tokens = [id_to_tok[i] for i in range(V)]

df_wc = pd.DataFrame(wc.numpy(), index=tokens, columns=tokens)

print("\n" + "=" * 60)
print("2. WATSON-CRICK LOOKUP MATRIX  (rows=query token, cols=key token)")
print("=" * 60)
# Show only nonzero rows/cols to keep output readable
nonzero_mask = (df_wc != 0).any(axis=1)
df_nonzero = df_wc.loc[nonzero_mask, nonzero_mask]
df_nonzero = df_nonzero.iloc[:20, :20]  # limit to first 20 rows/cols
with pd.option_context("display.max_columns", None, "display.width", 200,
                       "display.float_format", "{:.0f}".format):
    print(df_nonzero.to_string())

print(f"\nFull matrix shape: {wc.shape}  |  nonzero entries: {(wc != 0).sum().item()}")

# ---------------------------------------------------------------------------
# 3. Example bias for a small mixed UTR + CDS sequence
# ---------------------------------------------------------------------------
# Format mirrors the CSV: space-separated nucleotides for UTR, codons for CDS.
# 5'UTR: A T G   |  CDS: ATG GCG TAA  |  3'UTR: C G
EXAMPLE_SEQ = "A T G ATG GCG TAA C G"

print("\n" + "=" * 60)
print("3. EXAMPLE SEQUENCE THROUGH BiasedDataCollator")
print("=" * 60)
print(f"  Raw sequence: '{EXAMPLE_SEQ}'\n")

enc = tokenizer(
    EXAMPLE_SEQ,
    return_tensors="pt",
    padding=False,
    truncation=True,
)
input_ids = enc["input_ids"][0]  # (L,)

token_strs = [id_to_tok[i.item()] for i in input_ids]
print("  Token IDs and strings after tokenization:")
for pos, (tid, tok) in enumerate(zip(input_ids.tolist(), token_strs)):
    print(f"    pos {pos:2d} → id {tid:3d}  '{tok}'")

# Run through collator (expects list of dataset items)
collator = BiasedDataCollator(tokenizer=tokenizer, wc_lookup=wc)
dummy_instance = {"input_ids": input_ids, "labels": torch.zeros(1)}
batch = collator([dummy_instance])

bias = batch["bio_prior"][0, 0]  # (L, L) — drop batch and head dims
L = bias.shape[0]

print(f"\n  bio_prior tensor shape: {batch['bio_prior'].shape}  (B=1, heads=1, L={L}, L={L})")
print(f"\n  Bias matrix  (rows=query position, cols=key position):")
print(f"  {'':6s}", end="")
for j, tok in enumerate(token_strs):
    print(f"  {tok:>5s}", end="")
print()
for i, row_tok in enumerate(token_strs):
    print(f"  {i:2d} {row_tok:>3s} ", end="")
    for j in range(L):
        v = bias[i, j].item()
        print(f"  {v:5.0f}", end="")
    print()

print("\n  Nonzero entries (query → key : score):")
for i in range(L):
    for j in range(L):
        v = bias[i, j].item()
        if v != 0:
            print(f"    pos {i} '{token_strs[i]}' → pos {j} '{token_strs[j]}' : {v:.0f}")
