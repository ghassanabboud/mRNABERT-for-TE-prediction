#!/usr/bin/env python3
"""
Pre-compute LinearFold secondary-structure bias at the token level.

For each sequence in the input CSV (space-separated tokens: single nucleotides for
UTR regions, 3-letter codons for CDS), this script:
  1. Reconstructs the nucleotide string (UTR nuc + CDS codons expanded).
  2. Runs LinearFold to obtain the MFE dot-bracket structure.
  3. Parses the dot-bracket into nucleotide-level base pairs.
  4. Maps pairs to token level: nucleotide pair (i, j) → token pair (t_i, t_j),
     where a codon token covers 3 nucleotides (OR logic: any constituent pair → token paired).
  5. Saves a (K, 2) int32 array of unique token-index pairs per sequence.

The collator (LinearFoldDataCollator in train_biased_head.py) reads these pairs and
builds the binary (B, 1, L, L) bias tensor on the fly by setting bias[t_i+1, t_j+1] = 1
(+1 for the CLS token prepended by the mRNABERT tokenizer).

Storage is O(T) per sequence (one pair per base pair) rather than O(T²) for a full matrix.

Usage:
    python generate_linearfold_bias.py path/to/train.csv
    python generate_linearfold_bias.py path/to/train.csv -o train_lf_bias.npz
    python generate_linearfold_bias.py path/to/train.csv --max-tokens 1022 --jobs 4
"""

import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

LINEARFOLD_EXECUTABLE = '../LinearFold/linearfold'  # must be in PATH

# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------

def parse_token_ranges(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Convert a list of mixed nucleotide/codon tokens into:
      - nuc_seq: concatenated nucleotide string (T→U for RNA folding)
      - token_ranges: list of (start, length) in nuc_seq for each token

    Single-character tokens (UTR) contribute 1 nucleotide.
    Three-character tokens (CDS codons) contribute 3 nucleotides.
    """
    parts = []
    token_ranges: List[Tuple[int, int]] = []
    pos = 0
    for tok in tokens:
        token_ranges.append((pos, len(tok)))
        parts.append(tok)
        pos += len(tok)
    nuc_seq = ''.join(parts).upper().replace('T', 'U')
    return nuc_seq, token_ranges


# ---------------------------------------------------------------------------
# LinearFold runner
# ---------------------------------------------------------------------------

def run_linearfold(nuc_seq: str) -> str:
    """
    Run LinearFold on `nuc_seq` and return the MFE dot-bracket structure string.

    LinearFold output format:
        SEQUENCE
        .(((...)))  (-1.23)

    Returns just the dot-bracket part (same length as nuc_seq).
    """
    result = subprocess.run(
        [LINEARFOLD_EXECUTABLE],
        input=nuc_seq,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Unexpected LinearFold output: {result.stdout[:200]}")
    dot_bracket = lines[1].strip().split()[0]
    if len(dot_bracket) != len(nuc_seq):
        raise ValueError(
            f"Dot-bracket length {len(dot_bracket)} != sequence length {len(nuc_seq)}"
        )
    return dot_bracket


# ---------------------------------------------------------------------------
# Token-level pair extraction
# ---------------------------------------------------------------------------

def dotbracket_to_token_pairs(
    dot_bracket: str,
    token_ranges: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Parse a nucleotide-level dot-bracket structure into token-level base pairs
    with interaction counts.

    Mapping rule: nucleotide pair (i, j) → token pair (t_i, t_j). For codon tokens
    (3 nucleotides each), multiple nucleotide pairs can map to the same token pair,
    giving a count up to 3. Single-nucleotide UTR tokens contribute at most 1.
    Self-pairs (both nucleotides inside the same codon) are discarded.

    Returns an int32 array of shape (K, 3): columns are [t_i, t_j, count], where
    t_i < t_j, count ∈ {1, 2, 3}, and indices are 0-based excluding CLS/SEP.
    The collator sets bias[t_i+1, t_j+1] = count (symmetric).
    """
    N = len(dot_bracket)

    # Build nucleotide → token index lookup
    nuc_to_tok = np.empty(N, dtype=np.int32)
    for t, (s, k) in enumerate(token_ranges):
        nuc_to_tok[s:s + k] = t

    # Parse dot-bracket with a stack; count nucleotide pairs per token pair
    stack: List[int] = []
    counts: Counter = Counter()
    for i, c in enumerate(dot_bracket):
        if c == '(':
            stack.append(i)
        elif c == ')':
            j = stack.pop()
            ti, tj = int(nuc_to_tok[j]), int(nuc_to_tok[i])
            if ti != tj:  # discard intra-token pairs
                counts[(min(ti, tj), max(ti, tj))] += 1

    if not counts:
        return np.empty((0, 3), dtype=np.int32)
    rows = [(ti, tj, cnt) for (ti, tj), cnt in sorted(counts.items())]
    return np.array(rows, dtype=np.int32)


# ---------------------------------------------------------------------------
# Per-sequence worker (runs in a subprocess when jobs > 1)
# ---------------------------------------------------------------------------

def process_one(
    tx_id: str, sequence_field: str, max_tokens: int
) -> Tuple[str, np.ndarray]:
    #print(f"Processing {tx_id} (max_tokens={max_tokens})")
    #print(f"Sequence field: {sequence_field}")
    tokens = sequence_field.split()
    #print(f"Parsed tokens: {tokens})")
    if max_tokens > 0:
        tokens = tokens[:max_tokens]
    nuc_seq, token_ranges = parse_token_ranges(tokens)
    #print(f"Nucleotide sequence: {nuc_seq}")
    #print(f"Token ranges: {token_ranges}")
    dot_bracket = run_linearfold(nuc_seq)
    #print(f"Dot-bracket: {dot_bracket}")
    pairs = dotbracket_to_token_pairs(dot_bracket, token_ranges)
    #print(f"Token pairs: {pairs}")
    return tx_id, pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_csv(csv_path: Path, out_path: Path, max_tokens: int, jobs: int) -> None:
    df = pd.read_csv(csv_path)

    missing = {'tx_id', 'sequence'} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    tx_ids = df['tx_id'].astype(str).tolist()
    sequences = df['sequence'].tolist()
    n = len(tx_ids)
    print(f"Processing {n} sequences from {csv_path.name}")
    if max_tokens > 0:
        print(f"Truncating to first {max_tokens} tokens per sequence")

    arrays: dict = {}
    skipped = 0

    if jobs == 1:
        for tx_id, seq in tqdm(zip(tx_ids, sequences), total=n):
            try:
                _, pairs = process_one(tx_id, seq, max_tokens)
                arrays[tx_id] = pairs
            except Exception as e:
                print(f"\nFailed {tx_id}: {e}")
                skipped += 1
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(process_one, tx_id, seq, max_tokens): tx_id
                for tx_id, seq in zip(tx_ids, sequences)
            }
            for fut in tqdm(as_completed(futures), total=n):
                tx_id = futures[fut]
                try:
                    _, pairs = fut.result()
                    arrays[tx_id] = pairs
                except Exception as e:
                    print(f"\nFailed {tx_id}: {e}")
                    skipped += 1

    np.savez_compressed(out_path, **arrays)
    print(f"Saved {len(arrays)} pair arrays → {out_path}  (skipped {skipped})")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute LinearFold secondary-structure token-level pair arrays."
    )
    parser.add_argument('csv_file', help="Input CSV with tx_id and sequence columns")
    parser.add_argument(
        '--output', '-o',
        help="Output .npz path (default: <csv_stem>_lf_bias.npz next to the CSV)"
    )
    parser.add_argument(
        '--max-tokens', type=int, default=1022,
        help=(
            "Truncate to this many tokens before folding (default: 1022, matching "
            "model_max_length=1024 minus CLS and SEP). Set 0 to fold the full sequence."
        )
    )
    parser.add_argument(
        '--jobs', type=int, default=1,
        help="Number of parallel worker processes (default: 1)"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_path = (
        Path(args.output) if args.output
        else csv_path.with_name(csv_path.stem + '_lf_bias.npz')
    )

    process_csv(csv_path, out_path, max_tokens=args.max_tokens, jobs=args.jobs)


if __name__ == '__main__':
    main()
