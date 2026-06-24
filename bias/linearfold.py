"""
Utilities for computing LinearFold secondary-structure token-level pair arrays.

Pure functions (parse_token_ranges, dotbracket_to_token_pairs) have no external
dependencies and are unit-testable without LinearFold installed.
run_linearfold and process_one require the LinearFold executable.
"""

import subprocess
from collections import Counter
from typing import List, Tuple

import numpy as np


def parse_token_ranges(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """Convert a list of mixed nucleotide/codon tokens into:
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
    nuc_seq = "".join(parts).upper().replace("T", "U")
    return nuc_seq, token_ranges


def dotbracket_to_token_pairs(
    dot_bracket: str,
    token_ranges: List[Tuple[int, int]],
) -> np.ndarray:
    """Parse a nucleotide-level dot-bracket structure into token-level base pairs
    with interaction counts.

    Mapping rule: nucleotide pair (i, j) → token pair (t_i, t_j). For codon
    tokens (3 nucleotides each), multiple nucleotide pairs can map to the same
    token pair, giving a count up to 3. Single-nucleotide UTR tokens contribute
    at most 1. Self-pairs (both nucleotides inside the same codon) are discarded.

    Returns an int32 array of shape (K, 3): columns are [t_i, t_j, count], where
    t_i < t_j, count ∈ {1, 2, 3}, and indices are 0-based excluding CLS/SEP.
    The collator sets bias[t_i+1, t_j+1] = count (symmetric).
    """
    N = len(dot_bracket)

    nuc_to_tok = np.empty(N, dtype=np.int32)
    for t, (s, k) in enumerate(token_ranges):
        nuc_to_tok[s : s + k] = t

    stack: List[int] = []
    counts: Counter = Counter()
    for i, c in enumerate(dot_bracket):
        if c == "(":
            stack.append(i)
        elif c == ")":
            j = stack.pop()
            ti, tj = int(nuc_to_tok[j]), int(nuc_to_tok[i])
            if ti != tj:
                counts[(min(ti, tj), max(ti, tj))] += 1

    if not counts:
        return np.empty((0, 3), dtype=np.int32)
    rows = [(ti, tj, cnt) for (ti, tj), cnt in sorted(counts.items())]
    return np.array(rows, dtype=np.int32)


def run_linearfold(nuc_seq: str, executable: str) -> str:
    """Run LinearFold on `nuc_seq` and return the MFE dot-bracket string.

    LinearFold output format:
        SEQUENCE
        .(((...)))  (-1.23)
    """
    result = subprocess.run(
        [executable],
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


def process_one(
    tx_id: str,
    sequence_field: str,
    max_tokens: int,
    executable: str,
) -> Tuple[str, np.ndarray]:
    """Process a single sequence: tokenize → fold → extract token pairs."""
    tokens = sequence_field.split()
    if max_tokens > 0:
        tokens = tokens[:max_tokens]
    nuc_seq, token_ranges = parse_token_ranges(tokens)
    dot_bracket = run_linearfold(nuc_seq, executable)
    return tx_id, dotbracket_to_token_pairs(dot_bracket, token_ranges)