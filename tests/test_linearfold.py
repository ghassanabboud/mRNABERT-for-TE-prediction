"""
Tests for bias.linearfold.parse_token_ranges and dotbracket_to_token_pairs.

These functions are pure (no LinearFold executable needed) and convert a
dot-bracket string + token list into a (K, 3) int32 array of [t_i, t_j, count]
token-level pair interactions.

Dot-bracket notation reminder:
  '(' at position i and ')' at position j means nucleotide i pairs with j.
  '.' means unpaired.

Token index convention: 0-based, excluding CLS/SEP.  The collator later shifts
by +1 to account for CLS.
"""

import numpy as np
import pytest

from bias.linearfold import dotbracket_to_token_pairs, parse_token_ranges, run_linearfold, process_one
from conftest import LF_TEST_DATA


# ---------------------------------------------------------------------------
# parse_token_ranges
# ---------------------------------------------------------------------------

def test_single_nuc_tokens_ranges():
    tokens = ["A", "T", "G"]
    nuc_seq, ranges = parse_token_ranges(tokens)
    assert nuc_seq == "AUG"          # T → U
    assert ranges == [(0, 1), (1, 1), (2, 1)]


def test_codon_tokens_ranges():
    tokens = ["ATG", "GCT"]
    nuc_seq, ranges = parse_token_ranges(tokens)
    assert nuc_seq == "AUGGCU"       # T → U in both codons
    assert ranges == [(0, 3), (3, 3)]


def test_mixed_tokens_ranges():
    tokens = ["A", "T", "ATG"]
    nuc_seq, ranges = parse_token_ranges(tokens)
    assert nuc_seq == "AUAUG"
    assert ranges == [(0, 1), (1, 1), (2, 3)]


# ---------------------------------------------------------------------------
# dotbracket_to_token_pairs — helpers
# ---------------------------------------------------------------------------

def _pairs(tokens, dot_bracket):
    """Convenience wrapper: tokens + dot-bracket → list of (ti, tj, count) tuples."""
    _, ranges = parse_token_ranges(tokens)
    arr = dotbracket_to_token_pairs(dot_bracket, ranges)
    return [tuple(row) for row in arr.tolist()]


# ---------------------------------------------------------------------------
# dotbracket_to_token_pairs — no pairs
# ---------------------------------------------------------------------------

def test_all_dots_returns_empty():
    assert _pairs(["A", "T", "G"], "...") == []


def test_empty_sequence_returns_empty():
    _, ranges = parse_token_ranges([])
    arr = dotbracket_to_token_pairs("", ranges)
    assert arr.shape == (0, 3)


# ---------------------------------------------------------------------------
# dotbracket_to_token_pairs — single-nucleotide tokens (UTR-like)
#
# With single-nuc tokens each nucleotide IS its own token, so the count is
# always 1 and ti < tj is determined directly by nucleotide position.
# ---------------------------------------------------------------------------

def test_single_nuc_simple_stem():
    tokens = ["A", "T", "G", "C", "A", "T"]
    result = _pairs(tokens, "((..))")
    assert result == [(0, 5, 1), (1, 4, 1)]


def test_single_nuc_nested_pairs():
    # tokens: A(0) T(1) G(2) C(3)
    # "(())" → 0↔3, 1↔2
    tokens = ["A", "T", "G", "C"]
    result = _pairs(tokens, "(())")
    assert result == [(0, 3, 1), (1, 2, 1)]


def test_single_nuc_partial_pairing():

    tokens = ["A", "T", "G", "C", "A", "T"]
    result = _pairs(tokens, "(.(.))")
    assert result == [(0, 5, 1), (2, 4, 1)]


# ---------------------------------------------------------------------------
# dotbracket_to_token_pairs — codon tokens
#
# A codon token covers 3 nucleotides, so up to 3 nucleotide pairs can map
# to the same token pair, giving count ∈ {1, 2, 3}.
# ---------------------------------------------------------------------------

def test_codon_pair_count_one():
    tokens = ["ATG", "GCT"]
    result = _pairs(tokens, "(....)")
    assert result == [(0, 1, 1)]


def test_codon_pair_count_two():

    tokens = ["ATG", "GCT"]
    result = _pairs(tokens, "((..))")
    assert result == [(0, 1, 2)]


def test_codon_pair_count_three():

    tokens = ["ATG", "GCT"]
    result = _pairs(tokens, "((()))")
    assert result == [(0, 1, 3)]


def test_intra_codon_pairs_discarded():
    tokens = ["ATG"]
    result = _pairs(tokens, "(.)")
    assert result == []


# ---------------------------------------------------------------------------
# dotbracket_to_token_pairs — mixed single-nuc and codon tokens
# ---------------------------------------------------------------------------

def test_mixed_nuc_codon_pairs():
    tokens = ["A", "T", "ATG", "GCT"]
    result = _pairs(tokens, "(......)")
    assert result == [(0, 3, 1)]


def test_mixed_nuc_to_codon_and_nuc_to_nuc():
    tokens = ["G", "A", "T", "ATG", "GCT", "C"]
    result = _pairs(tokens, "(((....)))")
    assert result == [(0, 5, 1), (1, 4, 1), (2, 4, 1)]


def test_output_dtype_and_shape():
    tokens = ["ATG", "GCT"]
    _, ranges = parse_token_ranges(tokens)
    arr = dotbracket_to_token_pairs("((()))", ranges)
    assert arr.dtype == np.int32
    assert arr.ndim == 2
    assert arr.shape[1] == 3


# ---------------------------------------------------------------------------
# Integration tests — require --linearfold /path/to/linearfold
#
# Skipped automatically when the flag is absent.
# Run with:  pytest tests/ --linearfold ../LinearFold/linearfold
# ---------------------------------------------------------------------------

def test_run_linearfold_output_length(linearfold_exe):
    """dot-bracket returned by LinearFold must have the same length as the input."""
    nuc_seq = "AUGCAUGCAUGC"
    dot_bracket = run_linearfold(nuc_seq, linearfold_exe)
    assert len(dot_bracket) == len(nuc_seq)


def test_run_linearfold_valid_characters(linearfold_exe):
    """dot-bracket must contain only '.', '(', ')'."""
    nuc_seq = "GCGCGCGCGCGC"
    dot_bracket = run_linearfold(nuc_seq, linearfold_exe)
    assert set(dot_bracket) <= {".", "(", ")"}


def test_process_one_output_shape(linearfold_exe):
    """process_one must return a (K, 3) int32 array for a simple mixed sequence."""
    import pandas as pd
    df = pd.read_csv(LF_TEST_DATA / "test.csv")
    row = df.iloc[0]
    _, pairs = process_one(row["tx_id"], row["sequence"], max_tokens=0, executable=linearfold_exe)
    assert pairs.ndim == 2
    assert pairs.shape[1] == 3
    assert pairs.dtype == np.int32



def test_process_one_count_in_range(linearfold_exe):
    """Pair counts must be in {1, 2, 3} (at most 3 nuc pairs per codon-codon token pair)."""
    import pandas as pd
    df = pd.read_csv(LF_TEST_DATA / "test.csv")
    for _, row in df.iterrows():
        _, pairs = process_one(row["tx_id"], row["sequence"], max_tokens=0, executable=linearfold_exe)
        for ti, tj, count in pairs:
            assert 1 <= count <= 3, f"{row['tx_id']}: pair ({ti},{tj}) has invalid count {count}"
