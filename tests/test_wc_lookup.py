"""
Tests for bias.wc.build_wc_lookup.

Token IDs (from the mRNABERT tokenizer shipped with this repo):
  [PAD]=0, [CLS]=2, [SEP]=3, A=5, T=6, C=7, G=8, N=9
  ATG=17, GTG=65, GCT=67, GGC=72

Watson-Crick scoring rules (nuc level):
  A–T = 2   (and symmetric)
  C–G = 3   (and symmetric)
  G–T = 1   (and symmetric, wobble pair)

For codon tokens the full-mode score equals the sum of all 3×3 constituent
nucleotide-pair scores.  In utr_only mode codon tokens contribute 0.
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def ids(vocab, *names):
    """Return token IDs for the given token name(s)."""
    return [vocab[n] for n in names]


# ---------------------------------------------------------------------------
# nuc–nuc: canonical WC pairs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tok_i,tok_j,expected", [
    ("A", "T", 2),
    ("T", "A", 2),
    ("C", "G", 3),
    ("G", "C", 3),
    ("T", "G", 1),
    ("G", "T", 1),
])
def test_nuc_nuc_wc_pairs(wc_full, vocab, tok_i, tok_j, expected):
    i, j = vocab[tok_i], vocab[tok_j]
    assert wc_full[i, j].item() == expected


# ---------------------------------------------------------------------------
# nuc–nuc: non-pairing combinations must be 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tok_i,tok_j", [
    ("A", "A"), ("A", "C"), ("A", "G"),
    ("T", "T"), ("T", "C"),
    ("C", "C"), ("C", "T"), ("C", "A"),
    ("G", "G"), ("G", "A"),
])
def test_nuc_nuc_non_pairing_is_zero(wc_full, vocab, tok_i, tok_j):
    i, j = vocab[tok_i], vocab[tok_j]
    assert wc_full[i, j].item() == 0


# ---------------------------------------------------------------------------
# CLS and SEP must have all-zero rows and columns
# (their token_nucs slots are never filled, so all constituent pairs are 0)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("special", ["[CLS]", "[SEP]"])
def test_special_token_row_is_zero(wc_full, vocab, special):
    sid = vocab[special]
    assert wc_full[sid, :].sum().item() == 0, (
        f"{special} row should be all-zero but has nonzero entries"
    )


@pytest.mark.parametrize("special", ["[CLS]", "[SEP]"])
def test_special_token_col_is_zero(wc_full, vocab, special):
    sid = vocab[special]
    assert wc_full[:, sid].sum().item() == 0, (
        f"{special} column should be all-zero but has nonzero entries"
    )


# ---------------------------------------------------------------------------
# codon–codon scores in full mode
#
# Expected values derived by summing nuc_wc over all 3×3 constituent pairs:
#
#   ATG=[A,T,G]  GCT=[G,C,T]
#   A×{G,C,T}: 0+0+2=2   T×{G,C,T}: 1+0+0=1   G×{G,C,T}: 0+3+1=4  → total 7
#
#   ATG=[A,T,G]  ATG=[A,T,G]
#   A×{A,T,G}: 0+2+0=2   T×{A,T,G}: 2+0+1=3   G×{A,T,G}: 0+1+0=1  → total 6
#
#   GTG=[G,T,G]  GGC=[G,G,C]
#   G×{G,G,C}: 0+0+3=3   T×{G,G,C}: 1+1+0=2   G×{G,G,C}: 0+0+3=3  → total 8
# 
#   GGG=[G,G,G]  CCC=[C,C,C]
#   G×{C,C,C}: 3+3+3=9   G×{C,C,C}: 3+3+3=9   G×{C,C,C}: 3+3+3=9  → total 27
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tok_i,tok_j,expected", [
    ("ATG", "GCT", 7),
    ("GCT", "ATG", 7),   # symmetric
    ("ATG", "ATG", 6),
    ("GTG", "GGC", 8),
    ("GGC", "GTG", 8),   # symmetric
    ("GGG", "CCC", 27),
    ("CCC", "GGG", 27),  # symmetric
])
def test_codon_codon_full_mode(wc_full, vocab, tok_i, tok_j, expected):
    i, j = vocab[tok_i], vocab[tok_j]
    assert wc_full[i, j].item() == expected


# ---------------------------------------------------------------------------
# nuc–codon scores in full mode
#
#   A×{A,T,G}: 0+2+0=2
#   T×{A,T,G}: 2+0+1=3
#   G×{G,C,T}: 0+3+1=4
#   A×{G,C,C}: 0+0+0=0
#   Ax{A,A,A}: 0+0+0=0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nuc,codon,expected", [
    ("A", "ATG", 2),
    ("ATG", "A",  2),   # symmetric
    ("T", "ATG", 3),
    ("G", "GCT", 4),
    ("GCT", "G",  4),   # symmetric
    ("A", "GCC", 0),
    ("A", "AAA", 0),
])
def test_nuc_codon_full_mode(wc_full, vocab, nuc, codon, expected):
    i, j = vocab[nuc], vocab[codon]
    assert wc_full[i, j].item() == expected


# ---------------------------------------------------------------------------
# utr_only mode: codon tokens contribute 0; nuc–nuc unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tok_i,tok_j", [
    ("ATG", "GCT"),
    ("GCT", "ATG"),
    ("A",   "ATG"),
    ("ATG", "A"),
    ("GTG", "GGC"),
])
def test_codon_scores_zero_in_utr_only(wc_utr_only, vocab, tok_i, tok_j):
    i, j = vocab[tok_i], vocab[tok_j]
    assert wc_utr_only[i, j].item() == 0


@pytest.mark.parametrize("tok_i,tok_j,expected", [
    ("A", "T", 2),
    ("C", "G", 3),
    ("T", "G", 1),
    ("G", "G", 0),
    ("A", "A", 0),
])
def test_nuc_nuc_preserved_in_utr_only(wc_utr_only, vocab, tok_i, tok_j, expected):
    i, j = vocab[tok_i], vocab[tok_j]
    assert wc_utr_only[i, j].item() == expected
