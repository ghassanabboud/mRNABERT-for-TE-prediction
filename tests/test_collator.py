"""
Tests for finetuning.collators.SupervisedDataCollator.

Covers:
  - WC modes (full / utr_only): CLS+SEP rows/cols are 0, nuc–nuc / nuc–codon /
    codon–codon bio_prior values match the wc_lookup.
  - LinearFold mode: bio_prior matches the (ti, tj, count) arrays in the
    pre-computed .npz, including symmetry and all-zero CLS/SEP positions.

Token IDs assumed (mRNABERT tokenizer):
  [PAD]=0, [CLS]=2, [SEP]=3, A=5, T=6, C=7, G=8
  ATG=17, GCT=67

Synthetic input_ids are constructed directly so tests never depend on
the tokenizer's string-parsing path.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import LF_TEST_DATA
from finetuning.collators import SupervisedDataCollator
from finetuning.datasets import SupervisedDataset


# ---------------------------------------------------------------------------
# Constants matching the mRNABERT vocab
# ---------------------------------------------------------------------------

CLS = 2
SEP = 3
A, T, C, G = 5, 6, 7, 8
ATG, GCT = 17, 67

# Synthetic sequence: [CLS, A, T, G, ATG, GCT, SEP]
# positions:            0   1  2  3   4    5    6
SYNTHETIC_IDS = torch.tensor([CLS, A, T, G, ATG, GCT, SEP])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def col_full(tokenizer, wc_full):
    return SupervisedDataCollator(tokenizer, bias_mode="full", wc_lookup=wc_full)


@pytest.fixture(scope="module")
def col_utr_only(tokenizer, wc_utr_only):
    return SupervisedDataCollator(tokenizer, bias_mode="utr_only", wc_lookup=wc_utr_only)


@pytest.fixture(scope="module")
def col_lf(tokenizer):
    return SupervisedDataCollator(
        tokenizer,
        bias_mode="linearfold",
        bias_npz_path=str(LF_TEST_DATA / "test.npz"),
    )


@pytest.fixture(scope="module")
def lf_records(tokenizer):
    """List of (tx_id, input_ids, pairs) for every sequence in the LF test data."""
    df = pd.read_csv(LF_TEST_DATA / "test.csv")
    pairs_lookup = dict(np.load(str(LF_TEST_DATA / "test.npz"), allow_pickle=False))
    records = []
    for _, row in df.iterrows():
        tx_id = row["tx_id"]
        enc = tokenizer(row["sequence"], return_tensors="pt", truncation=True)
        records.append((tx_id, enc["input_ids"][0], pairs_lookup[tx_id]))
    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_instance(input_ids, tx_id=None):
    inst = {"input_ids": input_ids, "labels": [0.0]}
    if tx_id is not None:
        inst["tx_id"] = tx_id
    return inst


def _wc_bias(collator, input_ids):
    """Run a single-sequence WC batch; return bio_prior (L, L)."""
    return collator([_make_instance(input_ids)])["bio_prior"][0, 0]


def _lf_bias(collator, tx_id, input_ids):
    """Run a single-sequence LF batch; return bio_prior (L, L)."""
    return collator([_make_instance(input_ids.clone(), tx_id=tx_id)])["bio_prior"][0, 0]


# ---------------------------------------------------------------------------
# WC full mode — CLS and SEP rows/cols must be 0
# ---------------------------------------------------------------------------

SEP_POS = SYNTHETIC_IDS.tolist().index(SEP)


@pytest.mark.parametrize("pos,axis", [
    (0,       "row"),   # CLS row
    (0,       "col"),   # CLS col
    (SEP_POS, "row"),   # SEP row
    (SEP_POS, "col"),   # SEP col
])
def test_wc_full_special_token_bias_is_zero(col_full, pos, axis):
    bias = _wc_bias(col_full, SYNTHETIC_IDS)
    vec = bias[pos, :] if axis == "row" else bias[:, pos]
    assert vec.sum().item() == 0


# ---------------------------------------------------------------------------
# WC full mode — nuc–nuc scores
# Sequence: [CLS(0), A(1), T(2), G(3), ATG(4), GCT(5), SEP(6)]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pos_i,pos_j,expected", [
    (1, 2, 2),   # A–T = 2
    (2, 1, 2),   # T–A = 2  (symmetric)
    (2, 3, 1),   # T–G = 1
    (3, 2, 1),   # G–T = 1  (symmetric)
    (1, 3, 0),   # A–G = 0
    (1, 1, 0),   # A–A = 0
])
def test_wc_full_nuc_nuc(col_full, pos_i, pos_j, expected):
    assert _wc_bias(col_full, SYNTHETIC_IDS)[pos_i, pos_j].item() == expected


# ---------------------------------------------------------------------------
# WC full mode — nuc–codon scores
# A(1)–ATG(4)=2,  T(2)–ATG(4)=3,  G(3)–GCT(5)=4  (see test_wc_lookup.py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pos_i,pos_j,expected", [
    (1, 4, 2),   # A–ATG = 2
    (4, 1, 2),   # ATG–A = 2  (symmetric)
    (2, 4, 3),   # T–ATG = 3
    (4, 2, 3),   # ATG–T = 3  (symmetric)
    (3, 5, 4),   # G–GCT = 4
    (5, 3, 4),   # GCT–G = 4  (symmetric)
])
def test_wc_full_nuc_codon(col_full, pos_i, pos_j, expected):
    assert _wc_bias(col_full, SYNTHETIC_IDS)[pos_i, pos_j].item() == expected


# ---------------------------------------------------------------------------
# WC full mode — codon–codon scores
# ATG(4)–GCT(5) = 7  (see test_wc_lookup.py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pos_i,pos_j,expected", [
    (4, 5, 7),   # ATG–GCT
    (5, 4, 7),   # GCT–ATG (symmetric)
])
def test_wc_full_codon_codon(col_full, pos_i, pos_j, expected):
    assert _wc_bias(col_full, SYNTHETIC_IDS)[pos_i, pos_j].item() == expected


# ---------------------------------------------------------------------------
# WC utr_only mode — codon positions score 0; nuc–nuc unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pos_i,pos_j", [
    (4, 5), (5, 4),   # ATG–GCT and reverse
    (1, 4), (4, 1),   # A–ATG and reverse
    (2, 4),           # T–ATG
    (3, 5),           # G–GCT
])
def test_wc_utr_only_codon_positions_are_zero(col_utr_only, pos_i, pos_j):
    assert _wc_bias(col_utr_only, SYNTHETIC_IDS)[pos_i, pos_j].item() == 0


@pytest.mark.parametrize("pos_i,pos_j,expected", [
    (1, 2, 2),  # A–T
    (2, 3, 1),  # T–G
])
def test_wc_utr_only_nuc_nuc_preserved(col_utr_only, pos_i, pos_j, expected):
    assert _wc_bias(col_utr_only, SYNTHETIC_IDS)[pos_i, pos_j].item() == expected

# ---------------------------------------------------------------------------
# WC in conjunction with a supervised dataset.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_batch(tokenizer, wc_full):
    """Full-pipeline batch: all LF test sequences collated into one batch.

    Returns the full batch dict.  Use attention_mask[b].sum()-1 to locate the
    SEP token for sequence b: the mask is 1 for every real token (CLS, sequence
    tokens, SEP) and 0 for PAD, so sum()-1 is the index of the last real token.
    """
    dataset = SupervisedDataset(
        data_path=str(LF_TEST_DATA / "test.csv"),
        tokenizer=tokenizer,
    )
    col = SupervisedDataCollator(tokenizer, bias_mode="full", wc_lookup=wc_full)
    return col([dataset[i] for i in range(len(dataset))])


def test_wc_pipeline_shape(pipeline_batch):
    """bio_prior shape must be (B, 1, L_max, L_max) where L_max is the padded
    sequence length."""
    bio_prior = pipeline_batch["bio_prior"]
    B, L_max = pipeline_batch["attention_mask"].shape
    assert bio_prior.shape == (B, 1, L_max, L_max)


def test_wc_pipeline_cls_is_zero(pipeline_batch):
    """CLS row and column must be zero for every sequence in the batch."""
    bio_prior = pipeline_batch["bio_prior"]
    for b in range(bio_prior.shape[0]):
        assert bio_prior[b, 0,  0, :].sum().item() == 0, f"seq {b}: CLS row nonzero"
        assert bio_prior[b, 0, :,  0].sum().item() == 0, f"seq {b}: CLS col nonzero"


def test_wc_pipeline_sep_is_zero(pipeline_batch):
    """SEP row and column must be zero for every sequence in the batch.
    SEP is at the last non-PAD position: attention_mask[b].sum() - 1."""
    bio_prior = pipeline_batch["bio_prior"]
    mask = pipeline_batch["attention_mask"]
    for b in range(bio_prior.shape[0]):
        sep = int(mask[b].sum().item()) - 1
        assert bio_prior[b, 0, sep,  :].sum().item() == 0, f"seq {b}: SEP row nonzero"
        assert bio_prior[b, 0,  :, sep].sum().item() == 0, f"seq {b}: SEP col nonzero"


def test_wc_pipeline_padded_positions_are_zero(pipeline_batch):
    """All PAD positions of sequence 0 must have zero rows and columns.

    Padding for sequence 0 starts at attention_mask[0].sum() (the first 0 in
    the mask) and runs to L_max-1.
    """
    bio_prior = pipeline_batch["bio_prior"]
    mask = pipeline_batch["attention_mask"]
    L0 = int(mask[0].sum().item())
    L_max = bio_prior.shape[2]
    assert L0 < L_max, "Sequence 0 has no padding — test precondition not met"
    for pad_pos in range(L0, L_max):
        assert bio_prior[0, 0, pad_pos,  :].sum().item() == 0, f"pad pos {pad_pos}: row nonzero"
        assert bio_prior[0, 0,  :, pad_pos].sum().item() == 0, f"pad pos {pad_pos}: col nonzero"


def test_wc_full_pipeline_spot_checks(pipeline_batch):
    """Spot-check specific (batch_idx, pos_i, pos_j) entries against manually
    derived expected scores.

    Sequence 0 (ENST00000338591.8) tokens after tokenization:
      [CLS, G, G, G, A, GCT, GTG, GGC, GGC, GGG, C, A, T, G, T, C, T, T, T, SEP]
       0    1  2  3  4   5    6    7    8    9   10 11 12 13 14 15 16 17 18  19

    Sequence 1 (ENST00000435064.6) tokens after tokenization:
      [CLS, G, C, A, G, CGG, CAG, GTG, CTG, GAG, T, G, A, A, C, A, A, A, A, G, A, SEP]
       0    1  2  3  4   5    6    7    8    9   10 11 12 13 14 15 16 17 18 19 20  21
    """
    bio_prior = pipeline_batch["bio_prior"]

    # (batch_idx, pos_i, pos_j, expected_score)
    # Score derivations use the 3×3 nuc_wc sum rule (see test_wc_lookup.py).
    #   GCT=[G,C,T] × GTG=[G,T,G]: G×{G,T,G}=1 + C×{G,T,G}=6 + T×{G,T,G}=2 = 9
    #   CGG=[C,G,G] × CAG=[C,A,G]: C×{C,A,G}=3 + G×{C,A,G}=3 + G×{C,A,G}=3 = 9
    checks = [
        # seq 0 — nuc-codon: A(4) vs GCT(5)  →  A×{G,C,T}: 0+0+2 = 2
        (0,  4,  5, 2),
        # seq 0 — codon-codon: GCT(5) vs GTG(6)  →  9
        (0,  5,  6, 9),
        # seq 0 — nuc-nuc: T(12) vs A(11)  →  T–A = 2
        (0, 12, 11, 2),
        # seq 1 — nuc-nuc: G(1) vs C(2)  →  G–C = 3
        (1,  1,  2, 3),
        # seq 1 — codon-codon: CGG(5) vs CAG(6)  →  9
        (1,  5,  6, 9),
        # seq 1 — nuc-nuc: T(10) vs G(11)  →  T–G = 1
        (1, 10, 11, 1),
    ]

    for b, pi, pj, expected in checks:
        actual = bio_prior[b, 0, pi, pj].item()
        assert actual == expected, (
            f"batch[{b}] pos ({pi},{pj}): expected {expected}, got {actual}"
        )


# ---------------------------------------------------------------------------
# LinearFold mode
# ---------------------------------------------------------------------------

def test_lf_bio_prior_shape(col_lf, lf_records):
    for tx_id, input_ids, _ in lf_records:
        L = len(input_ids)
        batch = col_lf([_make_instance(input_ids.clone(), tx_id=tx_id)])
        assert batch["bio_prior"].shape == (1, 1, L, L)


@pytest.mark.parametrize("axis", ["row", "col"])
def test_lf_cls_is_zero(col_lf, lf_records, axis):
    for tx_id, input_ids, _ in lf_records:
        bias = _lf_bias(col_lf, tx_id, input_ids)
        vec = bias[0, :] if axis == "row" else bias[:, 0]
        assert vec.sum().item() == 0, f"{tx_id}: CLS {axis} nonzero"


@pytest.mark.parametrize("axis", ["row", "col"])
def test_lf_sep_is_zero(col_lf, lf_records, axis):
    for tx_id, input_ids, _ in lf_records:
        bias = _lf_bias(col_lf, tx_id, input_ids)
        sep = len(input_ids) - 1
        vec = bias[sep, :] if axis == "row" else bias[:, sep]
        assert vec.sum().item() == 0, f"{tx_id}: SEP {axis} nonzero"


def test_lf_pair_values_match_npz(col_lf, lf_records):
    """Each (ti, tj, count) in the npz must appear at (ti+1, tj+1) in bio_prior."""
    for tx_id, input_ids, pairs in lf_records:
        L = len(input_ids)
        bias = _lf_bias(col_lf, tx_id, input_ids)
        for ti, tj, count in pairs:
            pi, pj = int(ti) + 1, int(tj) + 1
            if pi < L and pj < L:
                assert bias[pi, pj].item() == count, (
                    f"{tx_id}: pair ({ti},{tj}) expected {count}, got {bias[pi, pj].item()}"
                )


def test_lf_bio_prior_is_symmetric(col_lf, lf_records):
    for tx_id, input_ids, _ in lf_records:
        bias = _lf_bias(col_lf, tx_id, input_ids)
        assert torch.equal(bias, bias.T), f"{tx_id}: bio_prior is not symmetric"


def test_lf_no_extra_nonzero_entries(col_lf, lf_records):
    """bio_prior must be zero at every position not covered by the npz pairs."""
    for tx_id, input_ids, pairs in lf_records:
        L = len(input_ids)
        bias = _lf_bias(col_lf, tx_id, input_ids)
        expected = torch.zeros(L, L)
        for ti, tj, count in pairs:
            pi, pj = int(ti) + 1, int(tj) + 1
            if pi < L and pj < L:
                expected[pi, pj] = count
                expected[pj, pi] = count
        assert torch.equal(bias, expected), f"{tx_id}: bio_prior has unexpected nonzero entries"


# ---------------------------------------------------------------------------
# LinearFold mode in conjunction with a supervised dataset.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lf_pipeline_batch(tokenizer):
    """Full-pipeline LF batch: all test sequences through SupervisedDataset +
    SupervisedDataCollator(linearfold).

    SupervisedDataset includes tx_id in each item so the collator can look up
    the pre-computed pairs from the .npz.  Use attention_mask[b].sum()-1 to
    locate the SEP token for sequence b.
    """
    dataset = SupervisedDataset(
        data_path=str(LF_TEST_DATA / "test.csv"),
        tokenizer=tokenizer,
    )
    col = SupervisedDataCollator(
        tokenizer,
        bias_mode="linearfold",
        bias_npz_path=str(LF_TEST_DATA / "test.npz"),
    )
    return col([dataset[i] for i in range(len(dataset))])


def test_lf_pipeline_shape(lf_pipeline_batch):
    """bio_prior shape must be (B, 1, L_max, L_max)."""
    bio_prior = lf_pipeline_batch["bio_prior"]
    B, L_max = lf_pipeline_batch["attention_mask"].shape
    assert bio_prior.shape == (B, 1, L_max, L_max)


def test_lf_pipeline_cls_is_zero(lf_pipeline_batch):
    """CLS row and column must be zero for every sequence in the batch."""
    bio_prior = lf_pipeline_batch["bio_prior"]
    for b in range(bio_prior.shape[0]):
        assert bio_prior[b, 0,  0, :].sum().item() == 0, f"seq {b}: CLS row nonzero"
        assert bio_prior[b, 0, :,  0].sum().item() == 0, f"seq {b}: CLS col nonzero"


def test_lf_pipeline_sep_is_zero(lf_pipeline_batch):
    """SEP row and column must be zero for every sequence in the batch."""
    bio_prior = lf_pipeline_batch["bio_prior"]
    mask = lf_pipeline_batch["attention_mask"]
    for b in range(bio_prior.shape[0]):
        sep = int(mask[b].sum().item()) - 1
        assert bio_prior[b, 0, sep,  :].sum().item() == 0, f"seq {b}: SEP row nonzero"
        assert bio_prior[b, 0,  :, sep].sum().item() == 0, f"seq {b}: SEP col nonzero"


def test_lf_pipeline_padded_positions_are_zero(lf_pipeline_batch):
    """All PAD positions of sequence 0 must have zero rows and columns."""
    bio_prior = lf_pipeline_batch["bio_prior"]
    mask = lf_pipeline_batch["attention_mask"]
    L0 = int(mask[0].sum().item())
    L_max = bio_prior.shape[2]
    assert L0 < L_max, "Sequence 0 has no padding — test precondition not met"
    for pad_pos in range(L0, L_max):
        assert bio_prior[0, 0, pad_pos,  :].sum().item() == 0, f"pad pos {pad_pos}: row nonzero"
        assert bio_prior[0, 0,  :, pad_pos].sum().item() == 0, f"pad pos {pad_pos}: col nonzero"


def test_lf_pipeline_spot_checks(lf_pipeline_batch):
    """Spot-check specific pairs from the npz appear at the right positions.

    From linearfold_test_data/test.npz (0-based token indices, excl. CLS):
      Seq 0 (ENST00000338591.8): pairs [[4,7,3], [5,6,1]]
        → bio_prior[0,0, 5, 8] = 3  (symmetric: [0,0, 8, 5] = 3)
        → bio_prior[0,0, 6, 7] = 1  (symmetric: [0,0, 7, 6] = 1)
      Seq 1 (ENST00000435064.6): pairs [[1,7,1], [2,7,1], [3,7,1]]
        → bio_prior[1,0, 2, 8] = 1  (symmetric: [1,0, 8, 2] = 1)
        → bio_prior[1,0, 4, 8] = 1  (symmetric: [1,0, 8, 4] = 1)
    """
    bio_prior = lf_pipeline_batch["bio_prior"]

    # (batch_idx, pos_i, pos_j, expected_count)
    checks = [
        # seq 0 — pair (4,7,3): ti+1=5, tj+1=8
        (0, 5, 8, 3),
        (0, 8, 5, 3),   # symmetric
        # seq 0 — pair (5,6,1): ti+1=6, tj+1=7
        (0, 6, 7, 1),
        (0, 7, 6, 1),   # symmetric
        # seq 0 — non-pair position must be 0
        (0, 1, 2, 0),
        # seq 1 — pair (1,7,1): ti+1=2, tj+1=8
        (1, 2, 8, 1),
        (1, 8, 2, 1),   # symmetric
        # seq 1 — pair (3,7,1): ti+1=4, tj+1=8
        (1, 4, 8, 1),
        (1, 8, 4, 1),   # symmetric
        # seq 1 — non-pair position must be 0
        (1, 1, 2, 0),
    ]

    for b, pi, pj, expected in checks:
        actual = bio_prior[b, 0, pi, pj].item()
        assert actual == expected, (
            f"batch[{b}] pos ({pi},{pj}): expected {expected}, got {actual}"
        )
