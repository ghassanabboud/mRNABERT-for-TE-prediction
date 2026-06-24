"""
Tests for finetuning.metrics.calculate_metric_for_regression.

All inputs are hardcoded numpy arrays — no CSV or model needed.
r2 is defined as pearson**2 (RiboNN convention), not sklearn's R².
"""

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

from finetuning.metrics import calculate_metric_for_regression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(logits_list, labels_list, label_names=None):
    """Build float32 arrays from nested lists and call the metric function."""
    logits = np.array(logits_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)
    return calculate_metric_for_regression(logits, labels, label_names=label_names)


# ---------------------------------------------------------------------------
# 1. Perfect predictions — single label
# ---------------------------------------------------------------------------

def test_perfect_single_label():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    m = _run([[v] for v in vals], [[v] for v in vals])
    assert m["pearson_corr_mean"] == pytest.approx(1.0)
    assert m["r2_score_mean"] == pytest.approx(1.0)
    assert m["mse_loss_mean"] == pytest.approx(0.0)
    assert m["pearson_mean_TE"] == pytest.approx(1.0)
    assert m["r2_mean_TE"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. Perfect predictions — multi-label
# ---------------------------------------------------------------------------

def test_perfect_multi_label():
    labels = [[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0],
              [10., 11., 12.]]
    m = _run(labels, labels, label_names=["A", "B", "C"])
    assert m["pearson_corr_mean"] == pytest.approx(1.0)
    assert m["r2_score_mean"] == pytest.approx(1.0)
    assert m["mse_loss_mean"] == pytest.approx(0.0)
    assert m["pearson_mean_TE"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. NaN masking per label
#    Label 0 has NaN for the last 3 samples; label 1 is fully observed.
#    Label 0's metrics should use only the first 2 samples; label 1 uses all 5.
# ---------------------------------------------------------------------------

def test_nan_masking_per_label():
    nan = float("nan")
    logits = [[0.1, 0.5],
              [0.2, 0.4],
              [0.9, 0.3],
              [0.8, 0.2],
              [0.7, 0.1]]
    labels = [[0.1, 0.5],
              [0.2, 0.4],
              [nan, 0.3],
              [nan, 0.2],
              [nan, 0.1]]

    m = _run(logits, labels, label_names=["X", "Y"])

    # Label X: only samples 0-1 are valid → perfect match → pearson=1, r2=1
    assert m["pearson_X"] == pytest.approx(1.0)
    assert m["r2_X"] == pytest.approx(1.0)

    # Label Y: all 5 samples valid → perfect match → pearson=1
    assert m["pearson_Y"] == pytest.approx(1.0)
    assert m["r2_Y"] == pytest.approx(1.0)

    # mse_loss_mean pools all valid (preds, labels) pairs → should be 0
    assert m["mse_loss_mean"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. Label with fewer than 2 valid samples is excluded from mean metrics
#    Label 1 has only one non-NaN sample → skipped.
#    Mean metrics should reflect only label 0 (which has perfect predictions).
# ---------------------------------------------------------------------------

def test_label_with_one_valid_sample_skipped():
    nan = float("nan")
    logits = [[1.0, 9.0],
              [2.0, 9.0],
              [3.0, 9.0]]
    labels = [[1.0, 5.0],
              [2.0, nan],
              [3.0, nan]]

    m = _run(logits, labels, label_names=["good", "bad"])

    # Only "good" contributes to the means
    assert m["pearson_corr_mean"] == pytest.approx(1.0)
    assert m["r2_score_mean"] == pytest.approx(1.0)

    # "bad" metrics should not appear in output keys (label was skipped)
    assert "pearson_bad" not in m
    assert "r2_bad" not in m


# ---------------------------------------------------------------------------
# 6. mean_TE: predictions masked by label NaN before averaging
#    Sequence 0 has NaN for label 1; its mean_TE should average only label 0.
#    Sequence 1 has NaN for label 0; its mean_TE should average only label 1.
#    Both predictions exactly match the valid labels → pearson_mean_TE = 1.
# ---------------------------------------------------------------------------

def test_mean_te_nan_masking():
    nan = float("nan")
    # seq 0: label0=1.0 (valid), label1=nan → mean_label_TE=1.0, mean_pred_TE=1.0
    # seq 1: label0=nan, label1=4.0 (valid) → mean_label_TE=4.0, mean_pred_TE=4.0
    # seq 2: both valid, labels=[2.0, 3.0], preds=[2.0, 3.0] → means=2.5
    logits = [[1.0, 99.],
              [99., 4.0],
              [2.0,  3.0]]
    labels = [[1.0, nan],
              [nan, 4.0],
              [2.0, 3.0]]

    m = _run(logits, labels)
    assert m["pearson_mean_TE"] == pytest.approx(1.0)
    assert m["r2_mean_TE"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 7. All-NaN sequence does not contribute to mean_TE
# ---------------------------------------------------------------------------

def test_all_nan_sequence_excluded_from_mean_te():
    nan = float("nan")
    # seq 0: all NaN → excluded from mean_TE
    # seq 1 and 2: perfect predictions
    logits = [[0.5, 0.5],
              [1.0, 2.0],
              [3.0, 4.0]]
    labels = [[nan, nan],
              [1.0, 2.0],
              [3.0, 4.0]]

    m = _run(logits, labels)
    assert m["pearson_mean_TE"] == pytest.approx(1.0)
    assert m["r2_mean_TE"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 8. 3D logits (B, 1, n_labels) are reshaped correctly
# ---------------------------------------------------------------------------

def test_3d_logits_reshaped():
    labels_2d = np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=np.float32)
    logits_3d = labels_2d[:, np.newaxis, :]   # (3, 1, 2)

    m = calculate_metric_for_regression(logits_3d, labels_2d)
    assert m["pearson_corr_mean"] == pytest.approx(1.0)
    assert m["r2_score_mean"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 9. label_names appear in per-label output keys
# ---------------------------------------------------------------------------

def test_label_names_in_output_keys():
    labels = [[1., 2.], [3., 4.], [5., 6.]]
    m = _run(labels, labels, label_names=["HeLa", "HEK293"])
    assert "pearson_HeLa" in m
    assert "spearman_HeLa" in m
    assert "r2_HeLa" in m
    assert "pearson_HEK293" in m
    assert "r2_HEK293" in m


# ---------------------------------------------------------------------------
# Cross-check: mean metrics agree with manual scipy computation
# ---------------------------------------------------------------------------

def test_mean_metrics_agree_with_scipy():
    rng = np.random.default_rng(42)
    labels = rng.standard_normal((20, 3)).astype(np.float32)
    logits = labels + 0.1 * rng.standard_normal((20, 3)).astype(np.float32)

    m = calculate_metric_for_regression(logits, labels)

    expected_pearsons = [pearsonr(labels[:, i], logits[:, i])[0] for i in range(3)]
    expected_spearmans = [spearmanr(labels[:, i], logits[:, i])[0] for i in range(3)]
    expected_r2s = [p ** 2 for p in expected_pearsons]

    assert m["pearson_corr_mean"] == pytest.approx(np.mean(expected_pearsons), rel=1e-5)
    assert m["spearman_corr_mean"] == pytest.approx(np.mean(expected_spearmans), rel=1e-5)
    assert m["r2_score_mean"] == pytest.approx(np.mean(expected_r2s), rel=1e-5)
    assert m["mse_loss_mean"] == pytest.approx(
        mean_squared_error(labels.ravel(), logits.ravel()), rel=1e-5
    )