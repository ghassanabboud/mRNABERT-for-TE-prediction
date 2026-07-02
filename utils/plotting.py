"""
Reusable helpers for the grouped CV-comparison box plots in figure_scripts/
(e.g. boxplot.py, boxplot_bias.py): CV-corrected significance testing and
seaborn dodge-position bookkeeping needed to annotate grouped box plots with
significance bars.
"""

from typing import Dict, Hashable, List, Sequence

import numpy as np
from scipy import stats


def nadeau_bengio_ttest(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compare two paired k-fold CV score arrays and return a p-value for
    whether their means differ, correcting for the fact that CV folds share
    overlapping training data (so a plain paired t-test understates the
    true variance and overstates significance).

    Inflates the standard error by sqrt(1/k + 1/(k-1)) instead of the usual
    sqrt(1/k), to account for the ~(k-1)/k overlap between the training sets
    of different folds.

    Parameters
    ----------
    v1 : np.ndarray
        Per-fold scores for the first condition, ordered by fold.
    v2 : np.ndarray
        Per-fold scores for the second condition, in the same fold order
        as `v1`.

    Returns
    -------
    float
        Two-sided p-value for the null hypothesis that `v1` and `v2` have
        equal mean.
    """
    k = len(v1)
    d = v2 - v1
    d_bar = d.mean()
    s2 = d.var(ddof=1)
    se = np.sqrt((1 / k + 1 / (k - 1)) * s2)
    t = d_bar / se
    p = 2 * stats.t.sf(np.abs(t), df=k - 1)
    return p


def bonferroni_correct(pvals: Sequence[float]) -> List[float]:
    """Apply a Bonferroni correction to a list of p-values from running
    multiple significance tests on the same figure (e.g. several pairwise
    CV comparisons), so the combined false-positive rate stays controlled.

    Parameters
    ----------
    pvals : Sequence[float]
        Raw p-values, one per test; NaN entries (e.g. from a skipped test)
        are passed through unchanged.

    Returns
    -------
    List[float]
        Corrected p-values, each multiplied by the number of tests and
        capped at 1.0. NaN entries stay NaN.
    """
    n = len(pvals)
    return [min(p * n, 1.0) if not np.isnan(p) else np.nan for p in pvals]


def sig_label(p: float) -> str:
    """Convert a p-value into the significance marker drawn above a
    box-plot comparison bar.

    Parameters
    ----------
    p : float
        Corrected p-value for the comparison; NaN if the test could not be
        run (e.g. fewer than 2 paired samples).

    Returns
    -------
    str
        "**" if p <= 0.01, "*" if p <= 0.05, "x" if not significant, or
        "x (p=N/A)" if `p` is NaN.
    """
    if np.isnan(p):
        return "x (p=N/A)"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return "x"


def hue_offsets(hue_order: Sequence[Hashable], width: float = 0.8) -> Dict[Hashable, float]:
    """Compute the x-offset of each hue's box within a seaborn grouped
    (dodged) box plot, so significance bars can be drawn at the exact x
    position of each box rather than at the category's center tick.

    Mirrors how seaborn spaces `n_hue` dodged boxes evenly across `width`
    around each category tick.

    Parameters
    ----------
    hue_order : Sequence[Hashable]
        Hue values in the order passed to seaborn's `hue_order`.
    width : float, optional
        Total width seaborn allocates to the dodged group at each category
        tick. Default 0.8 (seaborn's default).

    Returns
    -------
    Dict[Hashable, float]
        Maps each hue value to its x-offset from the category tick.
    """
    n_hue = len(hue_order)
    offsets = np.linspace(
        -width / 2 + width / (2 * n_hue), width / 2 - width / (2 * n_hue), n_hue
    )
    return dict(zip(hue_order, offsets))


def dodge_x(
    pos_map: Dict[Hashable, float],
    offset_map: Dict[Hashable, float],
    category: Hashable,
    hue: Hashable,
) -> float:
    """Compute the x-coordinate of one dodged box in a grouped box plot,
    for placing a significance bar over it.

    Parameters
    ----------
    pos_map : Dict[Hashable, float]
        Maps each x-axis category to its tick position (typically its index
        in `order`).
    offset_map : Dict[Hashable, float]
        Maps each hue value to its x-offset from the tick, as returned by
        `hue_offsets`.
    category : Hashable
        x-axis category of the target box.
    hue : Hashable
        Hue value of the target box.

    Returns
    -------
    float
        x-coordinate of the box's center.
    """
    return pos_map[category] + offset_map[hue]


def draw_sig_bar(
    ax,
    x1: float,
    x2: float,
    y: float,
    label: str,
    h: float = 0.005,
    text_gap: float = 0.003,
    fontsize: int = 10,
) -> None:
    """Draw a bracket-shaped significance bar between two box-plot x
    positions with a centered label above it (e.g. "*", "**", or "x").

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x1 : float
        x-coordinate of the bar's left end.
    x2 : float
        x-coordinate of the bar's right end.
    y : float
        y-coordinate of the bar's horizontal segment.
    label : str
        Text drawn centered above the bar, typically from `sig_label`.
    h : float, optional
        Height of the bar's vertical end-ticks. Default 0.005.
    text_gap : float, optional
        Vertical gap between the bar and the label text. Default 0.003.
    fontsize : int, optional
        Font size of the label text. Default 10.

    Returns
    -------
    None
    """
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=1.2)
    ax.text(
        (x1 + x2) / 2, y + h + text_gap, label, ha="center", va="bottom", fontsize=fontsize
    )
