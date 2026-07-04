"""Sanity-check the distance-matched negative sampling in `build_pair_records`
(utils/analysis.py): loads two attention-correlation pairs CSVs, as produced by
study_attention_ss_correlation.py, and plots the distribution of pair distance
|i - j|, split by positive (LinearFold contact) vs. negative (sampled) pairs.

If distance matching worked, the positive and negative distributions should
overlap closely in each subplot -- unlike the old uniform-random sampling,
where negatives skewed towards much larger distances than positives.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PAIRS_CSV_1 = "outputs/attention_correlation/balanced_sampling_200_seqs/attention_correlation_results.csv"
LABEL_1 = "unmatched (old)"
PAIRS_CSV_2 = "outputs/attention_correlation/balanced_sampling_400_seqs_distance_matched/attention_correlation_results.csv"
LABEL_2 = "distance-matched (new)"
OUTPUT_PATH = "figures/distance_matching_sanity_check.png"


def load_pair_distances(csv_path):
    """Load a pairs CSV and return one row per unique (tx_id, i, j) pair with its
    sequence distance and positive/negative label, deduplicating across layers
    (the same sampled pairs repeat once per layer in these files)."""
    df = pd.read_csv(csv_path, usecols=["tx_id", "i", "j", "bias_count"])
    df = df.drop_duplicates(subset=["tx_id", "i", "j"])
    df["distance"] = (df["i"] - df["j"]).abs()
    df["pair_type"] = df["bias_count"].apply(lambda c: "positive" if c > 0 else "negative")
    return df


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, csv_path, label in zip(axes, [PAIRS_CSV_1, PAIRS_CSV_2], [LABEL_1, LABEL_2]):
    df = load_pair_distances(csv_path)
    sns.histplot(
        data=df, x="distance", hue="pair_type", bins=50, ax=ax,
        element="step", stat="density", common_norm=False,
    )
    ax.set_title(label)
    ax.set_xlabel("Pair distance |i - j| (tokens)")
    ax.set_ylabel("Density")

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=300)
print(f"Saved figure to {OUTPUT_PATH}")
