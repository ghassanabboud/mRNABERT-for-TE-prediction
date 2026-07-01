"""
Plot the effect of AUG insertion on predicted translation efficiency (TE).

Loads the per-variant predictions produced by insertional_analysis.py, computes the
delta in predicted mean TE relative to each transcript's unmodified baseline, and plots
the average delta TE across transcripts as a function of insertion position.

See experiments/09-uAUG_insertion.md for the motivating analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update({"font.size": 12})

RESULTS_CSV_PATH = "/scratch/izar/gabboud/mRNABERT/outputs/insertional_analysis/max_seq_200_upstream_500/insertional_analysis_results.csv"
OUTPUT_FIGURE_PATH = "figures/uAUG_insertion_delta_TE.png"


def main():
    df = pd.read_csv(RESULTS_CSV_PATH)

    baseline = df[df["insertion_position"].isna()].set_index("tx_id")["predicted_mean_TE"]
    variants = df[df["insertion_position"].notna()].copy()
    variants["baseline_TE"] = variants["tx_id"].map(baseline)
    variants["delta_TE"] = variants["predicted_mean_TE"] - variants["baseline_TE"]

    plt.figure(figsize=(9, 6))
    sns.lineplot(x="insertion_position", y="delta_TE", data=variants, errorbar="se")
    plt.axvline(0, color="black", linestyle="--", alpha=0.5, label="Start codon")
    plt.axhline(0, color="grey", linestyle="-", alpha=0.5)
    plt.xlabel("Insertion position relative to start codon (nt)")
    #plt.xlim(-10,10)
    plt.ylabel("Mean ΔTE (inserted − baseline)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE_PATH)
    print(f"Saved figure to {OUTPUT_FIGURE_PATH}")


if __name__ == "__main__":
    main()
