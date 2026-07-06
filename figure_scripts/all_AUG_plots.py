import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({"font.size": 14})

INSERTION_RESULTS_CSV_PATH = "./outputs/insertional_analysis/max_seq_200_upstream_500/insertional_analysis_results.csv"
MULTI_AUG_RESULTS_CSV_PATH = "./outputs/multi_AUG_insertion/multi_AUG_results.csv"
OUTPUT_FIGURE_PATH = "figures/all_AUG_plots.png"


def main():
    insertion_df = pd.read_csv(INSERTION_RESULTS_CSV_PATH)
    insertion_baseline = insertion_df[insertion_df["insertion_position"].isna()].set_index("tx_id")["predicted_mean_TE"]
    insertion_variants = insertion_df[insertion_df["insertion_position"].notna()].copy()
    insertion_variants["baseline_TE"] = insertion_variants["tx_id"].map(insertion_baseline)
    insertion_variants["delta_TE"] = insertion_variants["predicted_mean_TE"] - insertion_variants["baseline_TE"]

    multi_df = pd.read_csv(MULTI_AUG_RESULTS_CSV_PATH)
    multi_baseline = multi_df[multi_df["num_augs"] == 0].set_index("tx_id")["predicted_mean_TE"]
    multi_variants = multi_df[multi_df["num_augs"] > 0].copy()
    multi_variants["baseline_TE"] = multi_variants["tx_id"].map(multi_baseline)
    multi_variants["delta_TE"] = multi_variants["predicted_mean_TE"] - multi_variants["baseline_TE"]

    fig, (ax_position, ax_count) = plt.subplots(2, 1, figsize=(9, 12))

    sns.lineplot(x="insertion_position", y="delta_TE", data=insertion_variants, errorbar="se", ax=ax_position)
    #ax_position.axvline(0, color="black", linestyle="--", alpha=0.5, label="Start codon")
    ax_position.axhline(0, color="grey", linestyle="-", alpha=0.5)
    ax_position.set_xlabel("Insertion position relative to start codon (nt)")
    ax_position.set_ylabel("Mean ΔTE (inserted − baseline)")
    ax_position.legend()
    ax_position.text(-0.1, 1.01, "A)", transform=ax_position.transAxes, fontsize=18, va="bottom")

    sns.lineplot(x="num_augs", y="delta_TE", data=multi_variants, errorbar="se", marker="o", color="crimson", ax=ax_count)
    ax_count.axhline(0, color="grey", linestyle="-", alpha=0.5)
    ax_count.set_xlabel("Number of inserted upstream AUGs")
    ax_count.set_ylabel("Mean ΔTE (inserted − baseline)")
    ax_count.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_count.minorticks_on()
    ax_count.grid(which="major", linestyle="-", alpha=0.4)
    ax_count.grid(which="minor", linestyle=":", alpha=0.2)
    ax_count.text(-0.1, 1.01, "B)", transform=ax_count.transAxes, fontsize=18, va="bottom")

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE_PATH)
    print(f"Saved figure to {OUTPUT_FIGURE_PATH}")


if __name__ == "__main__":
    main()