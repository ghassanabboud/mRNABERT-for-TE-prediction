import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({"font.size": 12})

RESULTS_CSV_PATH = "./outputs/multi_AUG_insertion/multi_AUG_results.csv"
OUTPUT_FIGURE_PATH = "figures/multi_uAUG_insertion_delta_TE.png"


def main():
    df = pd.read_csv(RESULTS_CSV_PATH)

    baseline = df[df["num_augs"] == 0].set_index("tx_id")["predicted_mean_TE"]
    variants = df[df["num_augs"] > 0].copy()
    variants["baseline_TE"] = variants["tx_id"].map(baseline)
    variants["delta_TE"] = variants["predicted_mean_TE"] - variants["baseline_TE"]

    plt.figure(figsize=(9, 6))
    sns.lineplot(x="num_augs", y="delta_TE", data=variants, errorbar="se", marker="o", color="crimson")
    plt.axhline(0, color="grey", linestyle="-", alpha=0.5)
    plt.xlabel("Number of inserted upstream AUGs")
    plt.ylabel("Mean ΔTE (inserted − baseline)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.minorticks_on()
    plt.grid(which="major", linestyle="-", alpha=0.4)
    plt.grid(which="minor", linestyle=":", alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE_PATH)
    print(f"Saved figure to {OUTPUT_FIGURE_PATH}")


if __name__ == "__main__":
    main()
