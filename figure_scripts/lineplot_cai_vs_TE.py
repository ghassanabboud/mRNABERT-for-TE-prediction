import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.analysis import get_cai

TE_COLUMN_PREFIX = "TE_"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate CAI for all sequences in a CSV and plot it against mean TE."
    )
    parser.add_argument("--input_csv", type=str, required=True,
                         help="Path to the input CSV (must have a 'sequence' column and TE_* label columns).")
    parser.add_argument("--output_csv", type=str, default="data_with_cai.csv",
                         help="Path to write the input CSV augmented with 'cai' and 'mean_TE' columns.")
    parser.add_argument("--output_fig", type=str, default="figures/check_cai_vs_TE.png",
                         help="Path to save the CAI vs. mean TE seaborn plot.")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    df["cai"] = df["sequence"].apply(get_cai)

    te_cols = [col for col in df.columns if col.startswith(TE_COLUMN_PREFIX)]
    df["mean_TE"] = df[te_cols].mean(axis=1, skipna=True)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} rows with CAI to {args.output_csv}")

    os.makedirs(os.path.dirname(args.output_fig) or ".", exist_ok=True)
    sns.lmplot(data=df, x="cai", y="mean_TE")
    plt.tight_layout()
    plt.savefig(args.output_fig)
    print(f"Saved CAI vs. mean TE plot to {args.output_fig}")


if __name__ == "__main__":
    main()
