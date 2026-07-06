import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from utils.plotting import bonferroni_correct, sig_label

plt.rcParams.update({'font.size': 14})

#df = pd.read_csv("./outputs/codon_test/codon_test_results.csv")
df = pd.read_csv("/scratch/izar/gabboud/mRNABERT/outputs/codon_analysis_all_sequences/codon_test_results.csv")

baseline = df[df["variant_type"] == "wildtype"].set_index("tx_id")["predicted_mean_TE"]
variants = df[df["variant_type"] != "wildtype"].copy()
variants["baseline_TE"] = variants["tx_id"].map(baseline)
variants["delta_TE"] = variants["predicted_mean_TE"] - variants["baseline_TE"]

mapping_for_plot = {
    "wildtype": "Wildtype",
    "optimal": "Codon-optimized",
    "least_optimal": "Codon-diminished"
}
variants["variant_type_plot"] = variants["variant_type"].map(mapping_for_plot)
df["variant_type_plot"] = df["variant_type"].map(mapping_for_plot)

plot_order = [
    "Wildtype",
    "Codon-optimized",
    "Codon-diminished",
]
palette = dict(zip(plot_order, sns.color_palette(n_colors=len(plot_order))))

print(df.groupby("variant_type_plot")["CAI"].mean())
print(variants.groupby("variant_type_plot")["delta_TE"].mean())

#histogram of CAI
plt.figure(figsize=(8,6))
ax = sns.histplot(data=df, x="CAI", hue="variant_type_plot", hue_order=plot_order, palette=palette, bins=20, kde=True)
ax.get_legend().set_title(None)
plt.grid(True, axis="x")
plt.xlabel("Codon Adaptation Index (CAI)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figures/cai_hist_more_seqs.png")

box_order = [v for v in plot_order if v in variants["variant_type_plot"].unique()]

plt.figure(figsize=(6,6))
ax = sns.boxplot(data=variants, x="variant_type_plot", y="delta_TE", order=box_order, palette=palette)
ax.yaxis.grid(True)
ax.set_axisbelow(True)

y_range = variants["delta_TE"].max() - variants["delta_TE"].min()

raw_p_values = []
for variant_type in box_order:
    deltas = variants.loc[variants["variant_type_plot"] == variant_type, "delta_TE"].dropna()
    _, p_value = stats.wilcoxon(deltas)
    raw_p_values.append(p_value)
p_values = bonferroni_correct(raw_p_values)

for i, (variant_type, p_value) in enumerate(zip(box_order, p_values)):
    deltas = variants.loc[variants["variant_type_plot"] == variant_type, "delta_TE"].dropna()
    box_top = deltas.max()
    ax.text(i, box_top + 0.01 * y_range, sig_label(p_value), ha="center", va="bottom")

plt.xlabel("")
plt.ylabel("ΔTE (variant − wildtype)")
plt.tight_layout()
plt.savefig("./figures/codon_delta_te_boxplot_more_seqs.png")

#histogram of delta TE, split by codon-optimized vs codon-diminished
plt.figure(figsize=(8,6))
ax = sns.histplot(data=variants, x="delta_TE", hue="variant_type_plot", hue_order=box_order, palette=palette, bins=30, kde=True)
ax.get_legend().set_title(None)
plt.grid(True, axis="x")
plt.xlabel("ΔTE (variant − wildtype)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("./figures/codon_delta_te_hist_more_seqs.png")

# combined figure: CAI histogram (top) + delta TE boxplot (bottom), stacked vertically
fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(8, 12))

sns.histplot(data=df, x="CAI", hue="variant_type_plot", hue_order=plot_order, palette=palette, bins=20, kde=True, ax=ax_hist)
ax_hist.get_legend().set_title(None)
ax_hist.grid(True, axis="x")
ax_hist.set_xlabel("Codon Adaptation Index (CAI)")
ax_hist.set_ylabel("Count")

sns.boxplot(data=variants, x="variant_type_plot", y="delta_TE", order=box_order, palette=palette, ax=ax_box)
ax_box.yaxis.grid(True)
ax_box.set_axisbelow(True)
for i, (variant_type, p_value) in enumerate(zip(box_order, p_values)):
    deltas = variants.loc[variants["variant_type_plot"] == variant_type, "delta_TE"].dropna()
    box_top = deltas.max()
    ax_box.text(i, box_top + 0.01 * y_range, sig_label(p_value), ha="center", va="bottom")
ax_box.set_xlabel("")
ax_box.set_ylabel("ΔTE (variant − wildtype)")

plt.tight_layout()
plt.savefig("./figures/codon_cai_hist_and_delta_te_boxplot_combined.png")

