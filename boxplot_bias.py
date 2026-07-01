import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

plt.rcParams.update({'font.size': 12})

paths = [
    "/scratch/izar/gabboud/mRNABERT/outputs/cv_biased_full_1024_frozen_1_layer_no_bias",
    "/scratch/izar/gabboud/mRNABERT/outputs/cv_FIXED_biased_full_1024_frozen_1_layer_wc_bias",
    "/scratch/izar/gabboud/mRNABERT/outputs/cv_biased_full_1024_frozen_1_layer_lf_bias",
    "/scratch/izar/gabboud/mRNABERT/outputs/cv_biased_full_1024_frozen_2_layer_no_bias",
    "/scratch/izar/gabboud/mRNABERT/outputs/cv_FIXED_biased_full_1024_frozen_2_layer_wc_bias",
    "/scratch/izar/gabboud/mRNABERT/outputs/cv_biased_full_1024_frozen_2_layer_lf_bias",
    "/scratch/izar/gabboud/mRNABERT/outputs/cv_biased_full_1024_frozen_3_layer_no_bias",
    "/scratch/izar/gabboud/mRNABERT/outputs/cv_FIXED_biased_full_1024_frozen_3_layer_wc_bias",
    "/scratch/izar/gabboud/mRNABERT/outputs/cv_biased_full_1024_frozen_3_layer_lf_bias",

]

# x-axis grouping
n_layers_list = [1, 1, 1, 2, 2, 2, 3, 3, 3]

# hue grouping
bias_list = ["No Bias", "Watson-Crick Bias", "LinearFold Bias",
             "No Bias", "Watson-Crick Bias", "LinearFold Bias",
             "No Bias", "Watson-Crick Bias", "LinearFold Bias"]

order     = [1, 2, 3]
hue_order = ["No Bias", "Watson-Crick Bias", "LinearFold Bias"]

df_list = []
for path, n_layers, bias in zip(paths, n_layers_list, bias_list):
    d = pd.read_csv(f"{path}/cv_results.csv")
    d["Number of Layers"] = n_layers
    d["Bias Type"] = bias
    df_list.append(d)

df = pd.concat(df_list, ignore_index=True)

plt.figure(figsize=(9, 6))
ax = sns.boxplot(x="Number of Layers", y="eval_r2_mean_TE", hue="Bias Type",
                 data=df, order=order, hue_order=hue_order, boxprops=dict(alpha=.3))
sns.stripplot(x="Number of Layers", y="eval_r2_mean_TE", hue="Bias Type",
              data=df, order=order, hue_order=hue_order,
              dodge=True, jitter=True, alpha=1, legend=False)
plt.grid(True, axis="y", linestyle="--", alpha=0.7)

# ---------------------------------------------------------------------------
# Significance bars
# ---------------------------------------------------------------------------

n_hue = len(hue_order)
width = 0.8
offsets = np.linspace(-width / 2 + width / (2 * n_hue),
                       width / 2 - width / (2 * n_hue), n_hue)
hue_offset = dict(zip(hue_order, offsets))
layer_pos  = {n: i for i, n in enumerate(order)}


def box_x(n_layers, bias):
    return layer_pos[n_layers] + hue_offset[bias]


def nadeau_bengio_ttest(v1, v2):
    """Nadeau-Bengio corrected paired t-test for k-fold CV."""
    k = len(v1)
    d = v2 - v1
    d_bar = d.mean()
    s2 = d.var(ddof=1)
    se = np.sqrt((1 / k + 1 / (k - 1)) * s2)
    t = d_bar / se
    p = 2 * stats.t.sf(np.abs(t), df=k - 1)
    return p


def group_vals(n_layers, bias):
    return (df[(df["Number of Layers"] == n_layers) & (df["Bias Type"] == bias)]
            .sort_values("test_fold")["eval_r2_mean_TE"]
            .values)


def draw_sig_bar(ax, x1, x2, y, label, h=0.005):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=1.2)
    ax.text((x1 + x2) / 2, y + h + 0.002, label,
            ha="center", va="bottom", fontsize=10)


# Each tuple: (n_layers1, bias1, n_layers2, bias2, bar_level)
# level 0: within-group bias comparisons (no x-overlap between groups)
# level 1+: cross-group comparisons that span multiple x positions
pairs = [
    (1, "No Bias", 1, "Watson-Crick Bias", 0),
    (2, "No Bias", 2, "Watson-Crick Bias", 0),
    (3, "No Bias", 3, "Watson-Crick Bias", 0),
    (1, "No Bias", 1, "LinearFold Bias", 1),
    (2, "No Bias", 2, "LinearFold Bias", 1),
    (3, "No Bias", 3, "LinearFold Bias", 1),
    (3, "No Bias", 2, "No Bias", 2),
    (1, "No Bias", 2, "No Bias", 2),
]

# Collect raw p-values for all pairs, then apply Bonferroni correction.
n_tests = len(pairs)
raw_pvals = []
for n1, b1, n2, b2, _ in pairs:
    v1, v2 = group_vals(n1, b1), group_vals(n2, b2)
    mask = ~(np.isnan(v1) | np.isnan(v2))
    v1, v2 = v1[mask], v2[mask]
    raw_pvals.append(nadeau_bengio_ttest(v1, v2) if len(v1) >= 2 else np.nan)

corrected_pvals = [min(p * n_tests, 1.0) if not np.isnan(p) else np.nan
                   for p in raw_pvals]


def sig_label(p):
    if np.isnan(p):
        return "x (p=N/A)"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return "x"


y_data_max = df["eval_r2_mean_TE"].max()
y_start    = y_data_max + 0.01
level_step = 0.01

for (n1, b1, n2, b2, level), p_corr in zip(pairs, corrected_pvals):
    x1 = box_x(n1, b1)
    x2 = box_x(n2, b2)
    y  = y_start + level * level_step
    draw_sig_bar(ax, x1, x2, y, sig_label(p_corr), h=0.002)

ax.set_ylim(top=y_start + 2 * level_step + 0.02, bottom=0.54)

plt.xlabel("Number of Layers")
plt.ylabel("R² Score")
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig("figures/r2_scores_cv_bias_FIXED.png")
