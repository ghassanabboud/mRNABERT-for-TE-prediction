import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

plt.rcParams.update({'font.size': 12})
paths = ["/scratch/izar/gabboud/mRNABERT/outputs/cv_cds_only_1024",
         "/scratch/izar/gabboud/mRNABERT/outputs/cv_utr5_only_1024",
         "/scratch/izar/gabboud/mRNABERT/outputs/cv_utr5_cds_1024",
         "/scratch/izar/gabboud/mRNABERT/outputs/cv_full_1024",
         #"/scratch/izar/gabboud/mRNABERT/outputs/cv_start_codon_window_600nt_400",
         "/scratch/izar/gabboud/mRNABERT/outputs/cv_utr5_cds_2044",
         "/scratch/izar/gabboud/mRNABERT/outputs/cv_full_2044"]

modes = ["CDS only",
         "UTR5 only",
         "UTR5 + CDS", "UTR5 + CDS + UTR3",
         #"600nt centered on start codon",
         "UTR5 + CDS", "UTR5 + CDS + UTR3"]


max_lengths = [1024,
               1024,
               1024,
               1024,
               #400,
               2044,
               2044]

order = ["CDS only", "UTR5 only", "UTR5 + CDS", "UTR5 + CDS + UTR3"]
hue_order = [1024, 2044]

df_list = []
for path, mode, max_length in zip(paths, modes, max_lengths):
    df = pd.read_csv(f"{path}/cv_results.csv")
    df["mode"] = mode
    df["Maximum Sequence Length"] = max_length
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)

plt.figure(figsize=(10, 7))
ax = sns.boxplot(x="mode", y="eval_r2_mean_TE", hue="Maximum Sequence Length",
                 data=df, order=order, hue_order=hue_order, boxprops=dict(alpha=.3))
sns.stripplot(x="mode", y="eval_r2_mean_TE", hue="Maximum Sequence Length",
              data=df, order=order, hue_order=hue_order,
              dodge=True, jitter=True, alpha=1, legend=False)

# ---------------------------------------------------------------------------
# Significance bars showing Δ mean R² between pairs
# ---------------------------------------------------------------------------

# Box x-positions: seaborn evenly spaces n_hue groups over `width` around each tick.
n_hue = len(hue_order)
width = 0.8
offsets = np.linspace(-width / 2 + width / (2 * n_hue),
                       width / 2 - width / (2 * n_hue), n_hue)
hue_offset = dict(zip(hue_order, offsets))
mode_pos = {m: i for i, m in enumerate(order)}


def box_x(mode, length):
    return mode_pos[mode] + hue_offset[length]


def nadeau_bengio_ttest(v1, v2):
    """
    Nadeau-Bengio corrected paired t-test for k-fold CV.
    Inflates the SE by sqrt(1/k + 1/(k-1)) instead of sqrt(1/k) to account
    for the ~(k-1)/k overlap between training sets of different folds.
    """
    k = len(v1)
    d = v2 - v1
    d_bar = d.mean()
    s2 = d.var(ddof=1)
    se = np.sqrt((1 / k + 1 / (k - 1)) * s2)
    t = d_bar / se
    p = 2 * stats.t.sf(np.abs(t), df=k - 1)
    return p


def group_vals(mode, length):
    """R² values per test_fold, sorted so pairing is consistent."""
    return (df[(df["mode"] == mode) & (df["Maximum Sequence Length"] == length)]
            .sort_values("test_fold")["eval_r2_mean_TE"]
            .values)


def draw_sig_bar(ax, x1, x2, y, label, h=0.005):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=1.2)
    ax.text((x1 + x2) / 2, y + h + 0.003, label,
            ha="center", va="bottom", fontsize=10)


# Each tuple: (mode1, length1, mode2, length2, bar_level)
# Levels 0 / 1 / 2 stack bars bottom-to-top to avoid overlap:
#   level 0: pair 1 [x≈2.00–2.27] and pair 2 [x≈3.00–3.27] — no x-overlap
#   level 1: pair 3 [x≈2.00–3.00] — overlaps pair 1 in x
#   level 2: pair 4 [x≈2.27–3.27] — overlaps pair 3 in x
pairs = [
    ("UTR5 + CDS",        1024, "UTR5 + CDS",        2044, 0),
    ("UTR5 + CDS + UTR3", 1024, "UTR5 + CDS + UTR3", 2044, 0),
    ("UTR5 + CDS",        1024, "UTR5 + CDS + UTR3", 1024, 1),
    ("UTR5 + CDS",        2044, "UTR5 + CDS + UTR3", 2044, 2),
    ("UTR5 + CDS", 1024,"UTR5 only", 1024, 1),
    ("UTR5 + CDS", 1024,"CDS only", 1024, 2),

]

# Collect raw p-values for all pairs, then apply Bonferroni correction.
n_tests = len(pairs)
raw_pvals = []
for m1, l1, m2, l2, _ in pairs:
    v1, v2 = group_vals(m1, l1), group_vals(m2, l2)
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


relevant_mask = df["mode"].isin(["UTR5 + CDS", "UTR5 + CDS + UTR3"])
y_data_max = df[relevant_mask]["eval_r2_mean_TE"].max()
y_start = y_data_max + 0.01
level_step = 0.02

for (m1, l1, m2, l2, level), p_corr in zip(pairs, corrected_pvals):
    x1 = box_x(m1, l1)
    x2 = box_x(m2, l2)
    y = y_start + level * level_step
    draw_sig_bar(ax, x1, x2, y, sig_label(p_corr))

# Expand y-axis to accommodate the top-most bar + label
ax.set_ylim(top=y_start + 2 * level_step + 0.1)

plt.xlabel("")
plt.ylabel("R² Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("figures/r2_scores_comparison.png")
