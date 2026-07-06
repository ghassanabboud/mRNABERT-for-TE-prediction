import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.plotting import bonferroni_correct, dodge_x, draw_sig_bar, hue_offsets, nadeau_bengio_ttest, sig_label

plt.rcParams.update({'font.size': 14})


# REPLACE WITH YOUR OWN PATHS AND CORRESPONDING MODES / MAX LENGTHS
paths = ["./outputs/cv_cds_only_1024",
         "./outputs/cv_utr5_only_1024",
         "./outputs/cv_utr5_cds_1024",
         "./outputs/cv_full_1024",
         #./outputs/cv_start_codon_window_600nt_400",
         "./outputs/cv_utr5_cds_2044",
         "./outputs/cv_full_2044"]

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
hue_palette = ["#2a78d6", "#1baf7a"]  # blue, aqua

df_list = []
for path, mode, max_length in zip(paths, modes, max_lengths):
    df = pd.read_csv(f"{path}/cv_results.csv")
    df["mode"] = mode
    df["Maximum Sequence Length"] = max_length
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)

summary = df.groupby(["mode", "Maximum Sequence Length"])["eval_r2_mean_TE"].agg(["mean", "std"])
print(summary)

plt.figure(figsize=(10, 7))
ax = sns.boxplot(x="mode", y="eval_r2_mean_TE", hue="Maximum Sequence Length",
                 data=df, order=order, hue_order=hue_order, palette=hue_palette,
                 boxprops=dict(alpha=.5))
sns.stripplot(x="mode", y="eval_r2_mean_TE", hue="Maximum Sequence Length",
              data=df, order=order, hue_order=hue_order, palette=hue_palette,
              dodge=True, jitter=True, alpha=1, legend=False)

# ---------------------------------------------------------------------------
# Significance bars 
# ---------------------------------------------------------------------------


hue_offset = hue_offsets(hue_order)
mode_pos = {m: i for i, m in enumerate(order)}

def box_x(mode, length):
    return dodge_x(mode_pos, hue_offset, mode, length)


def group_vals(mode, length):
    """R² values per test_fold, sorted so pairing is consistent."""
    return (df[(df["mode"] == mode) & (df["Maximum Sequence Length"] == length)]
            .sort_values("test_fold")["eval_r2_mean_TE"]
            .values)



pairs = [
    ("UTR5 + CDS",        1024, "UTR5 + CDS",        2044, 0),
    ("UTR5 + CDS + UTR3", 1024, "UTR5 + CDS + UTR3", 2044, 0),
    ("UTR5 + CDS",        1024, "UTR5 + CDS + UTR3", 1024, 1),
    ("UTR5 + CDS",        2044, "UTR5 + CDS + UTR3", 2044, 2),
    ("UTR5 + CDS", 1024,"UTR5 only", 1024, 1),
    ("UTR5 + CDS", 1024,"CDS only", 1024, 2),

]

# Collect raw p-values for all pairs, then apply Bonferroni correction.
raw_pvals = []
for m1, l1, m2, l2, _ in pairs:
    v1, v2 = group_vals(m1, l1), group_vals(m2, l2)
    mask = ~(np.isnan(v1) | np.isnan(v2))
    v1, v2 = v1[mask], v2[mask]
    raw_pvals.append(nadeau_bengio_ttest(v1, v2) if len(v1) >= 2 else np.nan)

corrected_pvals = bonferroni_correct(raw_pvals)

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
plt.savefig("figures/r2_boxplot_sequence_ablation.png")
