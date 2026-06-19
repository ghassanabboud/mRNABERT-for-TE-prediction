import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


paths = ["/scratch/izar/gabboud/mRNABERT/outputs/cv_cds_only_1024",
         "/scratch/izar/gabboud/mRNABERT/outputs/cv_utr5_only_1024",
         "/scratch/izar/gabboud/mRNABERT/outputs/cv_utr5_cds_1024",
         "/scratch/izar/gabboud/mRNABERT/outputs/cv_full_1024",
         "/scratch/izar/gabboud/mRNABERT/outputs/cv_start_codon_window_600nt_400"]

modes = ["CDS only", "UTR5 only", "UTR5 + CDS", "UTR5 + CDS + UTR3", "600nt centered on start codon"]
max_lengths = [1024, 1024, 1024, 1024, 400]


df_list = []

for path, mode, max_length in zip(paths, modes, max_lengths):
    df = pd.read_csv(f"{path}/cv_results.csv")
    df["mode"] = mode
    df["Maximum Input Sequence Length"] = max_length
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)
plt.figure(figsize=(10, 6))
ax = sns.boxplot(x="mode", y="eval_r2_mean_TE", hue="Maximum Input Sequence Length", data=df)
sns.stripplot(x="mode", y="eval_r2_mean_TE", hue="Maximum Input Sequence Length", data=df,
              dodge=False, jitter=True, color='black', alpha=0.5, legend=False)
#plt.ylim(0,1)
#plt.title("R² Scores by Sequences Included and Model Maxmimum Length")
#plt.xlabel("Sequences Included")
plt.xlabel("")
plt.ylabel("R² Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("figures/r2_scores_comparison.png")