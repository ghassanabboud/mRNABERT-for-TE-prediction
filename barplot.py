import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json


paths = ["/scratch/izar/gabboud/mRNABERT/outputs/biased_head_wc_utr5_cds_1024_frozen_2_layer_full_bias",
         "/scratch/izar/gabboud/mRNABERT/outputs/biased_head_wc_utr5_cds_1024_frozen_3_layer_full_bias",
         "/scratch/izar/gabboud/mRNABERT/outputs/biased_head_wc_utr5_cds_1024_frozen_1_layer_full_bias",
         "/scratch/izar/gabboud/mRNABERT/outputs/biased_head_wc_utr5_cds_1024_unfrozen_1_layer_full_bias",
         "/scratch/izar/gabboud/mRNABERT/outputs/biased_head_wc_utr5_cds_1024_unfrozen_2_layer_full_bias",
         "/scratch/izar/gabboud/mRNABERT/outputs/biased_head_wc_utr5_cds_1024_unfrozen_3_layer_full_bias",
         "/scratch/izar/gabboud/mRNABERT/outputs/biased_head_wc_utr5_cds_1024_frozen_1_layer_no_bias",
         "/scratch/izar/gabboud/mRNABERT/outputs/biased_head_wc_utr5_cds_1024_frozen_1_layer_utr_bias"]




row_list = []

for path in paths:
    frozen = False if "unfrozen" in path else True
    num_layers = int(path.split("_")[-4])
    bias_type = "full bias" if "full_bias" in path else ("UTR bias" if "utr_bias" in path else "no bias")
    with open(f"{path}/results/{path.split('/')[-1]}/test_results.json") as f:
        json_content = json.load(f)
    row_list.append({
        "Backbone Frozen": frozen,
        "Number of Bio-Prior Layers": num_layers,
        "Bias Type": bias_type,
        "R² Score": json_content["eval_r2_mean_TE"]
    })

df = pd.DataFrame(row_list)
df_frozen = df[df["Backbone Frozen"] == True]

plt.figure(figsize=(10, 6))
sns.barplot(x="Number of Bio-Prior Layers", y="R² Score", hue="Bias Type", data=df_frozen, ci=None)
#plt.title("R² Scores by Sequences Included and Model Maxmimum Length")
#plt.xlabel("Sequences Included")
#plt.xlabel("")
#plt.ylabel("R² Score")
#plt.xticks(rotation=15)
plt.tight_layout()
plt.ylim(0.62,0.64)
plt.savefig("figures/bioprior_bias_type.png")


df_full = df[df["Bias Type"] == "full bias"]
plt.figure(figsize=(10, 6))
sns.barplot(x="Number of Bio-Prior Layers", y="R² Score", hue="Backbone Frozen", data=df_full, ci=None)
#plt.title("R² Scores by Sequences Included and Model Maxmimum Length")
#plt.xlabel("Sequences Included")
#plt.xlabel("")
#plt.ylabel("R² Score")
#plt.xticks(rotation=15)
plt.tight_layout()
plt.ylim(0.6,0.7)
plt.savefig("figures/bioprior_frozen.png")



