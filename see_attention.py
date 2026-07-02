import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/scratch/izar/gabboud/mRNABERT/outputs/attention_correlation/balanced_sampling_200_seqs/attention_correlation_results.csv")
df["touches"] = df["bias_count"] > 0
df["log_attn_score"] = np.log1p(df["attn_score"])
print("number of contacts:", df["touches"].sum())
print("number of non-contacts:", (~df["touches"]).sum())
print(df.head())
plt.figure()
sns.boxplot(data=df, x="layer", y="attn_score", hue="touches")
plt.xlabel("Layer of the model")
plt.ylabel("Attention score")
plt.savefig("attention_at_layers.png", dpi=300)


plt.figure()
sns.histplot(data=df[df["layer"]==0], x="attn_score")
plt.xlabel("Attention score")
plt.ylabel("Count")
plt.savefig("attention_layer_0.png", dpi=300)

plt.figure()
sns.histplot(data=df[df["layer"]==2], x="attn_score", log_scale=True, hue="touches", stat="probability", common_norm=False)
plt.xlabel("Log Attention score")
plt.ylabel("Frequency")
plt.savefig("log_attention_layer_0.png", dpi=300)

plt.figure()
sns.kdeplot(data=df[df["layer"]==2], x="attn_score", log_scale=True, hue="bias_count", common_norm=False, fill=True)
plt.xlabel("Log Attention score")
plt.ylabel("Density")
plt.savefig("log_attention_layer_0_kde.png", dpi=300)

g = sns.displot(
    data=df, x="attn_score", row="layer", hue="touches", kind="kde",
    log_scale=True, common_norm=False, fill=True,
    height=1.5, aspect=4, facet_kws={"sharex": True, "sharey": False},
)
g.set_axis_labels("Log Attention score", "Density")
g.set_titles(row_template="Layer {row_name}")
g.savefig("attention_by_layer.png", dpi=300)