import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_DIRS = {
    "no_bias": "balanced_sampling_200_seqs",
    "wc": "balanced_sampling_200_seqs_wc",
    "linearfold": "balanced_sampling_200_seqs_lf",
}
BASE_PATH = "outputs/attention_correlation"

NUM_BACKBONE_LAYERS = 12
REFERENCE_MODEL = "no_bias"  # backbone is frozen/shared, so any model's backbone rows would do
FIGSIZE = (16, 12)  # shared across both figures so bioprior panels render larger (fewer, same canvas)
FIGSIZE_BACKBONE = (24, 12)

pairs_dfs = []
summaries = {}
for model_name, dir_name in MODEL_DIRS.items():
    pairs_df = pd.read_csv(f"{BASE_PATH}/{dir_name}/attention_correlation_results.csv")
    pairs_df["model"] = model_name
    pairs_dfs.append(pairs_df)
    summaries[model_name] = pd.read_csv(f"{BASE_PATH}/{dir_name}/attention_correlation_summary.csv")

df = pd.concat(pairs_dfs, ignore_index=True)
df["LinearFold-predicted contact"] = df["bias_count"] > 0


def get_spearman_rho(summary_df, layer_idx, layer_kind):
    """Look up spearman_r for a layer, matching either plain-int or 'backbone_i'/'bioprior_i' labels."""
    label_candidates = {str(layer_idx), f"{layer_kind}_{layer_idx}"}
    matches = summary_df[summary_df["layer"].astype(str).isin(label_candidates)]
    if matches.empty:
        return None
    return float(matches.iloc[0]["spearman_r"])


# --- Plot 1: 3x4 grid of backbone layers (shared frozen backbone -> one model suffices) ---
backbone_df = df[df["model"] == REFERENCE_MODEL]
fig, axes = plt.subplots(3, 4, figsize=FIGSIZE_BACKBONE, sharex=True)
for layer_idx in range(NUM_BACKBONE_LAYERS):
    ax = axes[layer_idx // 4, layer_idx % 4]
    layer_df = backbone_df[backbone_df["layer"] == layer_idx]
    sns.kdeplot(
        data=layer_df, x="attn_score", hue="LinearFold-predicted contact",
        log_scale=True, fill=True, common_norm=False, ax=ax, legend=(layer_idx == 0),
    )
    rho = get_spearman_rho(summaries[REFERENCE_MODEL], layer_idx, "backbone")
    if rho is not None:
        ax.text(0.05, 0.95, f"ρ={rho:.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, fontweight="bold")
    ax.set_title(f"Layer {layer_idx}")
    ax.set_xlabel("Log attention score")
    ax.set_ylabel("Density" if layer_idx % 4 == 0 else "")
fig.tight_layout()
fig.savefig("figures/attention_backbone_grid.png", dpi=300)

# --- Plot 2: 3x1 grid of bio-prior layers, one row per model ---
fig2, axes2 = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True)
for i, model_name in enumerate(MODEL_DIRS):
    ax = axes2[i]
    layer_df = df[(df["model"] == model_name) & (df["layer"] == NUM_BACKBONE_LAYERS)]
    sns.kdeplot(
        data=layer_df, x="attn_score", hue="LinearFold-predicted contact",
        log_scale=True, fill=True, common_norm=False, ax=ax, legend=(i == 0),
    )
    ax.set_title(model_name)
    ax.set_xlabel("Log attention score")
    ax.set_ylabel("Density")
    ax.set_xlim(left=10e-7,right=10e-1)
fig2.tight_layout()
fig2.savefig("figures/attention_bioprior_grid.png", dpi=300)
