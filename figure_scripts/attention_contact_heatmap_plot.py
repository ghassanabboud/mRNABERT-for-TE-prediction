import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

INPUT_NPZ = "outputs/attention_correlation/heatmap_examples.npz"
OUTPUT_PATH = "figures/attention_contact_heatmap_examples.png"

data = np.load(INPUT_NPZ, allow_pickle=False)
tx_ids = data["tx_ids"]
layer_idx = int(data["layer_idx"])

fig, axes = plt.subplots(len(tx_ids), 2, figsize=(11, 5 * len(tx_ids)))
if len(tx_ids) == 1:
    axes = axes[None, :]

for row, tx_id in enumerate(tx_ids):
    contact_matrix = data[f"{tx_id}__contact_matrix"]
    attn_matrix = data[f"{tx_id}__attn_matrix"]
    rho = float(data[f"{tx_id}__spearman_r"])

    ax_contact, ax_attn = axes[row]

    im_contact = ax_contact.imshow(contact_matrix, cmap="Blues", vmin=0, vmax=3)
    ax_contact.set_title(f"{tx_id}: LinearFold contact map")
    ax_contact.set_xlabel("Token position")
    ax_contact.set_ylabel("Token position")
    fig.colorbar(im_contact, ax=ax_contact, fraction=0.046, pad=0.04, label="Base-pair count")

    attn_floor = attn_matrix[attn_matrix > 0].min() if (attn_matrix > 0).any() else 1e-8
    im_attn = ax_attn.imshow(
        attn_matrix, cmap="Blues", norm=LogNorm(vmin=attn_floor, vmax=attn_matrix.max())
    )
    ax_attn.set_title(f"{tx_id}: attention score (layer {layer_idx}, ρ={rho:.2f})")
    ax_attn.set_xlabel("Token position")
    ax_attn.set_ylabel("Token position")
    fig.colorbar(im_attn, ax=ax_attn, fraction=0.046, pad=0.04, label="Attention score (log scale)")

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=300)
print(f"Saved figure to {OUTPUT_PATH}")
