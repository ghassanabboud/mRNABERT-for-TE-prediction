import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

INPUT_NPZ = "/scratch/izar/gabboud/mRNABERT/outputs/attention_correlation/balanced_sampling_400_seqs_distance_matched/heatmap_10_examples_with_start_index_layer_12.npz"
OUTPUT_PATH = "figures/attention_contact_heatmap_examples_layer_12_no_bias_distance_matched.png"
WINDOW_SIZE = 100  # tokens shown around the start codon
CROP = False  # whether to crop matrices to WINDOW_SIZE around the start codon


def crop_around_start_codon(matrix, start_codon_idx, window_size):
    """Crop a square (L, L) matrix to a window_size x window_size block centered
    on start_codon_idx, clamped to the matrix bounds."""
    half = window_size // 2
    lo = max(0, start_codon_idx - half)
    hi = min(matrix.shape[0], lo + window_size)
    lo = max(0, hi - window_size)  # shift back if hi got clamped near the end
    return matrix[lo:hi, lo:hi]


data = np.load(INPUT_NPZ, allow_pickle=False)
tx_ids = data["tx_ids"]
layer_idx = int(data["layer_idx"])

fig, axes = plt.subplots(len(tx_ids), 2, figsize=(11, 5 * len(tx_ids)))
if len(tx_ids) == 1:
    axes = axes[None, :]

for row, tx_id in enumerate(tx_ids):
    contact_matrix = data[f"{tx_id}__contact_matrix"]
    attn_matrix = data[f"{tx_id}__attn_matrix"]
    if CROP:
        start_codon_idx = int(data[f"{tx_id}__start_codon_idx"])
        contact_matrix = crop_around_start_codon(contact_matrix, start_codon_idx, WINDOW_SIZE)
        attn_matrix = crop_around_start_codon(attn_matrix, start_codon_idx, WINDOW_SIZE)
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
