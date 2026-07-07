import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.analysis import find_utr5_cds_boundaries

INPUT_NPZ = "./outputs/attention_correlation/balanced_sampling_400_seqs_distance_matched_ffn/multi_model_multi_layer_examples.npz"
TEST_CSV_PATH = "processed_data_RiboNN/cv_full/val_fold_4_test_fold_3/test.csv"
OUTPUT_PATH = "figures/attention_contact_heatmap_examples_multi_model_multi_layer_ffn.png"
OUTPUT_PATH_SINGLE = "figures/attention_contact_heatmap_overlay_single.png"

KEEP_INDICES = range(5)
KEEP_INDEX_END = 1
#KEEP_MODELS_AT_LAYERS = [("no_bias", 0), ("wc", 0), ("linearfold", 0), ("no_bias", 12)]
KEEP_MODELS_AT_LAYERS = [("no_bias", 12)]
ATTN_VMIN = 0.001
ATTN_VMAX = 0.005

REGION_CMAP = ListedColormap(["red", "blue", "green"])
REGION_LABELS = ["5'UTR", "CDS", "3'UTR"]


def region_track(seq_len, start_codon_idx, num_cds_codons):
    """Return a (1, seq_len) array labeling each token position 0=5'UTR, 1=CDS, 2=3'UTR.

    The leading CLS token is folded into the 5'UTR region and a trailing SEP
    (if present, i.e. the sequence wasn't truncated) into the 3'UTR region.
    """
    cds_end_idx = start_codon_idx + num_cds_codons  # first token after the stop codon
    track = np.zeros((1, seq_len), dtype=np.int32)
    track[0, start_codon_idx:min(cds_end_idx, seq_len)] = 1
    track[0, min(cds_end_idx, seq_len):seq_len] = 2
    return track


def add_region_bar(ax, region_track):
    divider = make_axes_locatable(ax)
    bar_ax = divider.append_axes("bottom", size="4%", pad=0.35)
    bar_ax.imshow(region_track, aspect="auto", cmap=REGION_CMAP, vmin=0, vmax=2)
    bar_ax.set_xlim(ax.get_xlim())
    bar_ax.set_yticks([])
    bar_ax.set_xlabel(ax.get_xlabel())
    ax.set_xlabel("")
    ax.set_xticklabels([])

    n_regions = len(REGION_LABELS)
    for region_id, label in enumerate(REGION_LABELS):
        positions = np.nonzero(region_track[0] == region_id)[0]
        if len(positions) == 0:
            continue
        if region_id == 0:
            x, ha = positions[0], "left"
        elif region_id == n_regions - 1:
            x, ha = positions[-1], "right"
        else:
            x, ha = (positions[0] + positions[-1]) / 2, "center"
        bar_ax.text(
            x, 0, label,
            ha=ha, va="center", fontsize=7, color="white",
            fontweight="bold", clip_on=False,
        )


seq_lookup = pd.read_csv(TEST_CSV_PATH, usecols=["tx_id", "sequence"]).set_index("tx_id")["sequence"]

data = np.load(INPUT_NPZ, allow_pickle=False)
tx_ids = data["tx_ids"][KEEP_INDICES]
first_model_name = KEEP_MODELS_AT_LAYERS[0][0]

n_cols = 1 + len(KEEP_MODELS_AT_LAYERS)
fig, axes = plt.subplots(len(tx_ids), n_cols, figsize=(5.5 * n_cols, 5 * len(tx_ids)))
if len(tx_ids) == 1:
    axes = axes[None, :]

for row, tx_id in enumerate(tx_ids):
    rho = float(data[f"{tx_id}__spearman_r"])
    contact_matrix = data[f"{first_model_name}__{tx_id}__contact_matrix"]

    tokens = data[f"{first_model_name}__{tx_id}__tokens"]
    sequence = "".join(tok for tok in tokens if len(tok) <= 3)
    start_codon_idx = int(data[f"{first_model_name}__{tx_id}__start_codon_idx"])

    full_sequence = seq_lookup.loc[tx_id]
    _, num_cds_codons = find_utr5_cds_boundaries(full_sequence.split(" "))
    end_codon_idx = start_codon_idx + num_cds_codons - 1  # position of the stop codon
    print(f"{tx_id} (start codon at token {start_codon_idx}, stop codon at token {end_codon_idx}): {sequence}")

    row_region_track = region_track(len(tokens), start_codon_idx, num_cds_codons)

    ax_contact = axes[row][0]
    contact_i, contact_j = np.nonzero(contact_matrix)
    ax_contact.set_facecolor("white")
    ax_contact.scatter(contact_j, contact_i, c="red", s=4, marker="s", linewidths=0)
    ax_contact.set_xlim(0, contact_matrix.shape[1])
    ax_contact.set_ylim(contact_matrix.shape[0], 0)
    ax_contact.set_aspect("equal")
    ax_contact.set_title(f"{tx_id}: LinearFold contact map")
    ax_contact.set_xlabel("Token position")
    ax_contact.set_ylabel("Token position")
    add_region_bar(ax_contact, row_region_track)

    for col, (model_name, layer_idx) in enumerate(KEEP_MODELS_AT_LAYERS, start=1):
        attn_matrix = data[f"{model_name}__{layer_idx}__{tx_id}__attn_matrix"]
        ax_attn = axes[row][col]

        im_attn = ax_attn.imshow(
            #attn_matrix, cmap="Blues", norm=LogNorm(vmin=ATTN_VMIN, vmax=ATTN_VMAX)
            attn_matrix, cmap="Blues", norm=LogNorm()
        )
        ax_attn.scatter(contact_j, contact_i, c="red", s=4, marker="s", linewidths=0, alpha=0.15)
        ax_attn.set_xlim(0, attn_matrix.shape[1])
        ax_attn.set_ylim(attn_matrix.shape[0], 0)

        title = f"{tx_id}: {model_name} attention (layer {layer_idx}"
        if col == 1:
            title += f", ρ={rho:.2f}"
        title += ")"
        ax_attn.set_title(title)
        ax_attn.set_xlabel("Token position")
        ax_attn.set_ylabel("Token position")
        fig.colorbar(im_attn, ax=ax_attn, fraction=0.046, pad=0.04, label="Attention score (log scale)")
        add_region_bar(ax_attn, row_region_track)

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=100)
print(f"Saved figure to {OUTPUT_PATH}")

single_tx_id = data["tx_ids"][KEEP_INDEX_END]
single_model_name, single_layer_idx = KEEP_MODELS_AT_LAYERS[0]
single_contact_matrix = data[f"{single_model_name}__{single_tx_id}__contact_matrix"]
single_attn_matrix = data[f"{single_model_name}__{single_layer_idx}__{single_tx_id}__attn_matrix"]
single_contact_i, single_contact_j = np.nonzero(single_contact_matrix)

single_tokens = data[f"{single_model_name}__{single_tx_id}__tokens"]
single_start_codon_idx = int(data[f"{single_model_name}__{single_tx_id}__start_codon_idx"])
single_full_sequence = seq_lookup.loc[single_tx_id]
_, single_num_cds_codons = find_utr5_cds_boundaries(single_full_sequence.split(" "))
single_region_track = region_track(len(single_tokens), single_start_codon_idx, single_num_cds_codons)

fig_single, ax_single = plt.subplots(figsize=(5.5, 5))
im_single = ax_single.imshow(single_attn_matrix, cmap="Blues", norm=LogNorm())
ax_single.scatter(single_contact_j, single_contact_i, c="red", s=4, marker="s", linewidths=0, alpha=1)
ax_single.set_xlim(0, single_attn_matrix.shape[1])
ax_single.set_ylim(single_attn_matrix.shape[0], 0)
ax_single.set_xlabel("Token position")
ax_single.set_ylabel("Token position")
fig_single.colorbar(im_single, ax=ax_single, fraction=0.046, pad=0.04, label="Attention score (log scale)")
add_region_bar(ax_single, single_region_track)

fig_single.tight_layout()
fig_single.savefig(OUTPUT_PATH_SINGLE, dpi=100)
print(f"Saved figure to {OUTPUT_PATH_SINGLE} (tx_id={single_tx_id})")
