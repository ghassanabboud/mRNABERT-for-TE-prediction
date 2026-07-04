"""
Extract dense attention-score vs. LinearFold-contact matrices for example sequences.

study_attention_ss_correlation.py summarizes, per layer, whether attention scores
correlate with LinearFold-predicted contacts -- but it only keeps a sampled subset
of token pairs per sequence, not the full (L, L) matrix, since it's built for
aggregate correlation statistics rather than visualization. This script picks a
handful of example transcripts that show the strongest per-sequence correlation
for one chosen layer (ranked from an existing pairs CSV) and re-runs inference on
just those to save the full dense attention matrix and the full dense LinearFold
contact matrix, for side-by-side heatmap plotting (see
figure_scripts/attention_contact_heatmap_plot.py).

Example:
    python study_attention_heatmap_examples.py \\
        --checkpoint_path outputs/cv_biased_full_1024_frozen_1_layer_no_bias/val_fold_4_test_fold_3 \\
        --test_csv_path processed_data_RiboNN/cv_full/val_fold_4_test_fold_3/test.csv \\
        --linearfold_bias_file processed_data_RiboNN/all_lf_bias.npz \\
        --input_pairs_csv outputs/attention_correlation/balanced_sampling_200_seqs/attention_correlation_results.csv \\
        --layer_idx 9 --num_examples 2 \\
        --output_npz outputs/attention_correlation/heatmap_examples.npz
"""

import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from bias import build_wc_lookup, mRNABERTWithBioPriorHead
from finetuning import SupervisedDataCollator
from utils.analysis import (
    extract_dense_matrices,
    patch_backbone_attention,
    patch_bioprior_attention,
    select_top_correlated_sequences,
)

CHECKPOINT_PATH = "outputs/cv_biased_full_1024_frozen_1_layer_no_bias/val_fold_4_test_fold_3"
BIAS_NPZ_PATH = "processed_data_RiboNN/all_lf_bias.npz"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract dense attention/contact matrices for the top correlated example sequences."
    )
    parser.add_argument("--checkpoint_path", type=str, default=CHECKPOINT_PATH,
                         help="Path to the trained checkpoint to load. Its bio_prior_config.json "
                              "supplies the architecture (num_heads, num_bio_layers, num_labels) "
                              "and the bias mode it was trained with.")
    parser.add_argument("--linearfold_bias_file", type=str, default=BIAS_NPZ_PATH,
                         help="Path to the LinearFold .npz used both for contact lookup and, when "
                              "--bias linearfold, as the model's own bio_prior_bias input.")
    parser.add_argument("--test_csv_path", type=str, required=True,
                         help="Path to the test.csv holding the sequences for the selected transcripts.")
    parser.add_argument("--input_pairs_csv", type=str, required=True,
                         help="Existing per-(layer, sequence, token pair) CSV, as produced by "
                              "study_attention_ss_correlation.py, used to rank sequences by "
                              "per-sequence correlation without needing a GPU pass.")
    parser.add_argument("--layer_idx", type=int, required=True,
                         help="Integer layer index matching the 'layer' column of --input_pairs_csv "
                              "(0..num_backbone_layers-1 for backbone layers, then bio-prior layers).")
    parser.add_argument("--num_examples", type=int, default=2,
                         help="Number of top-correlated example transcripts to extract.")
    parser.add_argument("--min_positive", type=int, default=5,
                         help="Minimum number of positive (contact) pairs a transcript must have in "
                              "--input_pairs_csv to be considered for selection.")
    parser.add_argument("--output_npz", type=str, required=True,
                         help="Path to save the extracted dense matrices.")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pairs_df = pd.read_csv(args.input_pairs_csv)
    selected = select_top_correlated_sequences(
        pairs_df, args.layer_idx, args.num_examples, min_positive=args.min_positive
    )
    if not selected:
        raise ValueError(
            f"No transcripts in {args.input_pairs_csv} had >= {args.min_positive} positive pairs "
            f"for layer {args.layer_idx}."
        )
    print(f"Selected {len(selected)} example transcripts for layer {args.layer_idx}:")
    for tx_id, rho in selected:
        print(f"  {tx_id}: spearman_r={rho:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    model, cfg = mRNABERTWithBioPriorHead.from_checkpoint(args.checkpoint_path, device=device)
    bias_mode = cfg["bias"]
    print(f"Loaded checkpoint: bias={bias_mode}  num_heads={cfg['num_heads']}  num_bio_layers={cfg['num_bio_layers']}")

    if bias_mode == "linearfold" and not args.linearfold_bias_file:
        raise ValueError(
            "This checkpoint was trained with bias=linearfold; pass --linearfold_bias_file."
        )

    patch_backbone_attention(model)
    patch_bioprior_attention(model)

    wc_lookup = None
    if bias_mode in ("utr_only", "full"):
        wc_lookup = build_wc_lookup(tokenizer, utr_only=(bias_mode == "utr_only"))

    data_collator = SupervisedDataCollator(
        tokenizer=tokenizer,
        bias_mode=bias_mode,
        wc_lookup=wc_lookup,
        bias_npz_path=args.linearfold_bias_file,
    )

    seq_lookup = pd.read_csv(args.test_csv_path, usecols=["tx_id", "sequence"]).set_index("tx_id")["sequence"]
    bias_lookup = dict(np.load(args.linearfold_bias_file, allow_pickle=False))

    output = {}
    for tx_id, rho in selected:
        sequence = seq_lookup.loc[tx_id]
        bias_pairs = bias_lookup[tx_id]
        attn_matrix, contact_matrix, tokens, start_codon_idx = extract_dense_matrices(
            model, tokenizer, data_collator, cfg, sequence, tx_id, bias_pairs, args.layer_idx, device
        )
        output[f"{tx_id}__attn_matrix"] = attn_matrix
        output[f"{tx_id}__contact_matrix"] = contact_matrix
        output[f"{tx_id}__tokens"] = np.array(tokens)
        output[f"{tx_id}__spearman_r"] = np.array(rho)
        output[f"{tx_id}__start_codon_idx"] = np.array(start_codon_idx)

    output["tx_ids"] = np.array([tx_id for tx_id, _ in selected])
    output["layer_idx"] = np.array(args.layer_idx)

    np.savez(args.output_npz, **output)
    print(f"Saved dense matrices for {len(selected)} transcripts to {args.output_npz}")


if __name__ == "__main__":
    main()
