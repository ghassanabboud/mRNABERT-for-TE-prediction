"""
Extract dense attention-score vs. LinearFold-contact matrices for example
sequences, across three bias variants of the same architecture.

study_attention_heatmap_examples.py does this for a single checkpoint and a
single layer. This version hardcodes three checkpoints -- no_bias, wc
(Watson-Crick), and linearfold -- the no_bias model's --input_pairs_csv, and
a list of layers (LAYERS) to extract. Example transcripts are selected once,
by per-sequence attention/contact correlation at SELECTION_LAYER using only
the no_bias model's pairs CSV (as the original script does), then re-run
through all three checkpoints at every layer in LAYERS, saving each
model/layer's dense attention matrix alongside the shared LinearFold contact
matrix for side-by-side heatmap plotting.

Example:
    python study_attention_heatmap_examples_multi_model.py \\
        --test_csv_path processed_data_RiboNN/cv_full/val_fold_4_test_fold_3/test.csv \\
        --linearfold_bias_file processed_data_RiboNN/all_lf_bias.npz \\
        --num_examples 2 \\
        --output_npz outputs/attention_correlation/heatmap_examples_multi_model.npz
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

CHECKPOINT_PATHS = {
    "no_bias": "outputs/ffn_cv_biased_full_1024_frozen_1_layer_no_bias/val_fold_4_test_fold_3",
    "wc": "outputs/ffn_cv_FIXED_biased_full_1024_frozen_1_layer_wc_bias/val_fold_4_test_fold_3",
    "linearfold": "outputs/ffn_cv_biased_full_1024_frozen_1_layer_lf_bias/val_fold_4_test_fold_3",
}

BIAS_NPZ_PATH = "processed_data_RiboNN/all_lf_bias.npz"
INPUT_PAIRS_CSV = "outputs/attention_correlation/balanced_sampling_400_seqs_distance_matched_ffn/attention_correlation_results.csv"

LAYERS = [0, 12]
SELECTION_LAYER = 12  # layer used to rank/select example transcripts; must be one of LAYERS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract dense attention/contact matrices for the top correlated example "
                    "sequences, run through the no_bias, wc, and linearfold checkpoints."
    )
    parser.add_argument("--linearfold_bias_file", type=str, default=BIAS_NPZ_PATH,
                         help="Path to the LinearFold .npz used both for contact lookup and, for "
                              "the linearfold checkpoint, as the model's own bio_prior_bias input.")
    parser.add_argument("--test_csv_path", type=str, required=True,
                         help="Path to the test.csv holding the sequences for the selected transcripts.")
    parser.add_argument("--num_examples", type=int, default=2,
                         help="Number of top-correlated example transcripts to extract.")
    parser.add_argument("--min_positive", type=int, default=5,
                         help="Minimum number of positive (contact) pairs a transcript must have in "
                              "the no_bias pairs CSV to be considered for selection.")
    parser.add_argument("--output_npz", type=str, required=True,
                         help="Path to save the extracted dense matrices.")
    return parser.parse_args()


def load_model_and_collator(checkpoint_path, linearfold_bias_file, device):
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    model, cfg = mRNABERTWithBioPriorHead.from_checkpoint(checkpoint_path, device=device)
    bias_mode = cfg["bias"]
    print(f"Loaded checkpoint {checkpoint_path}: bias={bias_mode}  num_heads={cfg['num_heads']}  "
          f"num_bio_layers={cfg['num_bio_layers']}")

    if bias_mode == "linearfold" and not linearfold_bias_file:
        raise ValueError(
            f"Checkpoint {checkpoint_path} was trained with bias=linearfold; "
            "pass --linearfold_bias_file."
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
        bias_npz_path=linearfold_bias_file,
    )
    return tokenizer, model, cfg, data_collator


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pairs_df = pd.read_csv(INPUT_PAIRS_CSV)
    selected = select_top_correlated_sequences(
        pairs_df, SELECTION_LAYER, args.num_examples, min_positive=args.min_positive
    )
    if not selected:
        raise ValueError(
            f"No transcripts in {INPUT_PAIRS_CSV} had >= {args.min_positive} positive pairs "
            f"for layer {SELECTION_LAYER}."
        )
    print(f"Selected {len(selected)} example transcripts for layer {SELECTION_LAYER} "
          "(ranked by the no_bias model):")
    for tx_id, rho in selected:
        print(f"  {tx_id}: spearman_r={rho:.3f}")

    seq_lookup = pd.read_csv(args.test_csv_path, usecols=["tx_id", "sequence"]).set_index("tx_id")["sequence"]
    bias_lookup = dict(np.load(args.linearfold_bias_file, allow_pickle=False))

    output = {}
    for model_name, checkpoint_path in CHECKPOINT_PATHS.items():
        tokenizer, model, cfg, data_collator = load_model_and_collator(
            checkpoint_path, args.linearfold_bias_file, device
        )

        for tx_id, rho in selected:
            sequence = seq_lookup.loc[tx_id]
            bias_pairs = bias_lookup[tx_id]
            for layer_idx in LAYERS:
                attn_matrix, contact_matrix, tokens, start_codon_idx = extract_dense_matrices(
                    model, tokenizer, data_collator, cfg, sequence, tx_id, bias_pairs, layer_idx, device
                )
                output[f"{model_name}__{layer_idx}__{tx_id}__attn_matrix"] = attn_matrix
            output[f"{model_name}__{tx_id}__contact_matrix"] = contact_matrix
            output[f"{model_name}__{tx_id}__tokens"] = np.array(tokens)
            output[f"{model_name}__{tx_id}__start_codon_idx"] = np.array(start_codon_idx)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for tx_id, rho in selected:
        output[f"{tx_id}__spearman_r"] = np.array(rho)

    output["model_names"] = np.array(list(CHECKPOINT_PATHS.keys()))
    output["tx_ids"] = np.array([tx_id for tx_id, _ in selected])
    output["layers"] = np.array(LAYERS)

    np.savez(args.output_npz, **output)
    print(f"Saved dense matrices for {len(selected)} transcripts x {len(CHECKPOINT_PATHS)} models "
          f"x {len(LAYERS)} layers to {args.output_npz}")


if __name__ == "__main__":
    main()