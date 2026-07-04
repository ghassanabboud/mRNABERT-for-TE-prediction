"""
Insertional analysis: effect of AUG motif insertion on predicted translation efficiency (TE).

For each qualifying transcript in the test set, insert an AUG codon at every admissible
position around the annotated start codon (out-of-frame in the 5'UTR, in-frame in the CDS),
run the fine-tuned mRNABERT checkpoint on each variant, and record the predicted mean TE.

Example:
    python study_AUG_insertion.py \\
        --checkpoint_path outputs/cv_biased_full_1024_frozen_1_layer_no_bias/val_fold_4_test_fold_3 \\
        --test_csv_path processed_data_RiboNN/cv_full/val_fold_4_test_fold_3/test.csv \\
        --max_sequences 200 --output_csv_path insertional_analysis_results.csv
"""

import argparse

import pandas as pd
import torch
from transformers import AutoTokenizer

from bias import mRNABERTWithBioPriorHead
from utils.analysis import find_utr5_cds_boundaries, generate_variants

MOTIFS = ["ATG"]

MIN_UTR5_LEN = 501
MIN_CDS_LEN = 300

def parse_args():
    parser = argparse.ArgumentParser(description="uAUG insertion analysis on a fine-tuned mRNABERT checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                         help="Path to the trained checkpoint to load.")
    parser.add_argument("--test_csv_path", type=str, required=True,
                         help="Path to the test.csv of qualifying transcripts to run the insertion analysis on.")
    parser.add_argument("--upstream_window", type=int, default=200,
                         help="Number of nucleotide positions upstream of the start codon to scan.")
    parser.add_argument("--downstream_window", type=int, default=100,
                         help="Number of in-frame nucleotide positions downstream (within the CDS) to scan.")
    parser.add_argument("--output_csv_path", type=str, required=True,
                         help="Path to write the per-variant predictions CSV.")
    parser.add_argument("--max_sequences", type=int, required=True,
                         help="Cap on the number of qualifying transcripts to process (use -1 to run on all).")
    parser.add_argument("--batch_size", type=int, default=32,
                         help="Batch size for model inference.")
                    
    return parser.parse_args()


def main():
    args = parse_args()
    max_sequences = None if args.max_sequences is not None and args.max_sequences < 0 else args.max_sequences

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    print(f"model_max_length inferred from checkpoint tokenizer: {tokenizer.model_max_length}")
    model, cfg = mRNABERTWithBioPriorHead.from_checkpoint(args.checkpoint_path, device=device)
    if cfg["bias"] != "no_bias":
        raise ValueError(
            f"This checkpoint was trained with bias='{cfg['bias']}'; this script only calls the "
            "model with bio_prior_bias=None, so it is only supported for checkpoints trained with "
            "bias='no_bias' so far."
        )

    df = pd.read_csv(args.test_csv_path, usecols=["tx_id", "sequence"])

    records = []
    num_valid_sequences = 0
    for tx_id, sequence in zip(df["tx_id"], df["sequence"]):
        if max_sequences is not None and num_valid_sequences >= max_sequences:
            break

        tokens = sequence.split()
        utr5_len_nt, num_cds_codons = find_utr5_cds_boundaries(tokens)
        cds_len_nt = num_cds_codons * 3

        if utr5_len_nt <= MIN_UTR5_LEN or cds_len_nt < MIN_CDS_LEN:
            continue

        num_valid_sequences += 1
        for motif in MOTIFS:
            records.extend(generate_variants(
                tx_id, tokens, utr5_len_nt, num_cds_codons, motif,
                upstream_window=args.upstream_window,
                downstream_window=args.downstream_window,
            ))

    print(f"Found {num_valid_sequences} qualifying transcripts with UTR5 > {MIN_UTR5_LEN} nt and CDS > {MIN_CDS_LEN} nt")
    print(f"Generated {len(records)} sequence variants from qualifying transcripts")

    predicted_mean_te = []
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            seqs = [r[2] for r in batch]
            inputs = tokenizer(
                seqs,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predicted_mean_te.extend(logits.mean(dim=1).cpu().tolist())

            if (start // args.batch_size) % 20 == 0:
                print(f"Processed {start + len(batch)}/{len(records)} variants")

    result_df = pd.DataFrame({
        "tx_id": [r[0] for r in records],
        "insertion_position": [r[1] for r in records],
        "sequence": [r[2] for r in records],
        "predicted_mean_TE": predicted_mean_te,
    })
    result_df.to_csv(args.output_csv_path, index=False)
    print(f"Saved {len(result_df)} rows to {args.output_csv_path}")


if __name__ == "__main__":
    main()
