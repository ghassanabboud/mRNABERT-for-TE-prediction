"""
Codon optimality analysis: effect of CDS codon usage on predicted translation efficiency (TE).

For each qualifying transcript in the test set (5'UTR no longer than MAX_UTR5_LEN nt), compute
the wildtype CAI, generate a most-optimal-codon and a least-optimal-codon variant of the CDS,
run the fine-tuned mRNABERT checkpoint on all three sequences, and record the predicted mean TE
alongside the CAI of each variant.

Example:
    python study_codon_optimality.py \\
        --checkpoint_path outputs/cv_biased_full_1024_frozen_1_layer_no_bias/val_fold_4_test_fold_3 \\
        --test_csv_path processed_data_RiboNN/cv_full/val_fold_4_test_fold_3/test.csv \\
        --max_sequences 200 --output_csv_path codon_optimality_analysis_results.csv
"""

import argparse

import pandas as pd
import torch

from utils.analysis import (
    find_utr5_cds_boundaries,
    get_cai,
    get_max_usage_sequence,
    get_min_usage_sequence,
    load_model,
)

BASE_MODEL_NAME = "YYLY66/mRNABERT"

NUM_HEADS = 8
NUM_BIO_LAYERS = 1
NUM_LABELS = 78
MODEL_MAX_LENGTH = 1024
MAX_UTR5_LEN = 300


def parse_args():
    parser = argparse.ArgumentParser(description="Codon optimality analysis on a fine-tuned mRNABERT checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                         help="Path to the trained checkpoint to load.")
    parser.add_argument("--test_csv_path", type=str, required=True,
                         help="Path to the test.csv of qualifying transcripts to run the codon optimality analysis on.")
    parser.add_argument("--output_csv_path", type=str, default="codon_optimality_analysis_results.csv",
                         help="Path to write the per-variant predictions CSV.")
    parser.add_argument("--max_sequences", type=int, default=200,
                         help="Cap on the number of qualifying transcripts to process (use -1 to run on all).")
    parser.add_argument("--batch_size", type=int, default=64,
                         help="Batch size for model inference.")

    return parser.parse_args()


def main():
    args = parse_args()
    max_sequences = None if args.max_sequences is not None and args.max_sequences < 0 else args.max_sequences

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer, model = load_model(
        device,
        checkpoint_path=args.checkpoint_path,
        base_model_name=BASE_MODEL_NAME,
        model_max_length=MODEL_MAX_LENGTH,
        num_heads=NUM_HEADS,
        num_labels=NUM_LABELS,
        num_bio_layers=NUM_BIO_LAYERS,
    )

    df = pd.read_csv(args.test_csv_path, usecols=["tx_id", "sequence"])

    records = []
    num_valid_sequences = 0
    for tx_id, sequence in zip(df["tx_id"], df["sequence"]):
        if max_sequences is not None and num_valid_sequences >= max_sequences:
            break

        tokens = sequence.split()
        utr5_len_nt, _ = find_utr5_cds_boundaries(tokens)

        if utr5_len_nt > MAX_UTR5_LEN:
            continue

        num_valid_sequences += 1

        wildtype_seq = sequence
        optimal_seq = get_max_usage_sequence(sequence)
        least_optimal_seq = get_min_usage_sequence(sequence)

        records.append((tx_id, "wildtype", wildtype_seq, get_cai(wildtype_seq)))
        records.append((tx_id, "optimal", optimal_seq, get_cai(optimal_seq)))
        records.append((tx_id, "least_optimal", least_optimal_seq, get_cai(least_optimal_seq)))

    print(f"Found {num_valid_sequences} qualifying transcripts with UTR5 <= {MAX_UTR5_LEN} nt")
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
                max_length=MODEL_MAX_LENGTH,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predicted_mean_te.extend(logits.mean(dim=1).cpu().tolist())

            if (start // args.batch_size) % 20 == 0:
                print(f"Processed {start + len(batch)}/{len(records)} variants")

    result_df = pd.DataFrame({
        "tx_id": [r[0] for r in records],
        "variant_type": [r[1] for r in records],
        "sequence": [r[2] for r in records],
        "CAI": [r[3] for r in records],
        "predicted_mean_TE": predicted_mean_te,
    })
    result_df.to_csv(args.output_csv_path, index=False)
    print(f"Saved {len(result_df)} rows to {args.output_csv_path}")


if __name__ == "__main__":
    main()
