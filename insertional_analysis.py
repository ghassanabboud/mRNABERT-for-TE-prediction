"""
Insertional analysis: effect of AUG motif insertion on predicted translation efficiency (TE).

For each qualifying transcript in the test set, insert an AUG codon at every admissible
position around the annotated start codon (out-of-frame in the 5'UTR, in-frame in the CDS),
run the fine-tuned mRNABERT checkpoint on each variant, and record the predicted mean TE.

See experiments/09-uAUG_insertion.md for the motivating analysis.
"""

import argparse

import pandas as pd
import torch

from utils.analysis import find_utr5_cds_boundaries, generate_variants, load_model

CHECKPOINT_PATH = "outputs/cv_biased_full_1024_frozen_1_layer_no_bias/val_fold_4_test_fold_3"
BASE_MODEL_NAME = "YYLY66/mRNABERT"
TEST_CSV_PATH = "processed_data_RiboNN/cv_full/val_fold_4_test_fold_3/test.csv"
MOTIFS = ["ATG"]

NUM_HEADS = 8
NUM_BIO_LAYERS = 1
NUM_LABELS = 78
MODEL_MAX_LENGTH = 1024
MIN_UTR5_LEN = 501
MIN_CDS_LEN = 300

def parse_args():
    parser = argparse.ArgumentParser(description="uAUG insertion analysis on a fine-tuned mRNABERT checkpoint.")
    parser.add_argument("--upstream_window", type=int, default=200,
                         help="Number of nucleotide positions upstream of the start codon to scan.")
    parser.add_argument("--downstream_window", type=int, default=100,
                         help="Number of in-frame nucleotide positions downstream (within the CDS) to scan.")
    parser.add_argument("--output_csv_path", type=str, default="insertional_analysis_results.csv",
                         help="Path to write the per-variant predictions CSV.")
    parser.add_argument("--max_sequences", type=int, default=200,
                         help="Cap on the number of qualifying transcripts to process (use -1 to run on all).")
    parser.add_argument("--batch_size", type=int, default=32,
                         help="Batch size for model inference.")
                    
    return parser.parse_args()


def main():
    args = parse_args()
    max_sequences = None if args.max_sequences is not None and args.max_sequences < 0 else args.max_sequences

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer, model = load_model(
        device,
        checkpoint_path=CHECKPOINT_PATH,
        base_model_name=BASE_MODEL_NAME,
        model_max_length=MODEL_MAX_LENGTH,
        num_heads=NUM_HEADS,
        num_labels=NUM_LABELS,
        num_bio_layers=NUM_BIO_LAYERS,
    )

    df = pd.read_csv(TEST_CSV_PATH, usecols=["tx_id", "sequence"])

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
        "insertion_position": [r[1] for r in records],
        "sequence": [r[2] for r in records],
        "predicted_mean_TE": predicted_mean_te,
    })
    result_df.to_csv(args.output_csv_path, index=False)
    print(f"Saved {len(result_df)} rows to {args.output_csv_path}")


if __name__ == "__main__":
    main()
