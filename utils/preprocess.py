"""
Preprocessing utilities and entry point for the RiboNN dataset.

Run as a script to produce a single train/dev/test split:
    python utils/preprocess.py --sequence_mode utr5_cds --val_fold 8 --test_fold 9 \
        --output_dir processed_data_RiboNN/

For 10-fold cross-validation splits, see utils/crossvalidation.py.
"""

import argparse
import os

import pandas as pd

RIBONN_DATA_PATH = "/scratch/izar/gabboud/mRNABERT/excel_data_RiboNN/41587_2025_2712_MOESM3_ESM.xlsx"


# ---------------------------------------------------------------------------
# Sequence extraction helpers
# ---------------------------------------------------------------------------

def extract_cds(row):
    seq, utr5_len, cds_len = row["tx_sequence"], row["utr5_size"], row["cds_size"]
    if cds_len % 3 != 0:
        raise ValueError(f"CDS length {cds_len} not divisible by 3 for tx_id {row['tx_id']}")
    codon_list = [seq[i:i + 3] for i in range(utr5_len, utr5_len + cds_len, 3)]
    return " ".join(codon_list)


def extract_utr5(row):
    return " ".join(row["tx_sequence"][:row["utr5_size"]])


def extract_utr3(row):
    seq, utr5_len, cds_len = row["tx_sequence"], row["utr5_size"], row["cds_size"]
    return " ".join(seq[utr5_len + cds_len:])


def extract_utr5_cds(row, max_cds_length=None):
    if max_cds_length is None:
        return extract_utr5(row) + " " + extract_cds(row)
    if max_cds_length % 3 != 0:
        raise ValueError(f"max_cds_length {max_cds_length} is not divisible by 3.")
    seq, utr5_len, cds_len = row["tx_sequence"], row["utr5_size"], row["cds_size"]
    if cds_len % 3 != 0:
        raise ValueError(f"CDS length {cds_len} not divisible by 3 for tx_id {row['tx_id']}")
    end_cds = min(utr5_len + cds_len, utr5_len + max_cds_length)
    codon_list = [seq[i:i + 3] for i in range(utr5_len, end_cds, 3)]
    return " ".join(seq[:utr5_len]) + " " + " ".join(codon_list)


def extract_full_sequence(row):
    return extract_utr5(row) + " " + extract_cds(row) + " " + extract_utr3(row)


def extract_start_codon_window(row, total_window_length):
    seq, utr5_len, cds_len = row["tx_sequence"], row["utr5_size"], row["cds_size"]
    half = total_window_length // 2
    if half % 3 != 0:
        raise ValueError(f"Half window {half} not divisible by 3 for tx_id {row['tx_id']}")
    start = max(0, utr5_len - half)
    end = min(utr5_len + cds_len, utr5_len + half)
    return (
        " ".join(seq[start:utr5_len])
        + " "
        + " ".join([seq[i:i + 3] for i in range(utr5_len, end, 3)])
    )


# ---------------------------------------------------------------------------
# Core export function (used by both preprocess.py and crossvalidation.py)
# ---------------------------------------------------------------------------

def export_sequences_for_mrnabert(
    output_file,
    folds=None,
    sequence_mode="utr5_cds",
    total_window_length=None,
    max_cds_length=None,
    data_path=RIBONN_DATA_PATH,
):
    """Load the RiboNN Excel file, apply sequence extraction, and write a CSV."""
    df = pd.read_excel(data_path)

    if folds is not None:
        df = df[df["fold"].isin(folds)]
        if df.empty:
            raise ValueError(f"No sequences found for folds {folds}")

    mode_fns = {
        "full": extract_full_sequence,
        "cds_only": extract_cds,
        "utr5_only": extract_utr5,
        "utr3_only": extract_utr3,
    }

    if sequence_mode in mode_fns:
        df["sequence"] = df.apply(mode_fns[sequence_mode], axis=1)
    elif sequence_mode == "utr5_cds":
        df["sequence"] = df.apply(extract_utr5_cds, axis=1, max_cds_length=max_cds_length)
    elif sequence_mode == "start_codon_window":
        if total_window_length is None:
            raise ValueError("total_window_length required for sequence_mode='start_codon_window'")
        df["sequence"] = df.apply(
            extract_start_codon_window, axis=1, total_window_length=total_window_length
        )
    else:
        valid = list(mode_fns) + ["utr5_cds", "start_codon_window"]
        raise ValueError(f"Invalid sequence_mode '{sequence_mode}'. Expected one of {valid}")

    cols = ["tx_id", "sequence"] + [c for c in df.columns if c.startswith("TE_")]
    df[cols].to_csv(output_file, index=False)


# ---------------------------------------------------------------------------
# Entry point: single train/dev/test split
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess RiboNN dataset for mRNABERT fine-tuning")
    parser.add_argument("--data_path", default=RIBONN_DATA_PATH)
    parser.add_argument("--output_dir", default="./processed_data_RiboNN/")
    parser.add_argument("--output_path", default=None, help="Overrides output_dir if set")
    parser.add_argument("--sequence_mode", default="utr5_cds",
                        choices=["full", "cds_only", "utr5_only", "utr3_only", "utr5_cds", "start_codon_window"])
    parser.add_argument("--total_window_length", type=int, default=None)
    parser.add_argument("--max_cds_length", type=int, default=None)
    parser.add_argument("--val_fold", type=int, default=8)
    parser.add_argument("--test_fold", type=int, default=9)
    args = parser.parse_args()

    if args.sequence_mode == "start_codon_window" and args.total_window_length:
        mode_label = f"{args.sequence_mode}_{args.total_window_length}nt"
    elif args.sequence_mode == "utr5_cds" and args.max_cds_length:
        mode_label = f"{args.sequence_mode}_{args.max_cds_length}nt"
    else:
        mode_label = args.sequence_mode

    out = args.output_path or os.path.join(
        args.output_dir,
        f"{mode_label}_val_fold_{args.val_fold}_test_fold_{args.test_fold}",
    )
    os.makedirs(out, exist_ok=True)
    print(f"Writing to {out}  (mode={args.sequence_mode}, val={args.val_fold}, test={args.test_fold})")

    all_folds = list(range(10))
    shared = dict(
        sequence_mode=args.sequence_mode,
        total_window_length=args.total_window_length,
        max_cds_length=args.max_cds_length,
        data_path=args.data_path,
    )
    export_sequences_for_mrnabert(
        os.path.join(out, "train.csv"),
        folds=[f for f in all_folds if f not in [args.val_fold, args.test_fold]],
        **shared,
    )
    export_sequences_for_mrnabert(os.path.join(out, "dev.csv"),  folds=[args.val_fold],  **shared)
    export_sequences_for_mrnabert(os.path.join(out, "test.csv"), folds=[args.test_fold], **shared)


if __name__ == "__main__":
    main()
