"""
Preprocessing utilities and entry point for the RiboNN dataset.

Run as a script to produce a single train/dev/test split:
    python preprocess_one_split.py --sequence_mode utr5_cds --val_fold 8 --test_fold 9 \
        --output_dir processed_data_RiboNN/

For 10-fold cross-validation splits, see utils/crossvalidation.py.
"""


import argparse
import os

from utils.preprocessing import export_sequences_for_mrnabert


def main():
    parser = argparse.ArgumentParser(description="Preprocess RiboNN dataset for mRNABERT fine-tuning")
    parser.add_argument("--data_path", required=True)
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
