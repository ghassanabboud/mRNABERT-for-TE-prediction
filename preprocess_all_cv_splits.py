import argparse
import os

from utils.preprocessing import export_sequences_for_mrnabert


def main():
    parser = argparse.ArgumentParser(description="Build 10-fold CV splits from RiboNN data")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--sequence_mode", default="utr5_cds",
                        choices=["full", "cds_only", "utr5_only", "utr3_only", "utr5_cds", "start_codon_window"])
    parser.add_argument("--total_window_length", type=int, default=None)
    parser.add_argument("--max_cds_length", type=int, default=None)
    args = parser.parse_args()

    if args.sequence_mode == "start_codon_window" and args.total_window_length:
        mode_label = f"{args.sequence_mode}_{args.total_window_length}nt"
    elif args.sequence_mode == "utr5_cds" and args.max_cds_length:
        mode_label = f"{args.sequence_mode}_{args.max_cds_length}nt"
    else:
        mode_label = args.sequence_mode

    out_root = args.output_dir or f"./processed_data_RiboNN/cv_{mode_label}/"
    os.makedirs(out_root, exist_ok=True)
    print(f"Writing 10-fold CV to {out_root}  (mode={args.sequence_mode})")

    shared = dict(
        sequence_mode=args.sequence_mode,
        total_window_length=args.total_window_length,
        max_cds_length=args.max_cds_length,
        data_path=args.data_path,
    )
    all_folds = list(range(10))
    for test_fold in all_folds:
        val_fold = (test_fold + 1) % 10
        fold_dir = os.path.join(out_root, f"val_fold_{val_fold}_test_fold_{test_fold}")
        os.makedirs(fold_dir, exist_ok=True)
        export_sequences_for_mrnabert(
            os.path.join(fold_dir, "train.csv"),
            folds=[f for f in all_folds if f not in [val_fold, test_fold]],
            **shared,
        )
        export_sequences_for_mrnabert(os.path.join(fold_dir, "dev.csv"),  folds=[val_fold],  **shared)
        export_sequences_for_mrnabert(os.path.join(fold_dir, "test.csv"), folds=[test_fold], **shared)


if __name__ == "__main__":
    main()
