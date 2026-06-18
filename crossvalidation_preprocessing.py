import pandas as pd
import os
import argparse
RIBONN_DATA_PATH = "/scratch/izar/gabboud/mRNABERT/excel_data_RiboNN/41587_2025_2712_MOESM3_ESM.xlsx"

def extract_cds(row):
    seq = row["tx_sequence"]
    utr5_len= row["utr5_size"]
    cds_len = row["cds_size"]
    if cds_len % 3 != 0:
        raise ValueError(f"CDS length {cds_len} is not divisible by 3 for tx_id {row['tx_id']}")
    codon_list = [seq[i:i+3] for i in range(utr5_len, utr5_len + cds_len, 3)]
    return " ".join(codon_list)


def extract_utr5(row):
    seq = row["tx_sequence"]
    utr5_len= row["utr5_size"]
    return " ".join(seq[:utr5_len])

def extract_utr3(row):
    seq = row["tx_sequence"]
    utr5_len= row["utr5_size"]
    cds_len = row["cds_size"]
    utr3_seq = seq[utr5_len + cds_len :]
    return " ".join(utr3_seq)

def extract_utr5_cds(row, max_cds_length=None):
    if max_cds_length is None:
        return extract_utr5(row) + " " + extract_cds(row)
    elif max_cds_length % 3 != 0:
        raise ValueError(f"max_cds_length {max_cds_length} is not divisible by 3.")
    else:
        seq = row["tx_sequence"]
        utr5_len= row["utr5_size"]
        cds_len = row["cds_size"]
        if cds_len % 3 != 0:
            raise ValueError(f"CDS length {cds_len} is not divisible by 3 for tx_id {row['tx_id']}")
        end_cds = min(utr5_len + cds_len, utr5_len + max_cds_length)
        codon_list = [seq[i:i+3] for i in range(utr5_len, end_cds, 3)]
        return " ".join(seq[:utr5_len]) + " " + " ".join(codon_list)

def extract_full_sequence(row):
    return extract_utr5(row) + " " + extract_cds(row) + " " + extract_utr3(row)

def extract_start_codon_window(row, total_window_length):
    seq = row["tx_sequence"]
    utr5_len = row["utr5_size"]
    cds_len = row["cds_size"]
    half = total_window_length // 2
    if half % 3 != 0:
        raise ValueError(f"Half of the total window length {half} is not divisible by 3 for tx_id {row['tx_id']}")
    start = max(0, utr5_len - half)
    end = min(utr5_len + cds_len, utr5_len + half)
    return " ".join(seq[start:utr5_len]) + " " + " ".join([seq[i:i+3] for i in range(utr5_len, end, 3)])


def export_sequences_for_mrnabert(output_file, folds=None, sequence_mode="complete", total_window_length=None, max_cds_length=None):
    """
    Exports sequences from the real dataset for a specific fold, transforming the sequence
    """

    df_real = pd.read_excel(RIBONN_DATA_PATH)

    if folds is not None:
        df_real = df_real[df_real["fold"].isin(folds)]
        if df_real.empty:
            raise ValueError(f"No sequences found for fold {fold}")

    if sequence_mode == "full":
        df_real["sequence"] = df_real.apply(extract_full_sequence, axis=1)
    elif sequence_mode == "cds_only":
        df_real["sequence"] = df_real.apply(extract_cds, axis=1)
    elif sequence_mode == "utr5_only":
        df_real["sequence"] = df_real.apply(extract_utr5, axis=1)
    elif sequence_mode == "utr3_only":
        df_real["sequence"] = df_real.apply(extract_utr3, axis=1)
    elif sequence_mode == "utr5_cds":
        df_real["sequence"] = df_real.apply(extract_utr5_cds, axis=1, max_cds_length=max_cds_length)
    elif sequence_mode == "start_codon_window":
        if total_window_length is None:
            raise ValueError("total_window_length must be specified for sequence_mode='start_codon_window'")
        df_real["sequence"] = df_real.apply(extract_start_codon_window, axis=1, total_window_length=total_window_length)
    else:
        raise ValueError(f"Invalid sequence_mode: {sequence_mode}, expected one of ['full', 'cds_only', 'utr5_only', 'utr3_only', 'utr5_cds', 'start_codon_window']")
    
    columns_to_export = ["tx_id", "sequence"] + [col for col in df_real.columns if col.startswith("TE_")]
    df_real = df_real[columns_to_export]
    df_real.to_csv(output_file, sep=",", index=False)


def main():

    parser = argparse.ArgumentParser(description="Preprocess RiboNN dataset for mRNABERT fine-tuning")
    parser.add_argument("--data_path", type=str, default=RIBONN_DATA_PATH, help="Path to the RiboNN dataset in Excel format")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the processed CSV files")
    parser.add_argument("--sequence_mode", type=str, default="full", help="Mode for sequence extraction: one of ['full', 'cds_only', 'utr5_only', 'utr3_only', 'utr5_cds', 'start_codon_window']")
    parser.add_argument("--total_window_length", type=int, default=None, help="Total window length in nucleotides for sequence_mode='start_codon_window' (split evenly around the start codon)")
    parser.add_argument("--max_cds_length", type=int, default=None, help="Maximum CDS length in nucleotides for sequence_mode='utr5_cds' (if specified, only the first max_cds_length nucleotides of the CDS will be included)")
    args = parser.parse_args()

    if args.sequence_mode == "start_codon_window" and args.total_window_length is not None:
        mode_label = f"{args.sequence_mode}_{args.total_window_length}nt"
    elif args.sequence_mode == "utr5_cds" and args.max_cds_length is not None:
        mode_label = f"{args.sequence_mode}_{args.max_cds_length}nt"
    else:
        mode_label = args.sequence_mode

    output_dir = args.output_dir if args.output_dir is not None else f"./processed_data_RiboNN/cv_{mode_label}/"



    print(f"Creating 10-fold cross-validation folder for RiboNN data with mode {args.sequence_mode} and saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for i in range(10):
        test_fold = i
        val_fold = (i + 1) % 10
        fold_output_dir = os.path.join(output_dir, f"val_fold_{val_fold}_test_fold_{test_fold}")
        os.makedirs(fold_output_dir, exist_ok=True)
        export_sequences_for_mrnabert(
            output_file=os.path.join(fold_output_dir, "train.csv"),
            folds=[f for f in range(10) if f not in [val_fold, test_fold]],
            sequence_mode=args.sequence_mode,
            total_window_length=args.total_window_length,
            max_cds_length=args.max_cds_length
        )
        export_sequences_for_mrnabert(
            output_file=os.path.join(fold_output_dir, "dev.csv"),
            folds=[val_fold],
            sequence_mode=args.sequence_mode,
            total_window_length=args.total_window_length,
            max_cds_length=args.max_cds_length
        )
        export_sequences_for_mrnabert(
            output_file=os.path.join(fold_output_dir, "test.csv"),
            folds=[test_fold],
            sequence_mode=args.sequence_mode,
            total_window_length=args.total_window_length,
            max_cds_length=args.max_cds_length
        )

if __name__ == "__main__":
    main()