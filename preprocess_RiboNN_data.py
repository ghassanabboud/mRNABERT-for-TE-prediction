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

def extract_utr5_cds(row):
    return extract_utr5(row) + " " + extract_cds(row)

def extract_full_sequence(row):
    return extract_utr5(row) + " " + extract_cds(row) + " " + extract_utr3(row)


def export_sequences_for_mrnabert(output_file, folds=None, sequence_mode="complete"):
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
        df_real["sequence"] = df_real.apply(extract_utr5_cds, axis=1)
    else:
        raise ValueError(f"Invalid sequence_mode: {sequence_mode}, expected one of ['full', 'cds_only', 'utr5_only', 'utr3_only', 'utr5_cds']")
    
    columns_to_export = ["tx_id", "sequence"] + [col for col in df_real.columns if col.startswith("TE_")]
    df_real = df_real[columns_to_export]
    df_real.to_csv(output_file, sep=",", index=False)


def main():

    parser = argparse.ArgumentParser(description="Preprocess RiboNN dataset for mRNABERT fine-tuning")
    parser.add_argument("--data_path", type=str, default=RIBONN_DATA_PATH, help="Path to the RiboNN dataset in Excel format")
    parser.add_argument("--output_dir", type=str, default="./processed_data_RiboNN/", help="Directory to save the processed CSV files")
    parser.add_argument("--sequence_mode", type=str, default="full", help="Mode for sequence extraction: one of ['full', 'cds_only', 'utr5_only', 'utr3_only', 'utr5_cds']")
    parser.add_argument("--val_fold", type=int, default=8, help="Fold(s) to use for validation")
    parser.add_argument("--test_fold", type=int, default=9, help="Fold(s) to use for testing") 
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir , f"{args.sequence_mode}_val_fold_{args.val_fold}_test_fold_{args.test_fold}/")
    print(f"Processing RiboNN data with mode {args.sequence_mode}, val_fold {args.val_fold}, test_fold {args.test_fold} and saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    export_sequences_for_mrnabert(
        output_file=os.path.join(output_dir, "train.csv"),
        folds=[f for f in range(10) if f not in [args.val_fold, args.test_fold]],
        sequence_mode=args.sequence_mode
    )
    export_sequences_for_mrnabert(
        output_file=os.path.join(output_dir, "dev.csv"),
        folds=[args.val_fold],
        sequence_mode=args.sequence_mode
    )
    export_sequences_for_mrnabert(
        output_file=os.path.join(output_dir, "test.csv"),
        folds=[args.test_fold],
        sequence_mode=args.sequence_mode
    )

if __name__ == "__main__":
    main()