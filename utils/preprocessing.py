"""
Sequence extraction helpers and CSV export shared by preprocess_one_split.py
(single train/dev/test split) and preprocess_all_cv_splits.py (10-fold CV).
"""

import pandas as pd


def extract_cds(row: pd.Series) -> str:
    """Extract a transcript's coding sequence (CDS) as space-separated
    codons, for mRNABERT's tokenizer to consume as one codon per token.


    Parameters
    ----------
    row : pd.Series
        One RiboNN row with "tx_sequence" (full transcript nucleotide
        string), "utr5_size" (5'UTR length in nt, CDS start offset),
        "cds_size" (CDS length in nt, must be a multiple of 3), and "tx_id".

    Returns
    -------
    str
        CDS codons separated by single spaces, e.g. "AUG CGU UAA".
    """
    seq, utr5_len, cds_len = row["tx_sequence"], row["utr5_size"], row["cds_size"]
    if cds_len % 3 != 0:
        raise ValueError(f"CDS length {cds_len} not divisible by 3 for tx_id {row['tx_id']}")
    codon_list = [seq[i:i + 3] for i in range(utr5_len, utr5_len + cds_len, 3)]
    return " ".join(codon_list)


def extract_utr5(row: pd.Series) -> str:
    """Extract a transcript's 5'UTR as space-separated single nucleotides,
    for mRNABERT's tokenizer to consume as one nucleotide per token

    Parameters
    ----------
    row : pd.Series
        One RiboNN row with "tx_sequence" (full transcript nucleotide
        string) and "utr5_size" (5'UTR length in nt).

    Returns
    -------
    str
        5'UTR nucleotides separated by single spaces, e.g. "A C G".
    """
    return " ".join(row["tx_sequence"][:row["utr5_size"]])


def extract_utr3(row: pd.Series) -> str:
    """Extract a transcript's 3'UTR (everything after the CDS) as
    space-separated single nucleotides, for mRNABERT's tokenizer to consume
    as one nucleotide per token.

    Parameters
    ----------
    row : pd.Series
        One RiboNN row with "tx_sequence" (full transcript nucleotide
        string), "utr5_size" (5'UTR length in nt), and "cds_size" (CDS
        length in nt), used together to locate where the CDS ends.

    Returns
    -------
    str
        3'UTR nucleotides separated by single spaces.
    """
    seq, utr5_len, cds_len = row["tx_sequence"], row["utr5_size"], row["cds_size"]
    return " ".join(seq[utr5_len + cds_len:])


def extract_utr5_cds(row: pd.Series, max_cds_length: int = None) -> str:
    """Extract a transcript's 5'UTR followed by its CDS, tokenized as
    nucleotides then codons.

    Parameters
    ----------
    row : pd.Series
        One RiboNN row with "tx_sequence", "utr5_size", "cds_size", and
        "tx_id".
    max_cds_length : int, optional
        If given, truncate the CDS to at most this many nucleotides
        (counted from its start codon) while keeping the full 5'UTR. Must be a multiple
        of 3. If None, the full CDS is kept. Default None.

    Returns
    -------
    str
        5'UTR nucleotides then CDS codons, all separated by single spaces,
        e.g. "A C G AUG CGU".
    """
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


def extract_full_sequence(row: pd.Series) -> str:
    """Extract a transcript's entire 5'UTR + CDS + 3'UTR, tokenized as
    nucleotides for the UTRs and codons for the CDS.

    Parameters
    ----------
    row : pd.Series
        One RiboNN row with "tx_sequence", "utr5_size", "cds_size", and
        "tx_id".

    Returns
    -------
    str
        5'UTR nucleotides, then CDS codons, then 3'UTR nucleotides, all
        separated by single spaces.
    """
    return extract_utr5(row) + " " + extract_cds(row) + " " + extract_utr3(row)


def extract_start_codon_window(row: pd.Series, total_window_length: int) -> str:
    """Extract a fixed-length window of a transcript centered on its start
    codon, tokenized as nucleotides upstream and codons downstream.

    Parameters
    ----------
    row : pd.Series
        One RiboNN row with "tx_sequence", "utr5_size", "cds_size", and
        "tx_id".
    total_window_length : int
        Total window length in nucleotides, split evenly before and after
        the start codon; half the window must be a multiple of 3. The
        window is clipped at the 5' end of the transcript and the end of
        the CDS, so it can be shorter than `total_window_length` for short
        UTRs or CDSs.

    Returns
    -------
    str
        Upstream nucleotides then downstream codons, separated by single
        spaces.
    """
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



def export_sequences_for_mrnabert(
    output_file: str,
    data_path: str,
    folds: list = None,
    sequence_mode: str = "utr5_cds",
    total_window_length: int = None,
    max_cds_length: int = None,
) -> None:
    """Load the RiboNN Excel file, optionally restrict to a subset of CV
    folds, extract each transcript's sequence in the requested mode, and
    write a CSV ready for `SupervisedDataset` (tx_id, sequence, TE_* label
    columns).

    Parameters
    ----------
    output_file : str
        Path to write the resulting CSV to.
    data_path : str
        Path to the RiboNN Excel file to load.
    folds : list, optional
        If given, keep only rows whose "fold" column is in this list (e.g.
        the training folds for one CV split); raises if none match. If
        None, all rows are kept. Default None.
    sequence_mode : str, optional
        Which extraction function to apply per row: "full", "cds_only",
        "utr5_only", "utr3_only", "utr5_cds", or "start_codon_window".
        Default "utr5_cds".
    total_window_length : int, optional
        Required when `sequence_mode="start_codon_window"`; forwarded to
        `extract_start_codon_window`. Default None.
    max_cds_length : int, optional
        Only used when `sequence_mode="utr5_cds"`; forwarded to
        `extract_utr5_cds` to truncate the CDS. Default None.

    Returns
    -------
    None
    """
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
