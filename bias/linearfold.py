import subprocess
from collections import Counter
from typing import List, Tuple

import numpy as np


def parse_token_ranges(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """Flatten mRNABERT's mixed UTR/codon tokens into a plain nucleotide sequence
    plus a per-token index into it, so LinearFold (which folds nucleotides, not
    tokens) can be run and its output mapped back to token positions.

    Single-character tokens (UTR) contribute 1 nucleotide; three-character
    tokens (CDS codons) contribute 3 nucleotides.

    Parameters
    ----------
    tokens : List[str]
        Sequence tokens as expected by the mRNABERT tokenizer, e.g. single
        UTR nucleotides ("A", "C", ...) and three-nucleotide CDS codons
        ("AUG", ...). May use "T" or "U".

    Returns
    -------
    nuc_seq : str
        Concatenated nucleotide string, uppercased with T replaced by U for
        RNA folding.
    token_ranges : List[Tuple[int, int]]
        One (start, length) pair per input token, giving its offset and
        length within `nuc_seq`.
    """
    parts = []
    token_ranges: List[Tuple[int, int]] = []
    pos = 0
    for tok in tokens:
        token_ranges.append((pos, len(tok)))
        parts.append(tok)
        pos += len(tok)
    nuc_seq = "".join(parts).upper().replace("T", "U")
    return nuc_seq, token_ranges


def dotbracket_to_token_pairs(
    dot_bracket: str,
    token_ranges: List[Tuple[int, int]],
) -> np.ndarray:
    """Transform a LinearFold structure in dot bracket format to a set of interaction counts 
    between tokens, for use as an attention bias.

    Mapping rule: nucleotide token pairs and nucleotide-codon pairs either have an interaction count of 1 or 0.
    codon-codon pairs can have an interaction count of 1, 2, or 3 through multiple contributing nucleotide pairs.
    Self-pairs (both nucleotides inside the same codon) are discarded.

    Parameters
    ----------
    dot_bracket : str
        MFE secondary structure in dot-bracket notation, one character per
        nucleotide in `nuc_seq` (as returned by `run_linearfold`).
    token_ranges : List[Tuple[int, int]]
        Per-token (start, length) offsets into the nucleotide sequence, as
        returned by `parse_token_ranges`.

    Returns
    -------
    np.ndarray
        int32 array of shape (K, 3): columns are [t_i, t_j, count], where
        count in {1, 2, 3}, and indices are 0-based excluding
        CLS/SEP. The collator sets bias[t_i+1, t_j+1] = count (symmetric).
    """
    N = len(dot_bracket)

    nuc_to_tok = np.empty(N, dtype=np.int32)
    for t, (s, k) in enumerate(token_ranges):
        nuc_to_tok[s : s + k] = t

    stack: List[int] = []
    counts: Counter = Counter()
    for i, c in enumerate(dot_bracket):
        if c == "(":
            stack.append(i)
        elif c == ")":
            j = stack.pop()
            ti, tj = int(nuc_to_tok[j]), int(nuc_to_tok[i])
            if ti != tj:
                counts[(min(ti, tj), max(ti, tj))] += 1

    if not counts:
        return np.empty((0, 3), dtype=np.int32)
    rows = [(ti, tj, cnt) for (ti, tj), cnt in sorted(counts.items())]
    return np.array(rows, dtype=np.int32)


def run_linearfold(nuc_seq: str, executable: str) -> str:
    """Run the external LinearFold binary on a nucleotide sequence and parse
    its minimum free energy (MFE) secondary structure prediction in dot-bracket notation from the output. The output is expected to be two lines:

    LinearFold output format:
        SEQUENCE
        .(((...)))  (-1.23)

    Parameters
    ----------
    nuc_seq : str
        Nucleotide sequence to fold (as produced by `parse_token_ranges`).
    executable : str
        Path to the LinearFold executable; the sequence is piped to it via
        stdin.

    Returns
    -------
    str
        MFE dot-bracket structure string, same length as `nuc_seq`.
    """
    result = subprocess.run(
        [executable],
        input=nuc_seq,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Unexpected LinearFold output: {result.stdout[:200]}")
    dot_bracket = lines[1].strip().split()[0]
    if len(dot_bracket) != len(nuc_seq):
        raise ValueError(
            f"Dot-bracket length {len(dot_bracket)} != sequence length {len(nuc_seq)}"
        )
    return dot_bracket


def process_one(
    tx_id: str,
    sequence_field: str,
    max_tokens: int,
    executable: str,
) -> Tuple[str, np.ndarray]:
    """Process a single sequence end-to-end: tokenize, fold with LinearFold,
    and extract token-level interaction counts. Top-level worker called for
    each row when pre-computing the LinearFold bias .npz in `generate_linearfold_bias.py`.

    Parameters
    ----------
    tx_id : str
        Transcript identifier, passed through unchanged for use as a key in
        the output .npz.
    sequence_field : str
        Whitespace-separated token string (mixed UTR nucleotides and CDS
        codons), e.g. as stored in a `processed_data_RiboNN` CSV column.
    max_tokens : int
        Maximum number of leading tokens to keep; no truncation if <= 0.
    executable : str
        Path to the LinearFold executable, forwarded to `run_linearfold`.

    Returns
    -------
    tx_id : str
        The input transcript identifier, unchanged.
    token_pairs : np.ndarray
        Token-level base-pair array as returned by `dotbracket_to_token_pairs`.
    """
    tokens = sequence_field.split()
    if max_tokens > 0:
        tokens = tokens[:max_tokens]
    nuc_seq, token_ranges = parse_token_ranges(tokens)
    dot_bracket = run_linearfold(nuc_seq, executable)
    return tx_id, dotbracket_to_token_pairs(dot_bracket, token_ranges)