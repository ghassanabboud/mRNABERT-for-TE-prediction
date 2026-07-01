"""Helper functions for insertional analysis of motif insertion effects on predicted TE."""

import os

import torch
from transformers import AutoModel, AutoTokenizer, BertConfig
from typing import Dict, List, Tuple
from bias import mRNABERTWithBioPriorHead
import math

HUMAN_CODON_USAGE = {
"TTT": 17.6,  "TCT": 15.2,  "TAT": 12.2, "TGT": 10.6,
"TTC": 20.3,  "TCC": 17.7,  "TAC": 15.3, "TGC": 12.6,
"TTA":  7.7,  "TCA": 12.2,  "TAA":  1.0, "TGA":  1.6,
"TTG": 12.9,  "TCG":  4.4,  "TAG":  0.8, "TGG": 13.2,
"CTT": 13.2,  "CCT": 17.5,  "CAT": 10.9, "CGT":  4.5,
"CTC": 19.6,  "CCC": 19.8,  "CAC": 15.1, "CGC": 10.4,
"CTA":  7.2,  "CCA": 16.9,  "CAA": 12.3, "CGA":  6.2,
"CTG": 39.6,  "CCG":  6.9,  "CAG": 34.2, "CGG": 11.4,
"ATT": 16.0,  "ACT": 13.1,  "AAT": 17.0, "AGT": 12.1,
"ATC": 20.8,  "ACC": 18.9,  "AAC": 19.1, "AGC": 19.5,
"ATA":  7.5,  "ACA": 15.1,  "AAA": 24.4, "AGA": 12.2,
"ATG": 22.0,  "ACG":  6.1,  "AAG": 31.9, "AGG": 12.0,
"GTT": 11.0,  "GCT": 18.4,  "GAT": 21.8, "GGT": 10.8,
"GTC": 14.5,  "GCC": 27.7,  "GAC": 25.1, "GGC": 22.2,
"GTA":  7.1,  "GCA": 15.8,  "GAA": 29.0, "GGA": 16.5,
"GTG": 28.1,  "GCG":  7.4,  "GAG": 39.6, "GGG": 16.5
} 


AMINO_ACID_TO_CODON = {
    "A": ["GCT", "GCC", "GCA", "GCG"],  # Alanine
    "C": ["TGT", "TGC"],  # Cysteine
    "D": ["GAT", "GAC"],  # Aspartic Acid
    "E": ["GAA", "GAG"],  # GlTtamic Acid
    "F": ["TTT", "TTC"],  # Phenylalanine
    "G": ["GGT", "GGC", "GGA", "GGG"],  # Glycine
    "H": ["CAT", "CAC"],  # Histidine
    "I": ["ATT", "ATC", "ATA"],  # IsoleTcine
    "K": ["AAA", "AAG"],  # Lysine
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],  # LeTcine
    "M": ["ATG"],  # Methionine (Start codon)
    "N": ["AAT", "AAC"],  # Asparagine
    "P": ["CCT", "CCC", "CCA", "CCG"],  # Proline
    "Q": ["CAA", "CAG"],  # GlTtamine
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],  # Arginine
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],  # Serine
    "T": ["ACT", "ACC", "ACA", "ACG"],  # Threonine
    "V": ["GTT", "GTC", "GTA", "GTG"],  # Valine
    "W": ["TGG"],  # Tryptophan
    "Y": ["TAT", "TAC"],  # Tyrosine
    "*": ["TAA", "TAG", "TGA"],  # Stop Codons
}

CODON_TO_AMINO_ACID = {
    codon: amino_acid
    for amino_acid, codons in AMINO_ACID_TO_CODON.items()
    for codon in codons
}

MOST_USED_CODON_PER_AA = {
    amino_acid: max(codons, key=lambda codon: HUMAN_CODON_USAGE[codon])
    for amino_acid, codons in AMINO_ACID_TO_CODON.items()
}

LEAST_USED_CODON_PER_AA = {
    amino_acid: min(codons, key=lambda codon: HUMAN_CODON_USAGE[codon])
    for amino_acid, codons in AMINO_ACID_TO_CODON.items()
}

MAX_USAGE_PER_AA = {
    amino_acid: max(HUMAN_CODON_USAGE[codon] for codon in AMINO_ACID_TO_CODON[amino_acid])
    for amino_acid in AMINO_ACID_TO_CODON.keys()
}


def find_utr5_cds_boundaries(tokens):
    """Return (utr5_len_nt, num_cds_codons) from a space-tokenized 'full' sequence.

    5'UTR tokens are single nucleotides (len 1); CDS tokens are codons (len 3),
    contiguous and starting right after the 5'UTR (see utils/preprocess.py).
    """
    utr5_len_nt = 0
    while utr5_len_nt < len(tokens) and len(tokens[utr5_len_nt]) == 1:
        utr5_len_nt += 1

    num_cds_codons = 0
    i = utr5_len_nt
    while i < len(tokens) and len(tokens[i]) == 3:
        num_cds_codons += 1
        i += 1

    return utr5_len_nt, num_cds_codons


def generate_variants(tx_id, tokens, utr5_len_nt, num_cds_codons, motif, upstream_window, downstream_window):
    """Yield (tx_id, insertion_position, sequence) records for one transcript/motif."""
    yield tx_id, float("nan"), " ".join(tokens)

    for k in range(1, upstream_window + 1):
        idx = utr5_len_nt - k + 1
        variant = tokens[:idx] + list(motif) + tokens[idx:]
        yield tx_id, -k, " ".join(variant)

    max_codon_offset = downstream_window // 3
    for codon_offset in range(0, max_codon_offset + 1):
        idx = utr5_len_nt + codon_offset
        variant = tokens[:idx] + [motif] + tokens[idx:]
        yield tx_id, codon_offset * 3, " ".join(variant)


def load_model(device, checkpoint_path, base_model_name, model_max_length, num_heads, num_labels, num_bio_layers):
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    config = BertConfig.from_pretrained(base_model_name)
    base_model = AutoModel.from_pretrained(base_model_name, config=config, trust_remote_code=True)

    model = mRNABERTWithBioPriorHead(
        base_model=base_model,
        hidden_size=768,
        num_heads=num_heads,
        num_labels=num_labels,
        num_bio_layers=num_bio_layers,
    )
    state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return tokenizer, model

def get_cds(rna_seq: str) -> str:
    """Extract the coding sequence (CDS) from a full RNA sequence in mRNABERT convention."""
    symbols = rna_seq.split(" ")
    symbols_to_keep = [s for s in symbols if len(s) == 3]
    return "".join(symbols_to_keep)

def get_cai(
    rna_seq: str,
    codon_usage_freq: Dict[str, float] = HUMAN_CODON_USAGE,
    max_aa_table: Dict[str, float] = MAX_USAGE_PER_AA,
) -> float:

    codons = rna_seq.split(" ")
    codons = [s for s in codons if len(s) == 3]

    # protein length (number of codons)
    protein_length = len(codons)
    cai = 0.0

    # iterate  RNA sequence in steps of 3 (each codon)
    for codon in codons:

        # corresponding amino acid for the codon
        amino_acid = CODON_TO_AMINO_ACID[codon]

        # codon usage frequency for the current codon
        codon_freq = codon_usage_freq[codon]

        # max codon frequency for the corresponding amino acid
        max_freq = max_aa_table[amino_acid]

        # relative adaptiveness
        w_i = codon_freq / max_freq

        # add  log2 of the relative adaptiveness to the CAI
        cai += math.log(w_i)

    # Return the normalized CAI by exponentiating the average log2 value
    return math.exp(cai / protein_length)

def get_max_usage_sequence(rna_seq: str) -> str:
    """Replace each CDS codon with the most-used synonymous codon, keeping UTRs as nucleotides."""
    tokens = rna_seq.split(" ")
    utr5_len_nt, num_cds_codons = find_utr5_cds_boundaries(tokens)
    cds_end = utr5_len_nt + num_cds_codons

    optimized = list(tokens)
    for i in range(utr5_len_nt, cds_end):
        amino_acid = CODON_TO_AMINO_ACID[tokens[i]]
        optimized[i] = MOST_USED_CODON_PER_AA[amino_acid]

    return " ".join(optimized)


def get_min_usage_sequence(rna_seq: str) -> str:
    """Replace each CDS codon with the least-used synonymous codon, keeping UTRs as nucleotides."""
    tokens = rna_seq.split(" ")
    utr5_len_nt, num_cds_codons = find_utr5_cds_boundaries(tokens)
    cds_end = utr5_len_nt + num_cds_codons

    optimized = list(tokens)
    for i in range(utr5_len_nt, cds_end):
        amino_acid = CODON_TO_AMINO_ACID[tokens[i]]
        optimized[i] = LEAST_USED_CODON_PER_AA[amino_acid]

    return " ".join(optimized)