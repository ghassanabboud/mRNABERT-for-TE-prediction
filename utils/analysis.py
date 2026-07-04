"""Helper functions for insertional analysis of motif insertion effects on predicted TE."""

import types

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import Dict, List, Tuple
from bias import mRNABERTWithBioPriorHead
from finetuning import SupervisedDataset, calculate_metric_for_regression
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
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
    contiguous and starting right after the 5'UTR (see utils/preprocessing.py).
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


def generate_multi_aug_variants(tx_id, tokens, utr5_len_nt, rng, min_offset, max_offset, num_augs_list, motif="ATG"):
    """Insert a random scattering of upstream AUGs into one transcript's 5'UTR, at
    a range of counts.

    For each value in `num_augs_list`, pick that many distinct nucleotide
    positions in the 5'UTR window between `min_offset` and `max_offset` nt
    upstream of the start codon, and splice one copy of `motif` in at each
    position. Also yields the untouched sequence as the num_augs=0 baseline.

    Parameters
    ----------
    tx_id : str
        Transcript identifier, passed through unchanged for use as a key in
        the output records.
    tokens : List[str]
        Space-split tokens of the full sequence (5'UTR nucleotides followed
        by CDS codons), as produced by `find_utr5_cds_boundaries`'s caller.
    utr5_len_nt : int
        Number of nucleotide tokens in the 5'UTR, i.e. the index of the first
        CDS codon token.
    rng : random.Random
        Random source used to sample insertion positions; pass a seeded
        instance for reproducible variants across a run.
    min_offset : int
        Smallest distance upstream of the start codon (in nt) an inserted AUG
        may land at, e.g. 50 keeps insertions at least 50 nt away.
    max_offset : int
        Largest distance upstream of the start codon (in nt) an inserted AUG
        may land at, e.g. 300 keeps insertions within 300 nt of it.
    num_augs_list : List[int]
        Counts of AUGs to insert, e.g. [1, 2, 3, 5, 10]; one variant is
        produced per count.
    motif : str
        Nucleotide motif inserted at each chosen position (default "ATG").

    Returns
    -------
    tx_id : str
        The input transcript identifier, unchanged.
    insertion_positions : List[int]
        Sorted positions of the inserted AUGs, in nt relative to the start
        codon (negative = upstream). Empty for the num_augs=0 baseline.
    num_augs : int
        Number of AUGs inserted in this variant.
    sequence : str
        Space-joined token sequence of the variant, ready for tokenization.
    """
    yield tx_id, [], 0, " ".join(tokens)

    low = utr5_len_nt - max_offset
    high = utr5_len_nt - min_offset
    valid_indices = list(range(low, high + 1))

    for num_augs in num_augs_list:
        positions = rng.sample(valid_indices, num_augs)
        variant = list(tokens)
        for idx in sorted(positions, reverse=True):
            variant[idx:idx] = list(motif)
        insertion_positions = sorted(idx - utr5_len_nt for idx in positions)
        yield tx_id, insertion_positions, num_augs, " ".join(variant)


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


def patch_backbone_attention(model):
    """Force the plain-PyTorch attention branch in every backbone layer and capture
    its post-softmax attention_probs as `last_attention_probs` on each self-attn module.

    This does not change the model's weights or the computation it performs at eval
    time (attention_probs_dropout_prob=0.0 makes the dropout below a no-op) -- it only
    intercepts a tensor the original code already computes and then discards.
    """
    layers = model.bert.encoder.layer
    orig_forward = type(layers[0].attention.self).forward
    g = orig_forward.__func__.__globals__ if hasattr(orig_forward, "__func__") else orig_forward.__globals__
    g["flash_attn_qkvpacked_func"] = None  # disable Triton fused path for every layer (shared module global)
    pad_input = g["pad_input"]
    rearrange = g["rearrange"]
    unpad_input_only = g["unpad_input_only"]

    def make_patched_forward():
        def patched_forward(self, hidden_states, cu_seqlens, max_seqlen_in_batch, indices, attn_mask, bias):
            qkv = self.Wqkv(hidden_states)
            qkv = pad_input(qkv, indices, cu_seqlens.shape[0] - 1, max_seqlen_in_batch)
            qkv = rearrange(qkv, "b s (t h d) -> b s t h d", t=3, h=self.num_attention_heads)
            q = qkv[:, :, 0, :, :].permute(0, 2, 1, 3)
            k = qkv[:, :, 1, :, :].permute(0, 2, 3, 1)
            v = qkv[:, :, 2, :, :].permute(0, 2, 1, 3)
            attention_scores = torch.matmul(q, k) / (self.attention_head_size ** 0.5)
            attention_scores = attention_scores + bias
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            self.last_attention_probs = attention_probs.detach()
            attention_probs = self.dropout(attention_probs)
            attention = torch.matmul(attention_probs, v).permute(0, 2, 1, 3)
            attention = unpad_input_only(attention, torch.squeeze(attn_mask) == 1)
            return rearrange(attention, "nnz h d -> nnz (h d)")
        return patched_forward

    for layer in layers:
        self_attn = layer.attention.self
        self_attn.forward = types.MethodType(make_patched_forward(), self_attn)


def get_backbone_attentions(model):
    """Return a list of (num_heads, L, L) attention tensors, one per backbone layer."""
    return [layer.attention.self.last_attention_probs[0] for layer in model.bert.encoder.layer]


def patch_bioprior_attention(model):
    """Capture attention weights from the trainable BioPriorAttention head layer(s).

    BioPriorAttention.forward (bias/model.py) computes `attn = dropout(scores.softmax(-1))`
    but only returns the pooled context, discarding `attn`. This mirrors
    patch_backbone_attention: same forward logic, plus stashing `attn` before it's dropped.
    Unlike the backbone, these layers operate directly on (B, L, H) with no
    unpad/pad step, so indices line up 1:1 with tokenized positions already.
    """
    def make_patched_forward():
        def patched_forward(self, hidden_states, extended_attention_mask, bio_prior_bias=None):
            B, L, H = hidden_states.shape
            nh, hd = self.num_heads, self.head_dim

            def split_heads(x):
                return x.view(B, L, nh, hd).transpose(1, 2)

            q = split_heads(self.q_proj(hidden_states))
            k = split_heads(self.k_proj(hidden_states))
            v = split_heads(self.v_proj(hidden_states))

            scores = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
            if bio_prior_bias is not None:
                scores = scores + bio_prior_bias
            scores = scores + extended_attention_mask
            attn = scores.softmax(dim=-1)
            self.last_attention_probs = attn.detach()
            attn = self.dropout(attn)

            context = (attn @ v).transpose(1, 2).contiguous().view(B, L, H)
            return self.out_proj(context)
        return patched_forward

    for bio_attn in model.bio_attn_layers:
        bio_attn.forward = types.MethodType(make_patched_forward(), bio_attn)


def get_bioprior_attentions(model):
    """Return a list of (num_heads, L, L) attention tensors, one per bio-prior layer."""
    return [bio_attn.last_attention_probs[0] for bio_attn in model.bio_attn_layers]


def build_pair_records(bias_pairs, seq_len, negative_ratio, max_negatives, rng):
    """Return (positive, negative) lists of (i, j, bias_count) tuples.

    bias_pairs indices must already be offset (+1) to align with tokenized positions
    (position 0 = CLS). Positions 0 and seq_len - 1 (CLS/SEP) are excluded from both sets.

    Negatives are distance-matched: for each positive pair (i, j) at sequence distance
    d = |i - j|, negatives are sampled at that same distance d (one endpoint random,
    the other placed d away, in whichever direction stays in bounds). This controls for
    the fact that LinearFold contacts skew local, so uniformly-random negatives would
    mostly be far apart -- letting any generic distance-attention correlation (e.g. from
    an ALiBi-style positional bias) masquerade as a base-pairing signal.
    """
    positive = []
    positive_set = set()
    for ti, tj, cnt in bias_pairs:
        ti, tj, cnt = int(ti), int(tj), int(cnt)
        if cnt <= 0 or ti <= 0 or tj <= 0 or ti >= seq_len - 1 or tj >= seq_len - 1:
            continue
        positive.append((ti, tj, cnt))
        positive_set.add((ti, tj))

    negative = []
    if seq_len > 3:
        for ti, tj, _ in positive:
            if len(negative) >= max_negatives:
                break
            d = abs(ti - tj)
            n_found = 0
            attempts = 0
            max_attempts = negative_ratio * 20 + 20
            while n_found < negative_ratio and attempts < max_attempts and len(negative) < max_negatives:
                attempts += 1
                i = rng.randrange(1, seq_len - 1)
                j = i + d if rng.random() < 0.5 else i - d
                if j < 1 or j >= seq_len - 1 or j == i:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in positive_set:
                    continue
                positive_set.add((a, b))  # avoid resampling the same negative pair
                negative.append((a, b, 0))
                n_found += 1

    return positive, negative


def select_top_correlated_sequences(pairs_df, layer_idx, num_examples, min_positive=5):
    """Rank transcripts by per-sequence Spearman correlation between attention
    score and LinearFold bias count, for one layer of an existing pairs CSV
    (as produced by study_attention_ss_correlation.py).

    Only transcripts with at least `min_positive` positive (bias_count > 0)
    rows are considered, since a per-sequence correlation estimated from very
    few contact pairs is unreliable.

    Returns a list of (tx_id, spearman_r) tuples, longest `num_examples`,
    sorted by descending correlation.
    """
    layer_df = pairs_df[pairs_df["layer"] == layer_idx]

    results = []
    for tx_id, group in layer_df.groupby("tx_id"):
        n_positive = int((group["bias_count"] > 0).sum())
        if n_positive < min_positive or len(group) < 2:
            continue
        rho, _ = spearmanr(group["bias_count"], group["attn_score"])
        if not np.isnan(rho):
            results.append((tx_id, float(rho)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:num_examples]


def extract_dense_matrices(model, tokenizer, data_collator, cfg, sequence, tx_id, bias_pairs, layer_idx, device):
    """Run one forward pass and return the full dense (L, L) attention matrix
    for `layer_idx` alongside the full dense (L, L) LinearFold contact-count
    matrix, plus the tokenized sequence for axis labeling.

    Mirrors the per-sequence loop in study_attention_ss_correlation.py, but
    keeps the whole matrix instead of indexing into a sampled subset of pairs
    -- needed to render a heatmap rather than compute a correlation.

    Also returns `start_codon_idx`, the position of the first CDS token (the
    start codon) within `tokens`, accounting for the leading CLS token, so
    callers can crop the matrices to a window around it.
    """
    utr5_len_nt, _ = find_utr5_cds_boundaries(sequence.split(" "))
    start_codon_idx = utr5_len_nt + 1  # +1 for CLS token

    token_ids = tokenizer(sequence, truncation=True, max_length=tokenizer.model_max_length)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    instance = {
        "input_ids": torch.tensor(token_ids),
        "labels": [float("nan")] * cfg["num_labels"],
        "tx_id": tx_id,
    }
    batch = data_collator([instance])
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    bio_prior_bias = batch["bio_prior"].to(device) if "bio_prior" in batch else None
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask, bio_prior_bias=bio_prior_bias)
    layer_attns = get_backbone_attentions(model) + get_bioprior_attentions(model)

    attn_mean = layer_attns[layer_idx].mean(dim=0)  # (L, L), averaged over heads
    attn_matrix = ((attn_mean + attn_mean.T) / 2.0).cpu().numpy()

    contact_matrix = np.zeros((seq_len, seq_len), dtype=np.int32)
    for ti, tj, cnt in bias_pairs:
        ti, tj, cnt = int(ti) + 1, int(tj) + 1, int(cnt)  # +1 for CLS token
        if cnt <= 0 or ti >= seq_len - 1 or tj >= seq_len - 1:
            continue
        contact_matrix[ti, tj] = cnt
        contact_matrix[tj, ti] = cnt

    return attn_matrix, contact_matrix, tokens, start_codon_idx


def evaluate_test_set(model, tokenizer, data_collator, device, batch_size, test_csv_path):
    """Batched inference over the full test.csv, reporting the same regression metrics
    used during training (finetuning/metrics.py). The patched attention forwards are
    mathematically identical to the originals, so this should reproduce the checkpoint's
    original test metrics -- any discrepancy would indicate the patching altered outputs.
    """
    test_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=test_csv_path)
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bio_prior_bias = batch["bio_prior"].to(device) if "bio_prior" in batch else None
            output = model(input_ids=input_ids, attention_mask=attention_mask, bio_prior_bias=bio_prior_bias)
            all_logits.append(output.logits.cpu().numpy())
            all_labels.append(batch["labels"].numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    metrics = calculate_metric_for_regression(logits, labels, label_names=test_dataset.label_names)

    print(f"Test-set evaluation ({len(test_dataset)} sequences) with patched attention:")
    for key in ("mse_loss_mean", "pearson_corr_mean", "spearman_corr_mean", "r2_score_mean",
                "pearson_mean_TE", "r2_mean_TE"):
        print(f"  {key}: {metrics[key]:.4f}")
    return metrics