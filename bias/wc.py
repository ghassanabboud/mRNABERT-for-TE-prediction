import torch


def build_wc_lookup(tokenizer, utr_only: bool = False) -> torch.Tensor:
    """Precompute a per-vocabulary-pair Watson-Crick base-pairing score table,
    so the collator can turn any pair of token ids into a bias value by a
    single lookup instead of re-deriving it from the tokens' nucleotides each
    batch.

    For single-nucleotide tokens (UTR) the score is the standard WC pairing
    value (A-T=2, C-G=3, G-T=1; 0 for non-pairing combinations). For codon
    tokens (CDS) the score is the sum of WC values over all 3x3 nucleotide
    pairs between the two codons, unless utr_only=True, in which case codon
    tokens always get a score of 0 (only UTR-UTR pairs are scored).

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        mRNABERT tokenizer whose vocabulary includes single-character
        nucleotide tokens ("A", "T", "C", "G", "N") and three-character
        codon tokens.
    utr_only : bool, optional
        If True, zero out all codon-involving pairs so only single-nucleotide
        UTR tokens get a nonzero WC score. Default False.

    Returns
    -------
    torch.Tensor
        Float tensor of shape (vocab_size, vocab_size) where entry [i, j] is
        the WC pairing score between token id i and token id j.
    """
    vocab = tokenizer.get_vocab()
    V = len(tokenizer)
    nuc_ids = {ch: vocab[ch] for ch in "ATCGN"}

    nuc_wc = torch.zeros(V, V)
    nuc_wc[nuc_ids["A"], nuc_ids["T"]] = 2
    nuc_wc[nuc_ids["T"], nuc_ids["A"]] = 2
    nuc_wc[nuc_ids["C"], nuc_ids["G"]] = 3
    nuc_wc[nuc_ids["G"], nuc_ids["C"]] = 3
    nuc_wc[nuc_ids["T"], nuc_ids["G"]] = 1
    nuc_wc[nuc_ids["G"], nuc_ids["T"]] = 1

    # token_nucs[i] = the (up to 3) nucleotide token IDs composing token i;
    # padded with 0 (PAD id) so unused slots contribute 0.
    # When utr_only=True, codon tokens stay all-zeros (no WC contribution).
    token_nucs = torch.zeros(V, 3, dtype=torch.long)
    for tok, tok_id in vocab.items():
        if len(tok) == 1 and tok in nuc_ids:
            token_nucs[tok_id, 0] = tok_id
        elif len(tok) == 3 and not utr_only:
            for k, ch in enumerate(tok):
                token_nucs[tok_id, k] = nuc_ids.get(ch, 0)

    # A[i, k] = sum_a nuc_wc[token_nucs[i,a], k]
    A = nuc_wc[token_nucs, :].sum(dim=1)   # (V, 3, V) → (V, V)
    # wc[i, j] = sum_b A[i, token_nucs[j, b]]
    wc = A[:, token_nucs].sum(dim=-1)      # (V, V, 3) → (V, V)
    return wc
