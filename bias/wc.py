import torch


def build_wc_lookup(tokenizer, utr_only: bool = False) -> torch.Tensor:
    """Build a (vocab_size, vocab_size) Watson-Crick base-pairing score matrix.

    For single-nucleotide tokens the score is the standard WC value (A-T=2, C-G=3, G-T=1).
    For codon tokens the score is the sum over all 3x3 constituent nucleotide pairs,
    unless utr_only=True, in which case codon tokens always get score 0.
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
