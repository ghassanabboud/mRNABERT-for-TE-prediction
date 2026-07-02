"""
Attention vs. LinearFold-bias correlation analysis.

Hypothesis (see experiments/06-CV_LinearFold_bias.md): none of the bio-prior bias
variants beat the no_bias model because the frozen mRNABERT backbone already
encodes secondary-structure information in its self-attention. All bias variants
share the same frozen backbone, so we inspect attention from the no_bias checkpoint.

The backbone (bert_layers.py, loaded via trust_remote_code) is a Mosaic-BERT
implementation with ALiBi + unpadded FlashAttention-Triton kernels, which never
exposes attention probabilities through HF's standard output_attentions. Its
BertUnpadSelfAttention.forward already contains a plain-PyTorch fallback branch
(taken whenever attention_probs_dropout_prob != 0 or Triton is unavailable) that
computes softmax(qk^T/sqrt(d) + alibi_bias) explicitly before discarding it. This
script forces that branch and captures its output via a patched forward method,
without altering the model's weights or arithmetic.

For each test sequence, attention (averaged over heads, symmetrized) is compared
against the pre-computed LinearFold token-pair bias at LinearFold-paired
positions ("positive" pairs) and a random sample of unpaired positions
("negative" pairs), and Pearson/Spearman correlation is computed per layer.
"""

import argparse
import random
import types

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from bias import build_wc_lookup
from finetuning import SupervisedDataCollator, SupervisedDataset, calculate_metric_for_regression
from utils.analysis import load_model

CHECKPOINT_PATH = "outputs/cv_biased_full_1024_frozen_1_layer_no_bias/val_fold_4_test_fold_3"
BASE_MODEL_NAME = "YYLY66/mRNABERT"
TEST_CSV_PATH = "processed_data_RiboNN/cv_full/val_fold_4_test_fold_3/test.csv"
BIAS_NPZ_PATH = "processed_data_RiboNN/all_lf_bias.npz"

NUM_HEADS = 8
NUM_BIO_LAYERS = 1
NUM_LABELS = 78
MODEL_MAX_LENGTH = 1024

VALID_BIAS_MODES = ("no_bias", "utr_only", "full", "linearfold")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Correlate per-layer backbone attention with LinearFold pairwise bias."
    )
    parser.add_argument("--checkpoint_path", type=str, default=CHECKPOINT_PATH,
                         help="Path to the trained checkpoint to load (must match --bias mode).")
    parser.add_argument("--num_heads", type=int, default=NUM_HEADS,
                         help="Attention heads per bio-prior layer (must match the checkpoint's architecture).")
    parser.add_argument("--num_bio_layers", type=int, default=NUM_BIO_LAYERS,
                         help="Number of stacked BioPriorAttention layers (must match the checkpoint's "
                              "architecture, e.g. --num_bio_layers 3 for a 3-layer bio-prior head).")
    parser.add_argument("--bias", type=str, default="no_bias", choices=VALID_BIAS_MODES,
                         help="Bio-prior bias mode the checkpoint was trained with. Determines what "
                              "bio_prior_bias (if any) is fed into the bio-prior head layer(s) during "
                              "this script's forward passes.")
    parser.add_argument("--linearfold_bias_file", type=str, default=BIAS_NPZ_PATH,
                         help="Path to the LinearFold .npz. Used both as the ground-truth pairs for "
                              "the correlation analysis and, when --bias linearfold, as the model's "
                              "own bio_prior_bias input.")
    parser.add_argument("--test_csv_path", type=str, default=TEST_CSV_PATH,
                         help="Path to the test.csv used both for the attention/LinearFold correlation "
                              "analysis and, when --evaluate_test_set is set, for the metrics pass.")
    parser.add_argument("--max_sequences", type=int, default=200,
                         help="Number of test-set transcripts to process (use -1 to run on all).")
    parser.add_argument("--negative_ratio", type=int, default=5,
                         help="Number of sampled unpaired (negative) pairs per positive pair.")
    parser.add_argument("--max_negatives_per_seq", type=int, default=2000,
                         help="Cap on sampled negative pairs per sequence.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for negative-pair sampling.")
    parser.add_argument("--output_pairs_csv", type=str, default="attention_linearfold_pairs.csv",
                         help="Path to write the raw per-(layer, sequence, token pair) records.")
    parser.add_argument("--output_correlation_csv", type=str, default="attention_linearfold_correlation.csv",
                         help="Path to write the per-layer correlation summary.")
    parser.add_argument("--evaluate_test_set", action="store_true",
                         help="Also run batched inference on the full test.csv and report regression "
                              "metrics, to confirm the attention patching does not alter model outputs.")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                         help="Batch size used for the --evaluate_test_set inference pass.")
    return parser.parse_args()


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
    """
    positive = []
    positive_set = set()
    for ti, tj, cnt in bias_pairs:
        ti, tj, cnt = int(ti), int(tj), int(cnt)
        if cnt <= 0 or ti <= 0 or tj <= 0 or ti >= seq_len - 1 or tj >= seq_len - 1:
            continue
        positive.append((ti, tj, cnt))
        positive_set.add((ti, tj))

    n_neg_target = min(max_negatives, negative_ratio * max(len(positive), 1))
    negative = []
    if seq_len > 3:
        attempts = 0
        max_attempts = n_neg_target * 20 + 100
        while len(negative) < n_neg_target and attempts < max_attempts:
            attempts += 1
            i = rng.randrange(1, seq_len - 1)
            j = rng.randrange(1, seq_len - 1)
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in positive_set:
                continue
            positive_set.add((a, b))  # avoid resampling the same negative pair
            negative.append((a, b, 0))

    return positive, negative


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


def main():
    args = parse_args()
    if args.bias == "linearfold" and not args.linearfold_bias_file:
        raise ValueError("--bias linearfold requires --linearfold_bias_file.")
    max_sequences = None if args.max_sequences is not None and args.max_sequences < 0 else args.max_sequences
    rng = random.Random(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer, model = load_model(
        device,
        checkpoint_path=args.checkpoint_path,
        base_model_name=BASE_MODEL_NAME,
        model_max_length=MODEL_MAX_LENGTH,
        num_heads=args.num_heads,
        num_labels=NUM_LABELS,
        num_bio_layers=args.num_bio_layers,
    )
    patch_backbone_attention(model)
    patch_bioprior_attention(model)
    num_layers = len(model.bert.encoder.layer)
    num_bio_layers = len(model.bio_attn_layers)
    layer_labels = [f"backbone_{i}" for i in range(num_layers)] + [f"bioprior_{i}" for i in range(num_bio_layers)]
    print(f"Patched {num_layers} backbone layers and {num_bio_layers} bio-prior head layer(s) "
          "for explicit attention_probs capture")
    if args.bias == "linearfold":
        print("Note: --bias linearfold injects the same LinearFold pairs into the bio-prior head's "
              "attention scores, so the bioprior_* correlation rows below are a sanity check that the "
              "bias propagates correctly, not evidence of learned structure. The backbone_* rows "
              "(never see bio_prior_bias) remain the meaningful cross-model comparison.")

    wc_lookup = None
    if args.bias in ("utr_only", "full"):
        wc_lookup = build_wc_lookup(tokenizer, utr_only=(args.bias == "utr_only"))

    data_collator = SupervisedDataCollator(
        tokenizer=tokenizer,
        bias_mode=args.bias,
        wc_lookup=wc_lookup,
        bias_npz_path=args.linearfold_bias_file,
    )

    if args.evaluate_test_set:
        evaluate_test_set(model, tokenizer, data_collator, device, args.eval_batch_size, args.test_csv_path)

    df = pd.read_csv(args.test_csv_path, usecols=["tx_id", "sequence"])
    bias_lookup = dict(np.load(args.linearfold_bias_file, allow_pickle=False))

    records = []
    num_processed = 0
    num_skipped = 0
    with torch.no_grad():
        for tx_id, sequence in zip(df["tx_id"], df["sequence"]):
            if max_sequences is not None and num_processed >= max_sequences:
                break

            pairs = bias_lookup.get(tx_id)
            if pairs is None:
                num_skipped += 1
                continue

            token_ids = tokenizer(sequence, truncation=True, max_length=MODEL_MAX_LENGTH)["input_ids"]
            instance = {
                "input_ids": torch.tensor(token_ids),
                "labels": [float("nan")] * NUM_LABELS,
                "tx_id": tx_id,
            }
            batch = data_collator([instance])
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bio_prior_bias = batch["bio_prior"].to(device) if "bio_prior" in batch else None
            seq_len = input_ids.shape[1]

            model(input_ids=input_ids, attention_mask=attention_mask, bio_prior_bias=bio_prior_bias)
            layer_attns = get_backbone_attentions(model) + get_bioprior_attentions(model)  # each (num_heads, L, L)

            offset_pairs = pairs.copy()
            offset_pairs[:, 0] += 1  # +1 for CLS token, matching finetuning/collators.py
            offset_pairs[:, 1] += 1
            positive, negative = build_pair_records(
                offset_pairs, seq_len, args.negative_ratio, args.max_negatives_per_seq, rng
            )
            pair_records = positive + negative
            if not pair_records:
                num_processed += 1
                continue

            is_arr = np.array([p[0] for p in pair_records])
            js_arr = np.array([p[1] for p in pair_records])
            counts_arr = np.array([p[2] for p in pair_records])

            for layer_idx, attn in enumerate(layer_attns):
                attn_mean = attn.mean(dim=0)  # (L, L), averaged over heads
                attn_sym = (attn_mean + attn_mean.T) / 2.0
                attn_sym = attn_sym.cpu().numpy()
                scores = attn_sym[is_arr, js_arr]
                for i, j, cnt, score in zip(is_arr, js_arr, counts_arr, scores):
                    records.append((tx_id, layer_idx, int(i), int(j), int(cnt), float(score)))

            num_processed += 1
            if num_processed % 20 == 0:
                print(f"Processed {num_processed} sequences ({len(records)} records so far)")

    print(f"Processed {num_processed} sequences, skipped {num_skipped} without LinearFold bias entries")

    pairs_df = pd.DataFrame(records, columns=["tx_id", "layer", "i", "j", "bias_count", "attn_score"])
    pairs_df.to_csv(args.output_pairs_csv, index=False)
    print(f"Saved {len(pairs_df)} rows to {args.output_pairs_csv}")

    summary_rows = []
    for layer_idx, layer_label in enumerate(layer_labels):
        layer_df = pairs_df[pairs_df["layer"] == layer_idx]
        n_positive = int((layer_df["bias_count"] > 0).sum())
        n_negative = int((layer_df["bias_count"] == 0).sum())
        if len(layer_df) > 1:
            pearson_r, pearson_p = pearsonr(layer_df["bias_count"], layer_df["attn_score"])
            spearman_r, spearman_p = spearmanr(layer_df["bias_count"], layer_df["attn_score"])
        else:
            pearson_r = pearson_p = spearman_r = spearman_p = float("nan")
        summary_rows.append((layer_label, n_positive, n_negative, pearson_r, pearson_p, spearman_r, spearman_p))

    summary_df = pd.DataFrame(
        summary_rows,
        columns=["layer", "n_positive", "n_negative", "pearson_r", "pearson_p", "spearman_r", "spearman_p"],
    )
    summary_df.to_csv(args.output_correlation_csv, index=False)
    print(f"Saved per-layer correlation summary to {args.output_correlation_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
