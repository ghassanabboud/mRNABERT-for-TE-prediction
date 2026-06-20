"""
Training script for the frozen-BERT + bio-prior attention head.

Watson-Crick bias matrices are built on the fly in the collator from
input_ids using a precomputed (vocab_size, vocab_size) lookup tensor.

All dataset/trainer infrastructure is reused from regression_multilabel.py.
"""

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertConfig,
    EarlyStoppingCallback,
)

from biased_attention_model import mRNABERTWithBioPriorHead
from regression_multilabel import (
    DataArguments,
    DataCollatorForSupervisedDataset,
    MaskedRegressionTrainer,
    ModelArguments,
    SupervisedDataset,
    TrainingArguments,
    calculate_metric_for_regression,
    safe_save_model_for_hf_trainer,
)


# ---------------------------------------------------------------------------
# Extra model arguments specific to this script
# ---------------------------------------------------------------------------

@dataclass
class BiasedModelArguments:
    num_heads: int = field(default=8, metadata={"help": "Attention heads per bio-prior layer."})
    num_bio_layers: int = field(default=1, metadata={"help": "Number of stacked BioPriorAttention layers."})
    freeze_backbone: bool = field(default=True, metadata={"help": "Freeze the BERT backbone; only train the bio-prior layers and classifier."})
    bias: str = field(
        default="full",
        metadata={
            "help": (
                "Watson-Crick bias mode: "
                "'no_bias' — plain attention, no prior injected; "
                "'utr_only' — WC bias only for single-nucleotide (UTR) tokens, codons get 0; "
                "'full' — WC bias for both UTR and CDS (codon) tokens."
            )
        },
    )
    dropout: float = field(default=0.1, metadata={"help": "Dropout for attention and classifier."})
    base_model_name: str = field(default="YYLY66/mRNABERT", metadata={"help": "HF model ID for the backbone."})


# ---------------------------------------------------------------------------
# Watson-Crick lookup matrix (built once from tokenizer vocab)
# ---------------------------------------------------------------------------

def build_wc_lookup(tokenizer, utr_only: bool = False) -> torch.Tensor:
    """
    Returns a (vocab_size, vocab_size) tensor of Watson-Crick scores.

    For single-nucleotide tokens the score is the standard WC value.
    For codon tokens the score is the sum over all 3x3 constituent nucleotide pairs,
    unless utr_only=True, in which case codon tokens always get score 0.
    Special and padding tokens contribute 0.
    """
    vocab = tokenizer.get_vocab()
    V = tokenizer.vocab_size
    nuc_ids = {ch: vocab[ch] for ch in "ATCGN"}

    nuc_wc = torch.zeros(V, V)
    nuc_wc[nuc_ids["A"], nuc_ids["T"]] = 2
    nuc_wc[nuc_ids["T"], nuc_ids["A"]] = 2
    nuc_wc[nuc_ids["C"], nuc_ids["G"]] = 3
    nuc_wc[nuc_ids["G"], nuc_ids["C"]] = 3
    nuc_wc[nuc_ids["T"], nuc_ids["G"]] = 1

    # token_nucs[i] = the (up to 3) nucleotide token IDs that make up token i;
    # padded with 0 (PAD) so that unused slots contribute 0 to the sum.
    # When utr_only=True, codon tokens are left as all-zeros (no contribution).
    token_nucs = torch.zeros(V, 3, dtype=torch.long)
    for tok, tok_id in vocab.items():
        if len(tok) == 1 and tok in nuc_ids:
            token_nucs[tok_id, 0] = tok_id
        elif len(tok) == 3 and not utr_only:
            for k, ch in enumerate(tok):
                token_nucs[tok_id, k] = nuc_ids.get(ch, 0)

    # A[i, k] = sum_a nuc_wc[token_nucs[i,a], k]
    A = nuc_wc[token_nucs, :].sum(dim=1)            # (V, 3, V) → (V, V)
    # wc[i, j] = sum_b A[i, token_nucs[j, b]]
    wc = A[:, token_nucs].sum(dim=-1)               # (V, V, 3) → (V, V)
    return wc


# ---------------------------------------------------------------------------
# Collator: computes Watson-Crick bias on the fly from input_ids
# ---------------------------------------------------------------------------

class BiasedDataCollator(DataCollatorForSupervisedDataset):
    """
    Extends DataCollatorForSupervisedDataset to inject a Watson-Crick bias.

    The (vocab_size, vocab_size) lookup is indexed with the padded input_ids
    to produce a (B, 1, L, L) bias tensor with no manual padding required.
    """

    def __init__(self, tokenizer, wc_lookup: Optional[torch.Tensor]):
        super().__init__(tokenizer=tokenizer)
        self.wc_lookup = wc_lookup

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(instances)
        if self.wc_lookup is not None:
            ids = batch["input_ids"]                                      # (B, L)
            # ids.unsqueeze(2): (B, L, 1) — query indices
            # ids.unsqueeze(1): (B, 1, L) — key indices
            # result[b, i, j] = wc_lookup[ids[b,i], ids[b,j]]
            bias = self.wc_lookup[ids.unsqueeze(2), ids.unsqueeze(1)]    # (B, L, L)
            batch["bio_prior"] = bias.unsqueeze(1)                        # (B, 1, L, L)
        return batch


# ---------------------------------------------------------------------------
# Trainer extension: extracts bio_prior from inputs before model call
# ---------------------------------------------------------------------------

class BiasedMaskedRegressionTrainer(MaskedRegressionTrainer):
    """
    Subclass of MaskedRegressionTrainer that pops bio_prior from inputs
    and passes it explicitly to the model as bio_prior_bias.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        bio_prior = inputs.pop("bio_prior", None)

        outputs = model(**inputs, bio_prior_bias=bio_prior)
        logits = outputs.logits

        mask = ~torch.isnan(labels)
        loss = torch.nn.functional.mse_loss(logits[mask], labels[mask], reduction="mean")

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train():
    parser = transformers.HfArgumentParser(
        (BiasedModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Per-run output dir isolation (wandb sweep compatible)
    run_id = os.environ.get("WANDB_RUN_ID")
    if run_id:
        run_name = f"run_{run_id}"
        training_args = dataclasses.replace(
            training_args,
            output_dir=os.path.join(training_args.output_dir, run_name),
            run_name=run_name,
        )
    print(f"Output dir: {training_args.output_dir}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.base_model_name,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    # --- Watson-Crick lookup (built once from vocab) ---
    if model_args.bias not in ("no_bias", "utr_only", "full"):
        raise ValueError(f"--bias must be one of 'no_bias', 'utr_only', 'full'; got '{model_args.bias}'")
    if model_args.bias == "no_bias":
        wc_lookup = None
    else:
        wc_lookup = build_wc_lookup(tokenizer, utr_only=(model_args.bias == "utr_only"))

    # --- Datasets ---
    make_ds = lambda split: SupervisedDataset(
        data_path=os.path.join(data_args.data_path, f"{split}.csv"),
        tokenizer=tokenizer,
    )
    train_dataset = make_ds("train")
    val_dataset = make_ds("dev")
    test_dataset = make_ds("test")

    print(
        f"Dataset sizes: train={len(train_dataset)}  "
        f"val={len(val_dataset)}  test={len(test_dataset)}"
    )
    print(
        f"num_labels={train_dataset.num_labels}  "
        f"label_names={train_dataset.label_names}"
    )

    # --- Load frozen backbone ---
    # The model's config_class is the standard transformers.BertConfig (not the
    # custom one from the repo), so we must pass it explicitly. trust_remote_code
    # is still needed for the model *code* (modeling_bert.py from the repo).
    config = BertConfig.from_pretrained(model_args.base_model_name)
    base_model = AutoModel.from_pretrained(
        model_args.base_model_name,
        config=config,
        trust_remote_code=True,
    )

    # --- Wrap with trainable bio-prior attention + classifier ---
    model = mRNABERTWithBioPriorHead(
        base_model=base_model,
        hidden_size=768,
        num_heads=model_args.num_heads,
        num_labels=train_dataset.num_labels,
        dropout=model_args.dropout,
        num_bio_layers=model_args.num_bio_layers,
    )
    if model_args.freeze_backbone:
        model.freeze_bert()

    counts = model.count_parameters()
    print(
        f"Parameters — trainable: {counts['trainable']:,}  |  "
        f"frozen: {counts['frozen']:,}  |  total: {counts['total']:,}"
    )

    data_collator = BiasedDataCollator(tokenizer=tokenizer, wc_lookup=wc_lookup)

    label_names = train_dataset.label_names

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        return calculate_metric_for_regression(logits, labels, label_names=label_names)

    # --- Trainer ---
    trainer = BiasedMaskedRegressionTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience,
                early_stopping_threshold=training_args.early_stopping_threshold,
            )
        ],
    )

    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if training_args.eval_and_save_results:
        results_path = os.path.join(
            training_args.output_dir, "results", training_args.run_name
        )
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "test_results.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    start = time.perf_counter()
    train()
    end = time.perf_counter()
    print(f"Training completed in {end - start:.2f} seconds")
