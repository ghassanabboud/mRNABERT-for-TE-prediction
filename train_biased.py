"""
Fine-tune mRNABERT with a bio-prior attention head on top of a frozen backbone.

The bio-prior attention injects a structural bias (Watson-Crick base pairing or
LinearFold secondary structure) into the attention scores before softmax.

Classes and utilities live in finetuning/ and bias/; this script is the entry point.

Example:
    python train_biased.py \
        --bias full \
        --data_path processed_data_RiboNN/utr5_cds_val_fold_8_test_fold_9 \
        --output_dir outputs/biased_full \
        --num_train_epochs 20

    python train_biased.py \
        --bias linearfold \
        --linearfold_bias_file processed_data_RiboNN/all_lf_bias.npz \
        --data_path processed_data_RiboNN/utr5_cds_val_fold_8_test_fold_9 \
        --output_dir outputs/biased_linearfold
"""

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import AutoModel, BertConfig, EarlyStoppingCallback

from bias import build_wc_lookup, mRNABERTWithBioPriorHead
from finetuning import (
    DataArguments,
    MaskedRegressionTrainer,
    SupervisedDataCollator,
    SupervisedDataset,
    TrainingArguments,
    calculate_metric_for_regression,
    safe_save_model_for_hf_trainer,
)


@dataclass
class BiasedModelArguments:
    base_model_name: str = field(default="YYLY66/mRNABERT", metadata={"help": "HF model ID for the backbone."})
    num_heads: int = field(default=8, metadata={"help": "Attention heads per bio-prior layer."})
    num_bio_layers: int = field(default=1, metadata={"help": "Number of stacked BioPriorAttention layers."})
    freeze_backbone: bool = field(default=True, metadata={"help": "Freeze BERT backbone; train only bio-prior layers and classifier."})
    bias: str = field(
        default="full",
        metadata={
            "help": (
                "Bias mode: "
                "'no_bias' — plain attention; "
                "'utr_only' — Watson-Crick bias for UTR tokens only; "
                "'full' — Watson-Crick bias for UTR + CDS tokens; "
                "'linearfold' — LinearFold secondary-structure bias (requires --linearfold_bias_file)."
            )
        },
    )
    linearfold_bias_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to .npz of LinearFold token-pair scores. Required when --bias linearfold."},
    )
    dropout: float = field(default=0.1, metadata={"help": "Dropout for attention and classifier."})


def train():
    parser = transformers.HfArgumentParser((BiasedModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    valid_bias_modes = ("no_bias", "utr_only", "full", "linearfold")
    if model_args.bias not in valid_bias_modes:
        raise ValueError(f"--bias must be one of {valid_bias_modes}; got '{model_args.bias}'")
    if model_args.bias == "linearfold" and not model_args.linearfold_bias_file:
        raise ValueError("--bias linearfold requires --linearfold_bias_file.")

    run_id = os.environ.get("WANDB_RUN_ID")
    if run_id:
        run_name = f"run_{run_id}"
        training_args = dataclasses.replace(
            training_args,
            output_dir=os.path.join(training_args.output_dir, run_name),
            run_name=run_name,
        )
    print(f"Output dir: {training_args.output_dir}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.base_model_name,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=os.path.join(data_args.data_path, "train.csv"))
    val_dataset   = SupervisedDataset(tokenizer=tokenizer, data_path=os.path.join(data_args.data_path, "dev.csv"))
    test_dataset  = SupervisedDataset(tokenizer=tokenizer, data_path=os.path.join(data_args.data_path, "test.csv"))

    print(f"Dataset sizes: train={len(train_dataset)}  val={len(val_dataset)}  test={len(test_dataset)}")
    print(f"num_labels={train_dataset.num_labels}  label_names={train_dataset.label_names}")

    config = BertConfig.from_pretrained(model_args.base_model_name)
    base_model = AutoModel.from_pretrained(
        model_args.base_model_name,
        config=config,
        trust_remote_code=True,
    )

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

    wc_lookup = None
    if model_args.bias in ("utr_only", "full"):
        wc_lookup = build_wc_lookup(tokenizer, utr_only=(model_args.bias == "utr_only"))

    data_collator = SupervisedDataCollator(
        tokenizer=tokenizer,
        bias_mode=model_args.bias,
        wc_lookup=wc_lookup,
        bias_npz_path=model_args.linearfold_bias_file,
    )

    label_names = train_dataset.label_names

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        return calculate_metric_for_regression(logits, labels, label_names=label_names)

    trainer = MaskedRegressionTrainer(
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

    # Trainer normally strips columns not in the model's forward() signature.
    # For linearfold, tx_id must survive until the collator pops it.
    if model_args.bias == "linearfold":
        object.__setattr__(trainer.args, "remove_unused_columns", False)

    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "test_results.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    start = time.perf_counter()
    train()
    print(f"Training completed in {time.perf_counter() - start:.2f} seconds")
