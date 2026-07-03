"""
Fine-tune mRNABERT on RiboNN translation efficiency (multi-label regression).

Classes and utilities live in finetuning/; this script is the entry point.

Example:
    python train.py \
        --model_name_or_path YYLY66/mRNABERT \
        --data_path processed_data_RiboNN/utr5_cds_val_fold_8_test_fold_9 \
        --output_dir outputs/finetune_utr5_cds \
        --num_train_epochs 20
"""

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import BertConfig, EarlyStoppingCallback

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
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="YYLY66/mRNABERT")
    freeze_base: bool = field(default=False, metadata={"help": "Freeze backbone; train only the classifier head"})


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
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

    config = BertConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=train_dataset.num_labels,
        problem_type="regression",
        id2label={i: name for i, name in enumerate(train_dataset.label_names)},
        label2id={name: i for i, name in enumerate(train_dataset.label_names)},
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        config=config,
    )

    if model_args.freeze_base:
        for param in model.base_model.parameters():
            param.requires_grad = False

    data_collator = SupervisedDataCollator(tokenizer=tokenizer, bias_mode="no_bias")
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

    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "test_results.json"), "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    start = time.perf_counter()
    train()
    print(f"Training completed in {time.perf_counter() - start:.2f} seconds")
