import os
import csv
import json
import time
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
import transformers
from transformers import Trainer, EarlyStoppingCallback
from transformers.modeling_outputs import SequenceClassifierOutput


@dataclass
class ModelArguments:
    input_dim: int = field(default=768, metadata={"help": "Embedding dimension (mRNABERT hidden size)."})
    hidden_dims: List[int] = field(
        default_factory=lambda: [512, 256],
        metadata={"help": "Hidden layer sizes, e.g. --hidden_dims 512 256 128"},
    )
    dropout: float = field(default=0.1)


@dataclass
class DataArguments:
    train_embeddings: str = field(metadata={"help": "Path to train embeddings .npz file."})
    train_labels: str = field(metadata={"help": "Path to train labels .csv (sequence col + label cols)."})
    eval_embeddings: str = field(metadata={"help": "Path to eval embeddings .npz file."})
    eval_labels: str = field(metadata={"help": "Path to eval labels .csv."})
    test_embeddings: Optional[str] = field(default=None, metadata={"help": "Path to test embeddings .npz file."})
    test_labels: Optional[str] = field(default=None, metadata={"help": "Path to test labels .csv."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    num_train_epochs: int = field(default=50)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=64)
    per_device_eval_batch_size: int = field(default=256)
    fp16: bool = field(default=False)
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.01)
    lr_scheduler_type: str = field(default="cosine_with_restarts")
    warmup_steps: int = field(default=100)
    logging_steps: int = field(default=50)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="r2_score_mean")
    greater_is_better: bool = field(default=True)
    output_dir: str = field(default="output_mlp")
    dataloader_pin_memory: bool = field(default=False)
    seed: int = field(default=42)
    report_to: Optional[str] = field(default="wandb")
    overwrite_output_dir: bool = field(default=True)
    log_level: str = field(default="info")
    save_model: bool = field(default=True)
    eval_and_save_results: bool = field(default=True)
    early_stopping_patience: int = field(default=10)
    early_stopping_threshold: float = field(default=0.0)


class EmbeddingDataset(Dataset):
    """Dataset backed by pre-computed embeddings (NPZ) and labels (CSV, row-aligned)."""

    def __init__(self, embeddings_path: str, labels_path: str):
        embeddings = np.load(embeddings_path)["embeddings"]
        self.embeddings = torch.from_numpy(embeddings).float()

        with open(labels_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = list(reader)

        # CSV format: sequence (col 0) then label columns — same as the CSVs fed to encode.py
        self.label_names = header[1:]
        self.labels = [
            [float(v) if v != "" else float("nan") for v in row[1:]]
            for row in data
        ]
        self.num_labels = len(self.labels[0]) if self.labels else 0

        assert len(self.embeddings) == len(self.labels), (
            f"Row count mismatch: {len(self.embeddings)} embeddings vs "
            f"{len(self.labels)} label rows in {labels_path}"
        )

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(embeddings=self.embeddings[i], labels=self.labels[i])


def collate_embeddings(instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    embeddings = torch.stack([inst["embeddings"] for inst in instances])
    labels = torch.tensor([inst["labels"] for inst in instances], dtype=torch.float)
    return dict(embeddings=embeddings, labels=labels)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], n_labels: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_labels))
        self.mlp = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor, **kwargs) -> SequenceClassifierOutput:
        return SequenceClassifierOutput(logits=self.mlp(embeddings))


class MaskedRegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        mask = ~torch.isnan(labels)
        loss = torch.nn.functional.mse_loss(logits[mask], labels[mask], reduction="mean")
        return (loss, outputs) if return_outputs else loss


def calculate_metric_for_regression(logits: np.ndarray, labels: np.ndarray, label_names=None):
    """Per-label Pearson, Spearman, R² and macro-averaged MSE."""
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])

    predictions = logits.squeeze()
    labels = labels.squeeze()
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]
        labels = labels[:, np.newaxis]

    all_valid_preds, all_valid_labels = [], []
    per_label = {"pearson": [], "spearman": [], "r2": []}

    for i in range(predictions.shape[1]):
        preds_i, labels_i = predictions[:, i], labels[:, i]
        valid = ~np.isnan(labels_i)
        if valid.sum() < 2:
            print(f"[Metrics]     -> skipped (fewer than 2 valid samples)")
            continue
        preds_i, labels_i = preds_i[valid], labels_i[valid]
        all_valid_preds.append(preds_i)
        all_valid_labels.append(labels_i)
        per_label["pearson"].append(pearsonr(labels_i, preds_i)[0])
        per_label["spearman"].append(spearmanr(labels_i, preds_i)[0])
        per_label["r2"].append(r2_score(labels_i, preds_i))

    return {
        "mse_loss_mean": float(mean_squared_error(
            np.concatenate(all_valid_labels), np.concatenate(all_valid_preds)
        )),
        "pearson_corr_mean": float(np.mean(per_label["pearson"])),
        "spearman_corr_mean": float(np.mean(per_label["spearman"])),
        "r2_score_mean": float(np.mean(per_label["r2"])),
    }


def safe_save_model(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


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

    train_dataset = EmbeddingDataset(data_args.train_embeddings, data_args.train_labels)
    val_dataset = EmbeddingDataset(data_args.eval_embeddings, data_args.eval_labels)
    test_dataset = (
        EmbeddingDataset(data_args.test_embeddings, data_args.test_labels)
        if data_args.test_embeddings and data_args.test_labels
        else None
    )

    print(
        f"Dataset sizes: train={len(train_dataset)}  val={len(val_dataset)}"
        + (f"  test={len(test_dataset)}" if test_dataset else "")
    )
    print(f"num_labels={train_dataset.num_labels}  label_names={train_dataset.label_names}")

    model = MLPRegressor(
        input_dim=model_args.input_dim,
        hidden_dims=model_args.hidden_dims,
        n_labels=train_dataset.num_labels,
        dropout=model_args.dropout,
    )

    label_names = train_dataset.label_names

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        return calculate_metric_for_regression(logits, labels, label_names=label_names)

    trainer = MaskedRegressionTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_embeddings,
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
        safe_save_model(trainer, training_args.output_dir)

    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        os.makedirs(results_path, exist_ok=True)
        eval_set = test_dataset if test_dataset else val_dataset
        results = trainer.evaluate(eval_dataset=eval_set)
        with open(os.path.join(results_path, "test_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}/test_results.json")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    start = time.perf_counter()
    train()
    end = time.perf_counter()
    print(f"Training completed in {end - start:.2f} seconds")
