"""
Run label-less inference with a fine-tuned mRNABERT checkpoint (see train.py):
given a CSV of sequences with no TE columns, predict TE for every cell type
the checkpoint was trained on and save per-sequence predictions.

Example:
    python predict.py \
        --checkpoint_path outputs/cv_full_1024/val_fold_4_test_fold_3 \
        --input_csv processed_data/example_inference/example_inference_short.csv \
        --output_dir predictions/example_inference
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig

from finetuning import SupervisedDataCollator, SupervisedDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run label-less inference with a fine-tuned mRNABERT checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the fine-tuned checkpoint directory.")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV with a 'sequence' column (and optional metadata columns); no label columns.")
    parser.add_argument("--output_dir", type=str, default="predictions")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    print(f"model_max_length inferred from checkpoint tokenizer: {tokenizer.model_max_length}")

    print("Loading full model from checkpoint...")
    config = BertConfig.from_pretrained(args.checkpoint_path)
    print(f"num_labels inferred from checkpoint: {config.num_labels}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint_path,
        config=config,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.input_csv)
    print(f"Input set size: {len(dataset)}")

    data_collator = SupervisedDataCollator(tokenizer=tokenizer, bias_mode="no_bias")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    all_logits = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)  # (N, num_labels)

    # id2label keys are strings on disk (JSON) but transformers may normalize
    # them back to int in memory after from_pretrained; handle both.
    names = [config.id2label.get(i, config.id2label.get(str(i))) for i in range(config.num_labels)]
    pred_cols = {f"predicted_{n}": all_logits[:, i] for i, n in enumerate(names)}

    # Relies on shuffle=False above so rows stay aligned with dataset's order.
    meta_cols = {
        name: [row[i] for row in dataset.metadata]
        for i, name in enumerate(dataset.metadata_names)
    }
    df = pd.DataFrame({**meta_cols, "sequence": dataset.sequences, **pred_cols})
    df["mean_predicted_TE"] = df[list(pred_cols)].mean(axis=1)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved predictions to {output_path}")


if __name__ == "__main__":
    main()