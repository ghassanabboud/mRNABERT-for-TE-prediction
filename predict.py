import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BertConfig
from peft import PeftModel
from transformers import AutoModelForSequenceClassification


from regression_multilabel import (
    DataCollatorForSupervisedDataset,
    SupervisedDataset,
    calculate_metric_for_regression,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned mRNABERT model.")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to the fine-tuned checkpoint directory.")
    parser.add_argument("--data_path", type=str, default="", help="Directory containing test.csv.")
    parser.add_argument("--output_dir", type=str, default="predictions")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_lora", type=bool, default=False, help="Whether the checkpoint is from a LoRA fine-tuning run.")
    parser.add_argument("--base_model_name", type=str, default="YYLY66/mRNABERT", help="The base model name to use when loading LoRA weights. Only needed if --use_lora is True.")
    parser.add_argument("--num_labels", type=int, default=78, help="Number of labels for the regression task. Only needed if --use_lora is True.")
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.checkpoint_path, "--checkpoint_path must be set to your checkpoint directory."
    assert args.data_path, "--data_path must be set to the directory containing test.csv."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if not args.use_lora:
        
        print("Loading full model from checkpoint...")
        config = BertConfig.from_pretrained(args.checkpoint_path)
        print(f"num_labels inferred from checkpoint: {config.num_labels}")
        print(f"model_max_length inferred from tokenizer: {tokenizer.model_max_length}")
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            args.checkpoint_path,
            config=config,
            trust_remote_code=True,
        )
    else:
        print("you are loading a LoRA checkpoint...")
        print("CAREFUL: IF THE LORA CHECKPOINT WAS NOT SAVED WITH TASK_TYPE=SEQ_CLS, THEN THIS CODE WILL YIELD ARBITRARY RESULTS. THE ADAPTER_MODEL.BIN WILL CONTAIN LORA LAYERS BUT NOT THE SEQUENCE CLASSIFICATION HEAD.")
        print("Loading base model...")

        config = BertConfig.from_pretrained(
            args.base_model_name,
            num_labels=args.num_labels,
            problem_type="regression",
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model_name,
            trust_remote_code=True,
            config=config
        )

        print("Integrating LoRA weights...")
        model = PeftModel.from_pretrained(base_model, args.checkpoint_path)

    model.to(device)
    model.eval()

    test_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(args.data_path, "test.csv"),
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (batch, num_labels)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    all_logits = np.concatenate(all_logits, axis=0)   # (N, num_labels)
    all_labels = np.concatenate(all_labels, axis=0)   # (N, num_labels)

    label_names = test_dataset.label_names
    print(f"Label names: {label_names}")
    print(len(label_names))
    metrics = calculate_metric_for_regression(all_logits, all_labels, label_names=label_names)
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "metrics_test_set.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    n_labels = all_logits.shape[1] if all_logits.ndim == 2 else 1
    names = label_names if len(label_names) == n_labels else [str(i) for i in range(n_labels)]
    pred_cols = {f"predicted_{n}": all_logits[:, i] for i, n in enumerate(names)}
    true_cols = {f"{n}": all_labels[:, i] for i, n in enumerate(names)}

    #CAREFUL: I CAN ONLY DO THIS BECAUSE I EXPLICITELY SAID SHUFFLE=FALSE WHEN INITIALIZING THE DATALOADER. OTHERWISE, THE SEQUENCES WOULD NOT BE IN THE SAME ORDER AS THE PREDICTIONS/LABELS.
    df = pd.DataFrame({"sequence": test_dataset.sequences, **true_cols, **pred_cols})
    df.to_csv(os.path.join(args.output_dir, "predictions_test_set.csv"), index=False)
    df.insert

    print(f"\nSaved predictions to {args.output_dir}/predictions.csv")
    print(f"Saved metrics    to {args.output_dir}/metrics.json")


if __name__ == "__main__":
    main()
