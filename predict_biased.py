"""
Run label-less inference with a trained bio-prior-head mRNABERT checkpoint
(see train_biased.py): given a CSV of sequences with no TE columns, predict
TE for every cell type the checkpoint was trained on and save per-sequence
predictions. No metrics are computed since there is nothing to compare
against.

mRNABERTWithBioPriorHead is a plain nn.Module, not a HF PreTrainedModel, so
the checkpoint directory has no config.json recording the architecture it
was trained with. num_labels is read back from the checkpoint's classifier
weight shape, but --num_heads, --num_bio_layers, and --bias must be passed
explicitly, matching the values used for that run (check the SLURM script
that launched training, e.g. under jobs/, or the run's wandb config).

Example:
    python predict_biased.py \
        --checkpoint_path outputs/biased_head_wc_utr5_cds_1024_frozen_1_layer_full_bias \
        --input_csv new_constructs.csv \
        --bias full \
        --num_heads 8 \
        --num_bio_layers 1 \
        --output_dir predictions/new_constructs
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertConfig

from bias import build_wc_lookup, mRNABERTWithBioPriorHead
from finetuning import SupervisedDataCollator, SupervisedDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run label-less inference with a trained bio-prior-head mRNABERT checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Directory containing pytorch_model.bin from train_biased.py.")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV with a 'sequence' column (and optional metadata columns); no label columns.")
    parser.add_argument("--output_dir", type=str, default="predictions")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_model_name", type=str, default="YYLY66/mRNABERT", help="HF model ID for the backbone; must match the training run.")
    parser.add_argument("--num_heads", type=int, required=True, help="Attention heads per bio-prior layer, as used at training time.")
    parser.add_argument("--num_bio_layers", type=int, required=True, help="Number of stacked BioPriorAttention layers, as used at training time.")
    parser.add_argument(
        "--bias",
        type=str,
        required=True,
        choices=("no_bias", "utr_only", "full", "linearfold"),
        help="Bias mode the checkpoint was trained with.",
    )
    parser.add_argument("--linearfold_bias_file", type=str, default=None, help="Path to .npz of LinearFold token-pair scores. Required when --bias linearfold.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bias == "linearfold" and not args.linearfold_bias_file:
        raise ValueError("--bias linearfold requires --linearfold_bias_file.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    print(f"model_max_length inferred from checkpoint tokenizer: {tokenizer.model_max_length}")

    dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.input_csv)
    print(f"Input set size: {len(dataset)}")

    state_dict = torch.load(os.path.join(args.checkpoint_path, "pytorch_model.bin"), map_location="cpu")
    num_labels = state_dict["classifier.weight"].shape[0]
    print(f"num_labels inferred from checkpoint classifier: {num_labels}")

    config = BertConfig.from_pretrained(args.base_model_name)
    base_model = AutoModel.from_pretrained(args.base_model_name, config=config, trust_remote_code=True)

    model = mRNABERTWithBioPriorHead(
        base_model=base_model,
        hidden_size=768,
        num_heads=args.num_heads,
        num_labels=num_labels,
        num_bio_layers=args.num_bio_layers,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    wc_lookup = None
    if args.bias in ("utr_only", "full"):
        wc_lookup = build_wc_lookup(tokenizer, utr_only=(args.bias == "utr_only"))

    data_collator = SupervisedDataCollator(
        tokenizer=tokenizer,
        bias_mode=args.bias,
        wc_lookup=wc_lookup,
        bias_npz_path=args.linearfold_bias_file,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    all_logits = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            batch.pop("labels")
            bio_prior = batch.pop("bio_prior", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            if bio_prior is not None:
                batch["bio_prior_bias"] = bio_prior.to(device)

            outputs = model(**batch)
            all_logits.append(outputs.logits.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)  # (N, num_labels)

    pred_cols = {f"predicted_{i}": all_logits[:, i] for i in range(num_labels)}

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
