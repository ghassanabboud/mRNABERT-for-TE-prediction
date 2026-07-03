"""
Run label-less inference with a trained bio-prior-head mRNABERT checkpoint
(see train_biased.py): given a CSV of sequences with no TE columns, predict
TE for every cell type the checkpoint was trained on and save per-sequence
predictions. 

mRNABERTWithBioPriorHead is a plain nn.Module, not a HF PreTrainedModel, so
train_biased.py writes its own bio_prior_config.json next to
pytorch_model.bin recording the architecture (num_heads, num_bio_layers,
bias mode, base model, cell-type names) it was trained with.
mRNABERTWithBioPriorHead.from_checkpoint reads that file to reconstruct the model.

If using a checkpoint trained with --bias linearfold, you must also pass --linearfold_bias_file
It must be generated for the sequences in --input_csv (via generate_linearfold_bias.py)

Example:
    python predict_biased.py \
        --checkpoint_path outputs/biased_head_wc_utr5_cds_1024_frozen_1_layer_full_bias \
        --input_csv processed_data/example_inference/example_inference_short.csv \
        --output_dir predictions/example_inference_wc_bias

    # linearfold checkpoints require a bias file generated for the input CSV:
    python predict_biased.py \
        --checkpoint_path outputs/biased_linearfold \
        --input_csv processed_data/example_inference/example_inference_short.csv \
        --linearfold_bias_file processed_data/example_inference/example_inference_short.npz \
        --output_dir predictions/example_inference_lf_bias
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from bias import build_wc_lookup, mRNABERTWithBioPriorHead
from finetuning import SupervisedDataCollator, SupervisedDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run label-less inference with a trained bio-prior-head mRNABERT checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Directory containing pytorch_model.bin and bio_prior_config.json from train_biased.py.")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV with a 'sequence' column (and optional metadata columns); no label columns.")
    parser.add_argument("--output_dir", type=str, default="predictions")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--linearfold_bias_file",
        type=str,
        default=None,
        help=(
            "Path to .npz of LinearFold token-pair scores for the sequences in --input_csv "
            "(generate with generate_linearfold_bias.py). Required if the checkpoint was "
            "trained with --bias linearfold. Must cover the inference sequences, not the "
            "training set's bias file."
        ),
    )
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

    dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.input_csv)
    print(f"Input set size: {len(dataset)}")

    model, cfg = mRNABERTWithBioPriorHead.from_checkpoint(args.checkpoint_path, device=device)
    bias_mode = cfg["bias"]
    num_labels = cfg["num_labels"]
    id2label = cfg["id2label"]
    print(f"Loaded checkpoint: bias={bias_mode}  num_heads={cfg['num_heads']}  num_bio_layers={cfg['num_bio_layers']}  num_labels={num_labels}")

    if bias_mode == "linearfold" and not args.linearfold_bias_file:
        raise ValueError(
            "This checkpoint was trained with --bias linearfold; pass --linearfold_bias_file "
            "generated for the sequences in --input_csv (see generate_linearfold_bias.py)."
        )

    wc_lookup = None
    if bias_mode in ("utr_only", "full"):
        wc_lookup = build_wc_lookup(tokenizer, utr_only=(bias_mode == "utr_only"))

    data_collator = SupervisedDataCollator(
        tokenizer=tokenizer,
        bias_mode=bias_mode,
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

    names = [id2label[str(i)] for i in range(num_labels)]
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
