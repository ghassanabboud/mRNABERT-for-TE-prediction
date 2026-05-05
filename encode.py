import argparse
import csv
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BertConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Extract mRNABERT embeddings from a CSV of pre-processed sequences.")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV file with sequences in the first column.")
    parser.add_argument("--output_path", type=str, required=True, help="Output .npz file path.")
    parser.add_argument("--model_name", type=str, default="YYLY66/mRNABERT")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pooling", type=str, default="mean", choices=["cls", "mean"],
                        help="cls: CLS token embedding; mean: mean over non-padding tokens.")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = BertConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="right", use_fast=True, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(args.model_name, config=config, trust_remote_code=True)
    model.to(device)
    model.eval()

    with open(args.csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        rows = list(reader)
    texts = [row[0] for row in rows]
    print(f"Loaded {len(texts)} sequences from {args.csv_path}")

    all_embeddings = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Encoding"):
        batch = texts[i : i + args.batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        last_hidden = out.last_hidden_state  # (B, L, H)
        if args.pooling == "cls":
            emb = last_hidden[:, 0, :]
        else:
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (last_hidden * mask).sum(1) / mask.sum(1)
        all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)  # (N, H)
    print(f"Embeddings shape: {embeddings.shape}")

    np.savez(args.output_path, embeddings=embeddings)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
