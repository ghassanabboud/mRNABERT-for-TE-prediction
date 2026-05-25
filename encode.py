import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, BertConfig

from regression_multilabel import SupervisedDataset, DataCollatorForSupervisedDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Extract mRNABERT embeddings from a CSV of pre-processed sequences.")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV with a 'sequence' column.")
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="right", use_fast=True, trust_remote_code=True
    )
    config = BertConfig.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, config=config, trust_remote_code=True)
    model.to(device)
    model.eval()

    dataset = SupervisedDataset(data_path=args.csv_path, tokenizer=tokenizer)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    print(f"Loaded {len(dataset)} sequences from {args.csv_path}")

    all_embeddings = []
    for batch in tqdm(loader, desc="Encoding"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state  # (B, L, H)
        print("shape of last_hidden:", last_hidden.shape)
        if args.pooling == "cls":
            emb = last_hidden[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            emb = (last_hidden * mask).sum(1) / mask.sum(1)
        all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)  # (N, H)
    print(f"Embeddings shape: {embeddings.shape}")

    np.savez(args.output_path, embeddings=embeddings)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
