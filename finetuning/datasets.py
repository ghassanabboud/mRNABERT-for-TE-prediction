import csv
from typing import Dict

import torch
import transformers
from torch.utils.data import Dataset


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning on RiboNN translation efficiency data.

    Expects a CSV with columns: [metadata...], sequence, [label...].
    Always includes tx_id in returned items when present in the CSV metadata;
    
    Note that the HFTrainer will drop it via remove_unused_columns=True because it detects
    that tx_id is not a model input. Hence, when we need tx_id to reach the batch collator (for linearfold bias), 
    we set the HFTrainer's remove_unused_columns=False then make sure to drop tx_id from the batch in the collator. 

    The tokenizer passed at initialization should have the desired model_max_length set as attribute,
    this is currently done in train.py and train_biased.py via the TrainingArguments.model_max_length attribute.
    """

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()

        with open(data_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            data = list(reader)

        if "sequence" not in header:
            raise ValueError(f"CSV must have a 'sequence' column. Got: {header}")

        seq_idx = header.index("sequence")
        metadata_names = header[:seq_idx]
        label_names = header[seq_idx + 1:]

        texts = [row[seq_idx] for row in data]
        metadata = [[row[i] for i in range(seq_idx)] for row in data]
        labels = [
            [float(v) if v != "" else float("nan") for v in row[seq_idx + 1:]]
            for row in data
        ]

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(labels[0]) if labels else 0
        self.label_names = label_names
        self.sequences = texts
        self.metadata = metadata
        self.metadata_names = metadata_names

        self._tx_id_idx = metadata_names.index("tx_id") if "tx_id" in metadata_names else None

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict:
        item = dict(input_ids=self.input_ids[i], labels=self.labels[i])
        if self._tx_id_idx is not None:
            item["tx_id"] = self.metadata[i][self._tx_id_idx]
        return item
