from typing import Dict, Optional, Sequence

import numpy as np
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence


class SupervisedDataCollator:
    """Collator for supervised fine-tuning, with optional bio-prior bias injection.

    bias_mode mirrors the --bias CLI argument:
      'no_bias'   — standard collation, no bio_prior added to batch
      'utr_only'  — Watson-Crick bias for single-nucleotide (UTR) tokens only
      'full'      — Watson-Crick bias for UTR and codon tokens
      'linearfold'— secondary-structure bias from pre-computed LinearFold pairs

    For WC modes, pass a pre-built wc_lookup tensor (from bias.wc.build_wc_lookup).
    For linearfold mode, pass the path to the .npz produced by generate_linearfold_bias.py.
    Collator's pads to the longest sequence though this is already done in SupervisedDataset.
    

    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        bias_mode: str = "no_bias",
        wc_lookup: Optional[torch.Tensor] = None,
        bias_npz_path: Optional[str] = None,
    ):
        valid = ("no_bias", "utr_only", "full", "linearfold")
        if bias_mode not in valid:
            raise ValueError(f"bias_mode must be one of {valid}; got '{bias_mode}'")
        if bias_mode in ("utr_only", "full") and wc_lookup is None:
            raise ValueError(f"bias_mode='{bias_mode}' requires wc_lookup")
        if bias_mode == "linearfold" and bias_npz_path is None:
            raise ValueError("bias_mode='linearfold' requires bias_npz_path")

        self.tokenizer = tokenizer
        self.bias_mode = bias_mode
        self.wc_lookup = wc_lookup

        if bias_mode == "linearfold":
            archive = np.load(bias_npz_path, allow_pickle=False)
            self.pairs_lookup: Dict[str, np.ndarray] = dict(archive)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Pop tx_id before collation — strings cannot be stacked into tensors
        tx_ids = [inst.pop("tx_id", None) for inst in instances]

        input_ids = pad_sequence(
            [inst["input_ids"] for inst in instances],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.tensor([inst["labels"] for inst in instances]).float()
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if self.bias_mode in ("utr_only", "full"):
            ids = batch["input_ids"]
            bias = self.wc_lookup[ids.unsqueeze(2), ids.unsqueeze(1)]  # (B, L, L)
            batch["bio_prior"] = bias.unsqueeze(1)                      # (B, 1, L, L)

        elif self.bias_mode == "linearfold":
            B, L = input_ids.shape
            bio_prior = torch.zeros(B, 1, L, L, dtype=torch.float32)
            for b, tx_id in enumerate(tx_ids):
                pairs = self.pairs_lookup.get(tx_id)
                if pairs is None:
                    raise KeyError(
                        f"tx_id '{tx_id}' not found in LinearFold bias NPZ. "
                        "Re-run generate_linearfold_bias.py to include all sequences."
                    )
                if len(pairs) == 0:
                    continue
                ti = pairs[:, 0] + 1  # +1 for CLS token at position 0
                tj = pairs[:, 1] + 1
                counts = pairs[:, 2].astype(np.float32)
                within = (ti < L) & (tj < L)
                ti, tj, counts = ti[within], tj[within], counts[within]
                bio_prior[b, 0, ti, tj] = torch.from_numpy(counts)
                bio_prior[b, 0, tj, ti] = torch.from_numpy(counts)
            batch["bio_prior"] = bio_prior

        return batch
