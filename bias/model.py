import contextlib
import json
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class BioPriorAttention(nn.Module):
    """Multi-head self-attention with an additive biological prior bias.

    The prior shifts raw attention logits before softmax:
        scores[b, h, i, j] = (Q_i · K_j) / sqrt(d_k) + bias[b, h, i, j]

    bias shape: (B, num_heads, L, L) — broadcast from (B, 1, L, L) if needed.
    Passing bias=None runs standard attention with no prior.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        """Build the query/key/value/output projections for multi-head attention.

        Parameters
        ----------
        hidden_size : int
            Width of the input and output hidden states. Must be divisible
            by `num_heads`.
        num_heads : int
            Number of attention heads to split `hidden_size` into.
        dropout : float, optional
            Dropout probability applied to attention weights after softmax.
            Default 0.1.
        """
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,            # (B, L, H)
        extended_attention_mask: torch.Tensor,  # (B, 1, 1, L) additive mask
        bio_prior_bias: Optional[torch.Tensor] = None,  # (B, num_heads|1, L, L)
    ) -> torch.Tensor:
        """Run one self-attention pass, adding the bio-prior bias to the
        attention scores before softmax so it steers which tokens attend to
        each other (e.g. base-paired nucleotides, Watson-Crick partners).

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input token representations, shape (B, L, H).
        extended_attention_mask : torch.Tensor
            Additive padding mask, shape (B, 1, 1, L): 0 for real tokens,
            large negative for padding, added to the attention scores so
            padding positions get ~0 weight after softmax.
        bio_prior_bias : Optional[torch.Tensor], optional
            Additive attention bias, shape (B, num_heads, L, L) or
            (B, 1, L, L) (broadcast across heads). Added to the raw
            attention scores before the padding mask and softmax. If None,
            attention runs with no prior. Default None.

        Returns
        -------
        torch.Tensor
            Attention output, shape (B, L, H).
        """
        B, L, H = hidden_states.shape
        nh, hd = self.num_heads, self.head_dim

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(B, L, nh, hd).transpose(1, 2)  # (B, nh, L, hd)

        q = split_heads(self.q_proj(hidden_states))
        k = split_heads(self.k_proj(hidden_states))
        v = split_heads(self.v_proj(hidden_states))

        scores = (q @ k.transpose(-2, -1)) / (hd ** 0.5)  # (B, nh, L, L)

        if bio_prior_bias is not None:
            scores = scores + bio_prior_bias

        scores = scores + extended_attention_mask  # zero out padding positions
        attn = self.dropout(scores.softmax(dim=-1))

        context = (attn @ v).transpose(1, 2).contiguous().view(B, L, H)
        return self.out_proj(context)


class mRNABERTWithBioPriorHead(nn.Module):
    """Frozen mRNABERT backbone + trainable bio-prior attention layers + classifier.

    Architecture:
        1. BERT backbone  →  last_hidden_state  (B, L, 768)
        2. for each of num_bio_layers:
               LayerNorm(hidden + BioPriorAttention(hidden, ext_mask, bias))
        3. CLS-pool  →  dropout  →  Linear  →  logits  (B, num_labels)
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int = 768,
        num_heads: int = 8,
        num_labels: int = 78,
        dropout: float = 0.1,
        num_bio_layers: int = 1,
    ):
        """Wrap a pre-trained BERT backbone with bio-prior attention layers
        and a regression head for predicting translation efficiency.

        Parameters
        ----------
        base_model : nn.Module
            Pre-trained BERT-style encoder (e.g. mRNABERT) that returns
            last_hidden_state as the first output element. Not frozen here;
            call `freeze_bert()` separately if needed.
        hidden_size : int, optional
            Hidden dimension of `base_model`'s outputs and of the bio-prior
            attention layers. Default 768.
        num_heads : int, optional
            Number of attention heads in each bio-prior attention layer.
            Default 8.
        num_labels : int, optional
            Number of output regression targets (e.g. cell types with a TE
            value to predict). Default 78 for human 
        dropout : float, optional
            Dropout probability used in the bio-prior attention layers and
            before the classifier. Default 0.1.
        num_bio_layers : int, optional
            Number of stacked bio-prior attention + LayerNorm blocks applied
            after the backbone. Default 1.
        """
        super().__init__()
        self.bert = base_model
        self.bio_attn_layers = nn.ModuleList([
            BioPriorAttention(hidden_size, num_heads, dropout)
            for _ in range(num_bio_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_bio_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> Tuple["mRNABERTWithBioPriorHead", dict]:
        """Reconstruct a trained model from a checkpoint directory written by
        train_biased.py, without needing the caller to know the architecture
        hyperparameters (num_heads, num_bio_layers, bias mode, ...) used at
        training time.

        Parameters
        ----------
        checkpoint_path : str
            Directory containing `bio_prior_config.json` and
            `pytorch_model.bin`, as written by train_biased.py.
        device : str, optional
            Device to move the model to after loading weights. Default "cpu".

        Returns
        -------
        Tuple[mRNABERTWithBioPriorHead, dict]
            The reconstructed model in eval mode, and the raw config dict
            (includes "bias" and "id2label", which callers need to build a
            matching SupervisedDataCollator and label predictions; not
            attributes of the model itself since the model has no notion of
            bias mode or label names).
        """
        with open(os.path.join(checkpoint_path, "bio_prior_config.json")) as f:
            cfg = json.load(f)

        base_config = BertConfig.from_pretrained(cfg["base_model_name"])
        base_model = AutoModel.from_pretrained(cfg["base_model_name"], config=base_config, trust_remote_code=True)

        model = cls(
            base_model=base_model,
            hidden_size=cfg["hidden_size"],
            num_heads=cfg["num_heads"],
            num_labels=cfg["num_labels"],
            dropout=cfg["dropout"],
            num_bio_layers=cfg["num_bio_layers"],
        )
        state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, cfg

    def freeze_bert(self) -> None:
        """Disable gradient updates for all backbone parameters, so training
        only updates the bio-prior attention layers and classifier head.

        Returns
        -------
        None
        """
        for p in self.bert.parameters():
            p.requires_grad = False

    def count_parameters(self) -> dict:
        """Count trainable vs. frozen parameters, e.g. to confirm
        `freeze_bert()` took effect or to report model size.

        Returns
        -------
        dict
            Dictionary with keys "trainable", "frozen", and "total", each
            mapping to a parameter count (int).
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "frozen": total - trainable, "total": total}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bio_prior_bias: Optional[torch.Tensor] = None,  # (B, 1|nh, L, L)
        labels: Optional[torch.Tensor] = None,           # unused; loss computed in trainer
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        """Run the backbone, apply the bio-prior attention layers on top of
        its output, and predict a translation-efficiency value per cell type
        from the CLS token.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token ids, shape (B, L).
        attention_mask : torch.Tensor
            1 for real tokens, 0 for padding, shape (B, L). Used both by the
            backbone and to build the additive padding mask for the
            bio-prior attention layers.
        bio_prior_bias : Optional[torch.Tensor], optional
            Additive attention bias passed to each `BioPriorAttention` layer,
            shape (B, num_heads, L, L) or (B, 1, L, L). If None, the
            bio-prior layers run as plain self-attention. Default None.
        labels : Optional[torch.Tensor], optional
            Unused here; the loss is computed by the trainer, not the model.
            Default None. it is kept here so that the model signature signals to the HFTrainer not to pop "labels" from the batch dict.
        token_type_ids : Optional[torch.Tensor], optional
            Segment ids forwarded to the backbone if provided. Default None. Effectively useless in this repo but kept to respect BERT signature.

        Returns
        -------
        SequenceClassifierOutput
            HF output object whose `logits` field holds the predicted TE
            values, shape (B, num_labels).
        """
        backbone_frozen = not next(self.bert.parameters()).requires_grad
        ctx = torch.no_grad() if backbone_frozen else contextlib.nullcontext()
        with ctx:
            bert_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
            if token_type_ids is not None:
                bert_kwargs["token_type_ids"] = token_type_ids
            hidden = self.bert(**bert_kwargs)[0]  # (B, L, H)

        # Additive mask: 0 for real tokens, large negative for padding
        ext_mask = (1.0 - attention_mask[:, None, None, :].float()) * -1e4  # (B,1,1,L)

        for bio_attn, norm in zip(self.bio_attn_layers, self.norms):
            hidden = norm(hidden + bio_attn(hidden, ext_mask, bio_prior_bias))

        cls_rep = self.dropout(hidden[:, 0, :])  # CLS token  (B, H)
        logits = self.classifier(cls_rep)        # (B, num_labels)

        return SequenceClassifierOutput(logits=logits)
