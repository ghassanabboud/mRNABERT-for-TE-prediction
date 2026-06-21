import contextlib
from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput


class BioPriorAttention(nn.Module):
    """
    Multi-head self-attention with an additive biological prior bias.

    The prior shifts raw attention logits before softmax:
        scores[b, h, i, j] = (Q_i · K_j) / sqrt(d_k) + bias[b, h, i, j]

    bias shape: (B, num_heads, L, L) — broadcast from (B, 1, L, L) if needed.
    Passing bias=None runs standard attention with no prior.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
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
    """
    Frozen mRNABERT backbone + one trainable bio-prior attention layer + classifier.

    Architecture (forward pass):
        1. BERT backbone  →  last_hidden_state  (B, L, 768)
        2. for each of num_bio_layers:
               LayerNorm(hidden + BioPriorAttention(hidden, ext_mask, bias))
        3. CLS-pool  →  dropout  →  Linear  →  logits  (B, num_labels)

    Args:
        base_model:     AutoModel backbone
        hidden_size:    BERT hidden dimension (768 for mRNABERT)
        num_heads:      attention heads per bio-prior layer
        num_labels:     regression/classification outputs
        dropout:        dropout rate for attention and classifier
        num_bio_layers: number of stacked BioPriorAttention + LayerNorm blocks
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

    def freeze_bert(self) -> None:
        for p in self.bert.parameters():
            p.requires_grad = False

    def count_parameters(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "frozen": total - trainable, "total": total}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bio_prior_bias: Optional[torch.Tensor] = None,  # (B, 1|nh, L, L) for Option A
        labels: Optional[torch.Tensor] = None,           # unused here; loss in trainer
        token_type_ids: Optional[torch.Tensor] = None,   # accepted but forwarded only if needed
    ) -> SequenceClassifierOutput:
        backbone_frozen = not next(self.bert.parameters()).requires_grad
        ctx = torch.no_grad() if backbone_frozen else contextlib.nullcontext()
        with ctx:
            bert_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
            if token_type_ids is not None:
                bert_kwargs["token_type_ids"] = token_type_ids
            hidden = self.bert(**bert_kwargs)[0]  # (B, L, H) — works for tuple or ModelOutput

        B, L, _ = hidden.shape

        # Additive mask: 0 for real tokens, large negative for padding
        ext_mask = (1.0 - attention_mask[:, None, None, :].float()) * -1e4  # (B,1,1,L)

        for bio_attn, norm in zip(self.bio_attn_layers, self.norms):
            hidden = norm(hidden + bio_attn(hidden, ext_mask, bio_prior_bias))

        cls_rep = self.dropout(hidden[:, 0, :])  # CLS token  (B, H)
        logits = self.classifier(cls_rep)        # (B, num_labels)

        return SequenceClassifierOutput(logits=logits)
