"""
In frozen mode, BERT weights don't update — only the projection head trains.
In fine-tuned mode, the entire BERT + projection is trained end-to-end.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class SemanticEncoder(nn.Module):
    """
    Text → ruBERT → [CLS] pooling → projection → (B, output_dim).
    """

    def __init__(
        self,
        model_name: str = "ai-forever/ruBert-base",
        output_dim: int = 256,
        dropout: float = 0.1,
        freeze_bert: bool = False,
        max_length: int = 128,
    ):
        super().__init__()
        self.max_length = max_length
        self.freeze_bert = freeze_bert

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_dim = self.bert.config.hidden_size  # 768 for ruBert-base

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.output_proj = nn.Sequential(
            nn.Linear(self.bert_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim

    def tokenize(self, texts: list[str], device: torch.device) -> dict:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoded.items()}

    def forward(
        self,
        texts: list[str] | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            texts: list of raw strings (will be tokenized internally)
            input_ids: (B, L) pre-tokenized IDs
            attention_mask: (B, L) mask
        Returns:
            (B, output_dim)
        """
        if texts is not None:
            device = next(self.parameters()).device
            tokens = self.tokenize(texts, device)
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]

        if self.freeze_bert:
            with torch.no_grad():
                outputs = self.bert(
                    input_ids=input_ids, attention_mask=attention_mask
                )
        else:
            outputs = self.bert(
                input_ids=input_ids, attention_mask=attention_mask
            )

        cls_emb = outputs.last_hidden_state[:, 0]  # (B, 768)
        return self.output_proj(cls_emb)
