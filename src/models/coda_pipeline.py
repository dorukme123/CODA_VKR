"""

    base           — concat fusion (no cross-attention)
    uniattn        — audio → text unidirectional cross-attention
    biattn         — audio ↔ text bidirectional cross-attention  [MAIN MODEL]
    biattn_context — + BiLSTM temporal context encoder
    full           — + dissonance detection head

"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.models.acoustic_encoder import AttentionPooling
from src.models.context_encoder import ContextEncoder
from src.models.cross_attention import (
    BidirectionalCrossAttention,
    UnidirectionalCrossAttention,
)
from src.models.dissonance_detector import DissonanceDetector

VARIANTS = ("base", "uniattn", "biattn", "biattn_context", "full")


class CODAPipeline(nn.Module):

    def __init__(
        self,
        variant: str = "biattn",
        # Encoder dims
        hubert_dim: int = 768,
        bert_model_name: str = "ai-forever/ruBert-base",
        prosodic_dim: int = 5,
        # Cross-attention
        cross_attn_dim: int = 256,
        cross_attn_heads: int = 8,
        cross_attn_layers: int = 1,
        cross_attn_dropout: float = 0.1,
        # Context encoder
        context_hidden: int = 128,
        context_layers: int = 2,
        context_dropout: float = 0.2,
        # Prosodic
        prosodic_proj_dim: int = 32,
        # Classifier
        classifier_hidden: int = 256,
        classifier_dropout: float = 0.3,
        num_classes: int = 4,
        # BERT config
        freeze_bert: bool = True,
        max_text_length: int = 128,
        # Dissonance
        dissonance_contamination: float = 0.05,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {VARIANTS}")
        self.variant = variant
        self.max_text_length = max_text_length

        # ---- Audio projection ----
        self.audio_proj = nn.Sequential(
            nn.Linear(hubert_dim, cross_attn_dim),
            nn.ReLU(),
            nn.Dropout(cross_attn_dropout),
        )

        # ---- Text encoder (ruBERT) ----
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.text_proj = nn.Sequential(
            nn.Linear(bert_hidden, cross_attn_dim),
            nn.ReLU(),
            nn.Dropout(cross_attn_dropout),
        )

        # ---- Cross-attention ----
        self._has_cross_attn = variant != "base"
        self._has_biattn = variant in ("biattn", "biattn_context", "full")

        if self._has_cross_attn:
            if self._has_biattn:
                self.cross_attention = BidirectionalCrossAttention(
                    cross_attn_dim, cross_attn_heads,
                    cross_attn_dropout, cross_attn_layers,
                )
            else:
                self.cross_attention = UnidirectionalCrossAttention(
                    cross_attn_dim, cross_attn_heads,
                    cross_attn_dropout, cross_attn_layers,
                )

        # ---- Context encoder ----
        self._has_context = variant in ("biattn_context", "full")
        if self._has_context:
            self.context_encoder = ContextEncoder(
                cross_attn_dim, context_hidden, context_layers, context_dropout,
            )
            audio_pool_dim = self.context_encoder.output_dim
        else:
            audio_pool_dim = cross_attn_dim

        # ---- Pooling ----
        self.audio_pool = AttentionPooling(audio_pool_dim)
        self.text_pool = AttentionPooling(cross_attn_dim)

        # ---- Prosodic ----
        self._has_prosodic = prosodic_dim > 0
        if self._has_prosodic:
            self.prosodic_proj = nn.Sequential(
                nn.Linear(prosodic_dim, prosodic_proj_dim),
                nn.ReLU(),
            )
            prosodic_out = prosodic_proj_dim
        else:
            prosodic_out = 0

        # ---- Classifier ----
        clf_input_dim = audio_pool_dim + cross_attn_dim + prosodic_out
        self.classifier = nn.Sequential(
            nn.Linear(clf_input_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

        # ---- Dissonance head ----
        self._has_dissonance = variant == "full"
        if self._has_dissonance:
            self.dissonance_head = DissonanceDetector(
                contamination=dissonance_contamination,
            )

    # ------------------------------------------------------------------
    def _encode_text(
        self, texts: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ruBERT → return (hidden_states, mask)."""
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        if not any(p.requires_grad for p in self.bert.parameters()):
            with torch.no_grad():
                out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        return out.last_hidden_state, attention_mask.bool()

    # ------------------------------------------------------------------
    def forward(
        self,
        hubert: torch.Tensor,
        hubert_mask: torch.Tensor | None = None,
        prosodic: torch.Tensor | None = None,
        texts: list[str] | None = None,
        rubert_emb: torch.Tensor | None = None,
        rubert_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        """
        Args:
            hubert:      (B, T_a, 768)
            hubert_mask: (B, T_a)
            prosodic:    (B, 5)
            texts:       list of B strings (used if rubert_emb is None)
            rubert_emb:  (B, T_t, 768) pre-computed ruBERT hidden states
            rubert_mask: (B, T_t) mask for pre-computed embeddings
        Returns dict:
            logits:           (B, num_classes)
            dissonance_score: (B,) or None
            a2t_weights:      (B, T_a, T_t) or None
            t2a_weights:      (B, T_t, T_a) or None
        """
        device = hubert.device
        B = hubert.shape[0]

        # --- Project audio ---
        audio_seq = self.audio_proj(hubert)  # (B, T_a, D)

        # --- Encode + project text ---
        if rubert_emb is not None:
            # Use pre-computed ruBERT embeddings (skips BERT forward pass)
            text_hidden = rubert_emb
            text_mask = rubert_mask
        else:
            # On-the-fly encoding via ruBERT
            text_hidden, text_mask = self._encode_text(texts, device)
        text_seq = self.text_proj(text_hidden)  # (B, T_t, D)

        # --- Fusion ---
        a2t_weights = None
        t2a_weights = None

        if self._has_cross_attn:
            if self._has_biattn:
                audio_seq, text_seq, a2t_weights, t2a_weights = (
                    self.cross_attention(audio_seq, text_seq, hubert_mask, text_mask)
                )
            else:
                audio_seq, a2t_weights = self.cross_attention(
                    audio_seq, text_seq, hubert_mask, text_mask
                )

        # --- Context ---
        if self._has_context:
            audio_seq = self.context_encoder(audio_seq, hubert_mask)

        # --- Pool ---
        audio_emb = self.audio_pool(audio_seq, hubert_mask)
        text_emb = self.text_pool(text_seq, text_mask)

        # --- Prosodic ---
        if self._has_prosodic and prosodic is not None:
            prosodic_emb = self.prosodic_proj(prosodic)
            fused = torch.cat([audio_emb, text_emb, prosodic_emb], dim=-1)
        else:
            fused = torch.cat([audio_emb, text_emb], dim=-1)
        logits = self.classifier(fused)

        # --- Dissonance ---
        dissonance_score = None
        if self._has_dissonance and a2t_weights is not None and t2a_weights is not None:
            dissonance_score, _ = self.dissonance_head(
                a2t_weights, t2a_weights, hubert_mask, text_mask
            )

        return {
            "logits": logits,
            "dissonance_score": dissonance_score,
            "a2t_weights": a2t_weights,
            "t2a_weights": t2a_weights,
        }
