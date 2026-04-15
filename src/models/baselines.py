import torch
import torch.nn as nn

from src.models.acoustic_encoder import AcousticEncoder
from src.models.semantic_encoder import SemanticEncoder


class AudioOnlyBaseline(nn.Module):

    def __init__(
        self,
        hubert_dim: int = 768,
        prosodic_dim: int = 5,
        hidden_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.3,
        pooling: str = "attention",
        use_prosodic: bool = True,
    ):
        super().__init__()
        self.encoder = AcousticEncoder(
            hubert_dim=hubert_dim,
            prosodic_dim=prosodic_dim,
            output_dim=hidden_dim,
            pooling=pooling,
            use_prosodic=use_prosodic,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        hubert: torch.Tensor,
        hubert_mask: torch.Tensor | None = None,
        prosodic: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        emb = self.encoder(hubert, hubert_mask, prosodic)
        return self.classifier(emb)


class TextOnlyBaseline(nn.Module):

    def __init__(
        self,
        model_name: str = "ai-forever/ruBert-base",
        hidden_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.3,
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.encoder = SemanticEncoder(
            model_name=model_name,
            output_dim=hidden_dim,
            freeze_bert=freeze_bert,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        texts: list[str] | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        emb = self.encoder(texts=texts, input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(emb)


class MultimodalConcatBaseline(nn.Module):

    def __init__(
        self,
        hubert_dim: int = 768,
        prosodic_dim: int = 5,
        text_model_name: str = "ai-forever/ruBert-base",
        hidden_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.3,
        pooling: str = "attention",
        use_prosodic: bool = True,
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.audio_encoder = AcousticEncoder(
            hubert_dim=hubert_dim,
            prosodic_dim=prosodic_dim,
            output_dim=hidden_dim,
            pooling=pooling,
            use_prosodic=use_prosodic,
        )
        self.text_encoder = SemanticEncoder(
            model_name=text_model_name,
            output_dim=hidden_dim,
            freeze_bert=freeze_bert,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        hubert: torch.Tensor,
        hubert_mask: torch.Tensor | None = None,
        prosodic: torch.Tensor | None = None,
        texts: list[str] | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        audio_emb = self.audio_encoder(hubert, hubert_mask, prosodic)
        text_emb = self.text_encoder(
            texts=texts, input_ids=input_ids, attention_mask=attention_mask
        )
        combined = torch.cat([audio_emb, text_emb], dim=-1)
        return self.classifier(combined)
