import torch
import torch.nn as nn


class AttentionPooling(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, T, D)
            mask: (B, T) bool — True = valid position

        Returns:
            (B, D) pooled output
        """
        scores = self.attention(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (B, T)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)


class AcousticEncoder(nn.Module):
    """
    Pipeline:
        HuBERT (B, T, 768)  →  pooling  →  (B, hubert_dim)
        Prosodic (B, 5)      →  linear   →  (B, prosodic_proj_dim)
        Concat               →  project  →  (B, output_dim)
    """

    def __init__(
        self,
        hubert_dim: int = 768,
        prosodic_dim: int = 5,
        prosodic_proj_dim: int = 32,
        output_dim: int = 256,
        pooling: str = "attention",
        dropout: float = 0.1,
        use_prosodic: bool = True,
    ):
        super().__init__()
        self.use_prosodic = use_prosodic
        self.pooling_type = pooling

        # Time-dimension pooling for HuBERT
        if pooling == "attention":
            self.pooling = AttentionPooling(hubert_dim)

        # Prosodic projection
        if use_prosodic:
            self.prosodic_proj = nn.Sequential(
                nn.Linear(prosodic_dim, prosodic_proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            combined_dim = hubert_dim + prosodic_proj_dim
        else:
            combined_dim = hubert_dim

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim

    def forward(
        self,
        hubert: torch.Tensor,
        hubert_mask: torch.Tensor | None = None,
        prosodic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            (B, output_dim) acoustic embedding
        """
        # Pool over time
        if self.pooling_type == "attention":
            h = self.pooling(hubert, hubert_mask)
        else:
            # Masked mean pooling
            if hubert_mask is not None:
                mask_f = hubert_mask.unsqueeze(-1).float()  # (B, T, 1)
                h = (hubert * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            else:
                h = hubert.mean(dim=1)

        # Fuse with prosodic features
        if self.use_prosodic and prosodic is not None:
            p = self.prosodic_proj(prosodic)
            h = torch.cat([h, p], dim=-1)

        return self.output_proj(h)
