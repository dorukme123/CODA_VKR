import torch
import torch.nn as nn


class CrossModalAttentionLayer(nn.Module):

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, T_q, D) query sequence
            kv:    (B, T_k, D) key-value sequence
            kv_mask: (B, T_k) bool — True = valid, False = padding
        Returns:
            output:       (B, T_q, D) attended query
            attn_weights: (B, T_q, T_k) attention probabilities
        """
        key_padding_mask = ~kv_mask if kv_mask is not None else None

        attended, attn_weights = self.cross_attn(
            query, kv, kv, key_padding_mask=key_padding_mask
        )
        x = self.norm1(query + attended)
        x = self.norm2(x + self.ffn(x))
        return x, attn_weights


class UnidirectionalCrossAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossModalAttentionLayer(dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        audio_seq: torch.Tensor,
        text_seq: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            audio_attended: (B, T_a, D)
            a2t_weights:    (B, T_a, T_t) from the last layer
        """
        x = audio_seq
        a2t_weights = None
        for layer in self.layers:
            x, a2t_weights = layer(x, text_seq, kv_mask=text_mask)
        return x, a2t_weights


class BidirectionalCrossAttention(nn.Module):
    """
    Audio ↔ Text bidirectional cross-attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()
        self.a2t_layers = nn.ModuleList(
            [CrossModalAttentionLayer(dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.t2a_layers = nn.ModuleList(
            [CrossModalAttentionLayer(dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        audio_seq: torch.Tensor,
        text_seq: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            audio_attended: (B, T_a, D)
            text_attended:  (B, T_t, D)
            a2t_weights:    (B, T_a, T_t) — audio attending to text
            t2a_weights:    (B, T_t, T_a) — text attending to audio
        """
        a = audio_seq
        t = text_seq
        a2t_w = t2a_w = None

        for a2t_layer, t2a_layer in zip(self.a2t_layers, self.t2a_layers):
            # Both directions use the PREVIOUS step's representations
            a_new, a2t_w = a2t_layer(a, t, kv_mask=text_mask)
            t_new, t2a_w = t2a_layer(t, a, kv_mask=audio_mask)
            a = a_new
            t = t_new

        return a, t, a2t_w, t2a_w
