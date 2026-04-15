import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """
    Input:  (B, T, input_dim)
    Output: (B, T, hidden_dim * 2)   [bidirectional]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * 2

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, T, D)
            mask: (B, T) bool — True = valid
        Returns:
            (B, T, 2*hidden_dim)
        """
        if mask is not None:
            lengths = mask.sum(dim=1).cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            output, _ = self.bilstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=x.shape[1]
            )
        else:
            output, _ = self.bilstm(x)

        return output
