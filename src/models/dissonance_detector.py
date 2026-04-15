import numpy as np
import torch
import torch.nn as nn


class DissonanceDetector(nn.Module):
    """
    Pipeline:
        attention_weights → entropy H(α) → MLP → dissonance score ∈ [0, 1]
        entropy features  → Isolation Forest → anomaly flag (-1 or 1)
    """

    def __init__(self, hidden_dim: int = 64, contamination: float = 0.05):
        super().__init__()
        # Learnable head: 2 entropy values (a2t, t2a) → dissonance score
        self.entropy_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.contamination = contamination
        self._isolation_forest = None

    @staticmethod
    def compute_attention_entropy(
        attn_weights: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            attn_weights: (B, T_q, T_k) attention probabilities
            mask:         (B, T_q) bool — valid query positions
        Returns:
            (B,) mean entropy per sample
        """
        p = attn_weights.clamp(min=1e-8)
        entropy = -(p * p.log()).sum(dim=-1)  # (B, T_q)

        if mask is not None:
            entropy = entropy * mask.float()
            return entropy.sum(dim=-1) / mask.float().sum(dim=-1).clamp(min=1)
        return entropy.mean(dim=-1)

    def forward(
        self,
        a2t_weights: torch.Tensor,
        t2a_weights: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            a2t_weights: (B, T_a, T_t)
            t2a_weights: (B, T_t, T_a)
            audio_mask:  (B, T_a)
            text_mask:   (B, T_t)
        Returns:
            dissonance_score:  (B,) in [0, 1]
            entropy_features:  (B, 2) raw [H_a2t, H_t2a]
        """
        h_a2t = self.compute_attention_entropy(a2t_weights, audio_mask)
        h_t2a = self.compute_attention_entropy(t2a_weights, text_mask)

        entropy_features = torch.stack([h_a2t, h_t2a], dim=-1)  # (B, 2)
        score = self.entropy_proj(entropy_features).squeeze(-1)  # (B,)
        return score, entropy_features

    # --- Isolation Forest (post-training) ---

    def fit_isolation_forest(self, entropy_features: np.ndarray) -> None:
        from sklearn.ensemble import IsolationForest

        self._isolation_forest = IsolationForest(
            contamination=self.contamination, random_state=42
        )
        self._isolation_forest.fit(entropy_features)

    def predict_anomaly(self, entropy_features: np.ndarray) -> np.ndarray:
        if self._isolation_forest is None:
            raise RuntimeError("Call fit_isolation_forest() first")
        return self._isolation_forest.predict(entropy_features)
