import numpy as np
import torch
import torch.nn as nn


def compute_class_weights(
    labels: list | np.ndarray,
    num_classes: int = 4,
    method: str = "sqrt_inverse",
) -> torch.Tensor:
    """
    Args:
            "inverse"      — weight = 1 / count, normalized. Aggressive — can
                             over-penalize majority class on imbalanced data.
            "sqrt_inverse" — weight = 1 / sqrt(count), normalized. Good balance
                             between correcting imbalance and not over-shooting.
            "effective"    — effective number of samples (Cui et al., 2019).
            "none"         — uniform weights.
    Return:
        (num_classes,) float tensor.
    """
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)

    if method == "inverse":
        w = 1.0 / counts
        w = w / w.sum() * num_classes
    elif method == "sqrt_inverse":
        w = 1.0 / np.sqrt(counts)
        w = w / w.sum() * num_classes
    elif method == "effective":
        beta = 0.9999
        effective = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / effective
        w = w / w.sum() * num_classes
    else:
        w = np.ones(num_classes)

    return torch.tensor(w, dtype=torch.float32)


class WeightedCELoss(nn.Module):

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=label_smoothing
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets)
