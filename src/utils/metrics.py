import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    label_names: list[str] | None = None,
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    wa = accuracy_score(y_true, y_pred)
    ua = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    if label_names is None:
        label_names = [str(i) for i in range(len(per_class))]

    return {
        "weighted_accuracy": float(wa),
        "unweighted_accuracy": float(ua),
        "macro_f1": float(macro_f1),
        "per_class_f1": {
            name: float(f) for name, f in zip(label_names, per_class)
        },
        "confusion_matrix": cm.tolist(),
    }


def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    CCC = 2 * cov(y_true, y_pred) / (var(y_true) + var(y_pred) + (mean_true - mean_pred)^2)

    Returns value in [-1, 1]. 1 = perfect agreement.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mean_true = y_true.mean()
    mean_pred = y_pred.mean()
    var_true = y_true.var()
    var_pred = y_pred.var()
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom < 1e-12:
        return 0.0
    return float(2.0 * covariance / denom)


def compute_dimensional_metrics(
    y_true_labels: np.ndarray,
    y_pred_labels: np.ndarray,
    label_to_valence: dict[int, float],
    label_to_arousal: dict[int, float],
) -> dict:
    y_true_labels = np.asarray(y_true_labels)
    y_pred_labels = np.asarray(y_pred_labels)

    # Map labels to continuous V/A values
    true_valence = np.array([label_to_valence[l] for l in y_true_labels])
    pred_valence = np.array([label_to_valence[l] for l in y_pred_labels])
    true_arousal = np.array([label_to_arousal[l] for l in y_true_labels])
    pred_arousal = np.array([label_to_arousal[l] for l in y_pred_labels])

    # MAE
    valence_mae = float(np.mean(np.abs(true_valence - pred_valence)))
    arousal_mae = float(np.mean(np.abs(true_arousal - pred_arousal)))

    # CCC
    valence_ccc = concordance_correlation_coefficient(true_valence, pred_valence)
    arousal_ccc = concordance_correlation_coefficient(true_arousal, pred_arousal)

    # Pearson correlation
    if true_valence.std() > 1e-12 and pred_valence.std() > 1e-12:
        valence_pearson = float(np.corrcoef(true_valence, pred_valence)[0, 1])
    else:
        valence_pearson = 0.0

    if true_arousal.std() > 1e-12 and pred_arousal.std() > 1e-12:
        arousal_pearson = float(np.corrcoef(true_arousal, pred_arousal)[0, 1])
    else:
        arousal_pearson = 0.0

    return {
        "valence_mae": valence_mae,
        "valence_ccc": valence_ccc,
        "valence_pearson": valence_pearson,
        "arousal_mae": arousal_mae,
        "arousal_ccc": arousal_ccc,
        "arousal_pearson": arousal_pearson,
    }


def dimensional_metrics_from_confusion_matrix(
    cm: np.ndarray,
    label_to_valence: dict[int, float],
    label_to_arousal: dict[int, float],
) -> dict:
    cm = np.asarray(cm)
    num_classes = cm.shape[0]

    true_labels = []
    pred_labels = []
    for i in range(num_classes):
        for j in range(num_classes):
            count = int(cm[i, j])
            true_labels.extend([i] * count)
            pred_labels.extend([j] * count)

    return compute_dimensional_metrics(
        np.array(true_labels), np.array(pred_labels),
        label_to_valence, label_to_arousal,
    )


def format_metrics(metrics: dict) -> str:
    lines = [
        f"WA:       {metrics['weighted_accuracy']:.4f}",
        f"UA:       {metrics['unweighted_accuracy']:.4f}",
        f"Macro F1: {metrics['macro_f1']:.4f}",
        "Per-class F1:",
    ]
    for name, f1_val in metrics["per_class_f1"].items():
        lines.append(f"  {name:12s} {f1_val:.4f}")
    return "\n".join(lines)
