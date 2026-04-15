"""
Creates:
    - fig7_valence_arousal[_ru].png/pdf — Grouped bar chart of V/A metrics
    - table5_valence_arousal[_ru].csv — Formatted table for thesis
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import FIGURES_DIR, METRICS_DIR

MODEL_ORDER = [
    "HF Text-Only (ruBERT)",
    "HF Audio-Only (HuBERT)",
    "HF-Dusha Pretrained",
    "CODA-Base",
    "CODA-UniAttn",
    "CODA-BiAttn",
    "CODA-BiAttn-Context",
    "CODA-Full",
    "CODA-Full (no prosodic)",
]


def load_results():
    with open(METRICS_DIR / "valence_arousal_all_models.json") as f:
        data = json.load(f)
    # Reorder
    by_name = {r["model"]: r for r in data}
    return [by_name[name] for name in MODEL_ORDER if name in by_name]


def make_bar_chart(results, lang="en"):
    models = [r["model"] for r in results]
    v_ccc = [r["valence_ccc"] for r in results]
    a_ccc = [r["arousal_ccc"] for r in results]
    v_mae = [r["valence_mae"] for r in results]
    a_mae = [r["arousal_mae"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(models))
    width = 0.35

    # Panel 1: CCC
    ax = axes[0]
    bars1 = ax.bar(x - width / 2, v_ccc, width, label="Valence" if lang == "en" else "Valentnost'", color="#2196F3")
    bars2 = ax.bar(x + width / 2, a_ccc, width, label="Arousal" if lang == "en" else "Vozbuzhdeniye", color="#FF5722")
    ax.set_ylabel("CCC", fontsize=12)
    title_ccc = "Concordance Correlation Coefficient (CCC)" if lang == "en" else "Koeffitsient konkordantnosti (CCC)"
    ax.set_title(title_ccc, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    # Panel 2: MAE
    ax = axes[1]
    bars3 = ax.bar(x - width / 2, v_mae, width, label="Valence" if lang == "en" else "Valentnost'", color="#2196F3")
    bars4 = ax.bar(x + width / 2, a_mae, width, label="Arousal" if lang == "en" else "Vozbuzhdeniye", color="#FF5722")
    ax.set_ylabel("MAE", fontsize=12)
    title_mae = "Mean Absolute Error (MAE)" if lang == "en" else "Srednyaya absolyutnaya oshibka (MAE)"
    ax.set_title(title_mae, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 0.5)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    # Arrow indicating lower is better
    note = "(lower is better)" if lang == "en" else "(men'she — luchshe)"
    ax.annotate(note, xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, fontstyle="italic", color="gray")

    for bar in bars3:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    for bar in bars4:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    return fig


def make_tables(results):
    rows = []
    for r in results:
        rows.append({
            "Model": r["model"],
            "Valence MAE": r["valence_mae"],
            "Valence CCC": r["valence_ccc"],
            "Valence Pearson": r["valence_pearson"],
            "Arousal MAE": r["arousal_mae"],
            "Arousal CCC": r["arousal_ccc"],
            "Arousal Pearson": r["arousal_pearson"],
        })
    df_en = pd.DataFrame(rows)

    df_ru = df_en.copy()
    df_ru.columns = [
        "Model'", "MAE (valentnost')", "CCC (valentnost')",
        "Pearson (valentnost')", "MAE (vozbuzhdeniye)",
        "CCC (vozbuzhdeniye)", "Pearson (vozbuzhdeniye)",
    ]
    return df_en, df_ru


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()

    for lang, suffix in [("en", ""), ("ru", "_ru")]:
        fig = make_bar_chart(results, lang=lang)
        for ext in ["png", "pdf"]:
            path = FIGURES_DIR / f"fig7_valence_arousal{suffix}.{ext}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved fig7_valence_arousal{suffix}.png/pdf")

    df_en, df_ru = make_tables(results)

    csv_en = FIGURES_DIR / "table5_valence_arousal.csv"
    csv_ru = FIGURES_DIR / "table5_valence_arousal_ru.csv"
    df_en.to_csv(csv_en, index=False, float_format="%.4f")
    df_ru.to_csv(csv_ru, index=False, float_format="%.4f")
    print(f"Saved {csv_en}")
    print(f"Saved {csv_ru}")


if __name__ == "__main__":
    main()
