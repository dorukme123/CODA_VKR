"""
    angry    → valence=-1.0, arousal= 1.0
    sad      → valence=-1.0, arousal=-1.0
    positive → valence= 1.0, arousal= 1.0
    neutral  → valence= 0.0, arousal= 0.0

Usage:
    python -m src.training.eval_valence_arousal
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DUSHA_EMOTIONS, DUSHA_LABEL2ID, METRICS_DIR
from src.data.occ_mapping import get_occ_features
from src.utils.metrics import dimensional_metrics_from_confusion_matrix


# V/A mappings from OCC
LABEL_TO_VALENCE = {}
LABEL_TO_AROUSAL = {}
for emotion in DUSHA_EMOTIONS:
    idx = DUSHA_LABEL2ID[emotion]
    occ = get_occ_features(emotion, dataset="dusha")
    LABEL_TO_VALENCE[idx] = occ["valence_numeric"]
    LABEL_TO_AROUSAL[idx] = occ["arousal_numeric"]

MODEL_DISPLAY_NAMES = {
    "hf_text_only": "HF Text-Only (ruBERT)",
    "hf_audio_only": "HF Audio-Only (HuBERT)",
    "hf_dusha_pretrained": "HF-Dusha Pretrained",
    "coda_base": "CODA-Base",
    "coda_uniattn": "CODA-UniAttn",
    "coda_biattn": "CODA-BiAttn",
    "coda_biattn_context": "CODA-BiAttn-Context",
    "coda_full": "CODA-Full",
    "coda_full_noprosodic": "CODA-Full (no prosodic)",
}


def main():
    print("=" * 70)
    print("VALENCE / AROUSAL EVALUATION (OCC-derived targets)")
    print("=" * 70)
    print()
    print("OCC Mapping:")
    for emotion in DUSHA_EMOTIONS:
        idx = DUSHA_LABEL2ID[emotion]
        v, a = LABEL_TO_VALENCE[idx], LABEL_TO_AROUSAL[idx]
        print(f"  {emotion:10s} -> valence={v:+.1f}, arousal={a:+.1f}")
    print()

    test_files = sorted(METRICS_DIR.glob("*_test_dusha.json"))
    if not test_files:
        print("No test result files found in", METRICS_DIR)
        return

    all_results = []

    for fpath in test_files:
        with open(fpath) as f:
            data = json.load(f)

        model_key = data["model"]
        cm = np.array(data["metrics"]["confusion_matrix"])
        display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)

        dim_metrics = dimensional_metrics_from_confusion_matrix(
            cm, LABEL_TO_VALENCE, LABEL_TO_AROUSAL,
        )

        result = {
            "model": display_name,
            "model_key": model_key,
            **dim_metrics,
        }
        all_results.append(result)

        print(f"  {display_name}")
        print(f"    Valence — MAE: {dim_metrics['valence_mae']:.4f}  "
              f"CCC: {dim_metrics['valence_ccc']:.4f}  "
              f"Pearson: {dim_metrics['valence_pearson']:.4f}")
        print(f"    Arousal — MAE: {dim_metrics['arousal_mae']:.4f}  "
              f"CCC: {dim_metrics['arousal_ccc']:.4f}  "
              f"Pearson: {dim_metrics['arousal_pearson']:.4f}")
        print()

    out_json = METRICS_DIR / "valence_arousal_all_models.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved JSON: {out_json}")

    df = pd.DataFrame(all_results)
    df = df[["model", "valence_mae", "valence_ccc", "valence_pearson",
             "arousal_mae", "arousal_ccc", "arousal_pearson"]]

    csv_path = METRICS_DIR / "valence_arousal_all_models.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved CSV:  {csv_path}")

    # (Russian headers)
    df_ru = df.copy()
    df_ru.columns = ["Модель", "MAE (валентность)", "CCC (валентность)",
                     "Pearson (валентность)", "MAE (возбуждение)",
                     "CCC (возбуждение)", "Pearson (возбуждение)"]
    csv_ru_path = METRICS_DIR / "valence_arousal_all_models_ru.csv"
    df_ru.to_csv(csv_ru_path, index=False, float_format="%.4f")
    print(f"Saved CSV (RU): {csv_ru_path}")

    print()
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    header = f"{'Model':<28s} {'V-MAE':>6s} {'V-CCC':>6s} {'V-r':>6s}  {'A-MAE':>6s} {'A-CCC':>6s} {'A-r':>6s}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['model']:<28s} "
              f"{r['valence_mae']:>6.4f} {r['valence_ccc']:>6.4f} {r['valence_pearson']:>6.4f}  "
              f"{r['arousal_mae']:>6.4f} {r['arousal_ccc']:>6.4f} {r['arousal_pearson']:>6.4f}")


if __name__ == "__main__":
    main()
