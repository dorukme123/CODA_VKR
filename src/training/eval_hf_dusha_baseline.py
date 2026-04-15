"""
Usage:
    python -m src.training.eval_hf_dusha_baseline
    python -m src.training.eval_hf_dusha_baseline --max-samples 100
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DEVICE,
    DUSHA_EMOTIONS,
    HUBERT_DUSHA,
    METRICS_DIR,
    PREPROCESSED_DIR,
    SAMPLE_RATE,
    MAX_AUDIO_SAMPLES,
    seed_everything,
)
from src.utils.metrics import compute_metrics, format_metrics

# HF: {0: neutral, 1: angry, 2: positive, 3: sad, 4: other}
# CODA: {0: angry, 1: sad, 2: neutral, 3: positive}
HF_ID2EMOTION = {0: "neutral", 1: "angry", 2: "positive", 3: "sad", 4: "other"}
OUR_EMOTION2ID = {e: i for i, e in enumerate(DUSHA_EMOTIONS)}


def hf_pred_to_our_label(hf_pred_id: int) -> int | None:
    emotion = HF_ID2EMOTION[hf_pred_id]
    return OUR_EMOTION2ID.get(emotion)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate HF Dusha baseline")
    p.add_argument("--test-manifest", type=str,
                   default=str(PREPROCESSED_DIR / "dusha" / "crowd_test" / "manifest.jsonl"))
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Batch size (HuBERT-Large is big, keep low)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_audio(audio_path: str) -> torch.Tensor | None:
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.shape[1] > MAX_AUDIO_SAMPLES:
            waveform = waveform[:, :MAX_AUDIO_SAMPLES]
        return waveform.squeeze(0)
    except Exception as e:
        print(f"  Error loading {audio_path}: {e}")
        return None


def main():
    args = parse_args()
    seed_everything(args.seed)

    # Load manifest
    entries = []
    with open(args.test_manifest, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    if args.max_samples:
        entries = entries[:args.max_samples]
    print(f"Test samples: {len(entries)}")

    print(f"Loading model: {HUBERT_DUSHA}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertForSequenceClassification.from_pretrained(HUBERT_DUSHA)
    model = model.to(DEVICE).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    all_preds = []
    all_labels = []
    skipped = 0
    other_preds = 0

    for entry in tqdm(entries, desc="Evaluating HF-Dusha"):
        audio_path = entry["audio_path"]
        label_id = entry["label_id"]

        waveform = load_audio(audio_path)
        if waveform is None:
            skipped += 1
            continue

        inputs = feature_extractor(
            waveform.numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            max_length=MAX_AUDIO_SAMPLES,
            truncation=True,
        )
        input_values = inputs["input_values"].to(DEVICE)

        with torch.no_grad():
            logits = model(input_values).logits
        hf_pred = logits.argmax(dim=-1).item()

        our_pred = hf_pred_to_our_label(hf_pred)
        if our_pred is None:
            logits_4class = logits.clone()
            logits_4class[:, 4] = float("-inf")
            hf_pred_fallback = logits_4class.argmax(dim=-1).item()
            our_pred = hf_pred_to_our_label(hf_pred_fallback)
            other_preds += 1

        all_labels.append(label_id)
        all_preds.append(our_pred)

    print(f"\nEvaluated: {len(all_labels)} | Skipped: {skipped} | 'other' predictions: {other_preds}")

    metrics = compute_metrics(all_labels, all_preds, DUSHA_EMOTIONS)

    print("\n" + "=" * 50)
    print("HF-DUSHA BASELINE TEST RESULTS")
    print(f"(xbgoose/hubert-large, {n_params:,} params)")
    print("=" * 50)
    print(format_metrics(metrics))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "hf_dusha_pretrained_test_dusha.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": "hf_dusha_pretrained",
            "hf_model": HUBERT_DUSHA,
            "params": n_params,
            "test_manifest": args.test_manifest,
            "test_samples": len(all_labels),
            "skipped": skipped,
            "other_predictions": other_preds,
            "metrics": metrics,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
