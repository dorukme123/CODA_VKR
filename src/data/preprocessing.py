"""
Features extracted per utterance:
    1. HuBERT embeddings    → (T, 768) tensor  → .pt file (float16 by default)
    2. Prosodic features    → (5,) array        → stored in manifest
    3. Text + label info    →                   → stored in manifest

Usage:
    # Verification (500 samples)
    python -m src.data.preprocessing --dataset dusha --subset crowd_train --limit 500

    # Full crowd_train (~181K, ~6-8 hours on RTX 4080)
    python -m src.data.preprocessing --dataset dusha --subset crowd_train --batch-size 32 --prosodic-workers 6

    # Crowd test set
    python -m src.data.preprocessing --dataset dusha --subset crowd_test --batch-size 32 --prosodic-workers 6

    # IEMOCAP
    python -m src.data.preprocessing --dataset iemocap --sessions 1 2 3 4 5 --batch-size 32 --prosodic-workers 6
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    PREPROCESSED_DIR,
    SAMPLE_RATE,
    MAX_AUDIO_SAMPLES,
    HUBERT_BASE,
    DEVICE,
    seed_everything,
)
from src.data.dusha_loader import load_dusha
from src.data.iemocap_loader import load_iemocap
from src.data.prosodic_features import extract_prosodic_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess")


def parse_args():
    p = argparse.ArgumentParser(description="CODA Preprocessing PRE-STEP")
    p.add_argument("--dataset", choices=["dusha", "iemocap"], required=True)
    p.add_argument("--subset", type=str, default=None,
                   help="Dusha subset: crowd_train, crowd_test, podcast_train, podcast_test")
    p.add_argument("--sessions", type=int, nargs="+", default=None,
                   help="IEMOCAP sessions to process (e.g. 1 2 3 4 5)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N samples (for verification)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for HuBERT inference (default: 32)")
    p.add_argument("--prosodic-workers", type=int, default=4,
                   help="Parallel workers for prosodic feature extraction (default: 4)")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip utterances that already have cached features")
    p.add_argument("--skip-hubert", action="store_true",
                   help="Skip HuBERT extraction (if already done)")
    p.add_argument("--skip-prosodic", action="store_true",
                   help="Skip prosodic extraction (if already done)")
    p.add_argument("--fp16", action="store_true", default=True,
                   help="Save HuBERT features in float16 (halves disk usage)")
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    return p.parse_args()


def load_audio(audio_path: str) -> torch.Tensor | None:
    """Load audio file, resample to 16kHz mono, truncate to MAX_AUDIO_SAMPLES."""
    try:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        if waveform.shape[1] > MAX_AUDIO_SAMPLES:
            waveform = waveform[:, :MAX_AUDIO_SAMPLES]
        return waveform.squeeze(0)
    except Exception as e:
        log.debug("Failed to load %s: %s", audio_path, e)
        return None


@torch.no_grad()
def extract_hubert_batch(
    waveforms: list[torch.Tensor],
    model: HubertModel,
    feature_extractor: Wav2Vec2FeatureExtractor,
    fp16: bool = True,
) -> list[torch.Tensor]:
    """Returns list of (T_i, 768) tensors."""
    np_waveforms = [w.numpy() for w in waveforms]
    inputs = feature_extractor(
        np_waveforms, sampling_rate=SAMPLE_RATE,
        return_tensors="pt", padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=False)
    hidden = outputs.last_hidden_state  # (B, T, 768)

    results = []
    for i, w in enumerate(waveforms):
        out_len = w.shape[0] // 320
        out_len = min(out_len, hidden.shape[1])
        h = hidden[i, :out_len].cpu()
        if fp16:
            h = h.half()
        results.append(h)
    return results


def _extract_one_prosodic(audio_path: str) -> list[float]:
    try:
        pf = extract_prosodic_features(audio_path)
        return pf.to_array().tolist()
    except Exception:
        return [0.0] * 5


def preprocess_dataset(df: pd.DataFrame, output_dir: Path, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    hubert_dir = output_dir / "hubert"
    hubert_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    # Load HuBERT
    if not args.skip_hubert:
        log.info("Loading HuBERT model: %s", HUBERT_BASE)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_BASE)
        model = HubertModel.from_pretrained(HUBERT_BASE).to(DEVICE).eval()
        log.info("HuBERT loaded on %s (fp16 save: %s)", DEVICE, args.fp16)
    else:
        model = feature_extractor = None

    # Prosodic thread pool
    prosodic_pool = None
    if not args.skip_prosodic and args.prosodic_workers > 1:
        prosodic_pool = ThreadPoolExecutor(max_workers=args.prosodic_workers)
        log.info("Prosodic extraction: %d parallel workers", args.prosodic_workers)

    # Existing entries
    existing_ids = set()
    if args.skip_existing and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                existing_ids.add(entry["id"])
        log.info("Found %d existing entries — will skip them", len(existing_ids))

    if args.limit:
        df = df.head(args.limit)
        log.info("Limited to %d samples", len(df))

    id_col = "hash_id" if "hash_id" in df.columns else "utterance_id"
    text_col = "speaker_text" if "speaker_text" in df.columns else "transcript"

    manifest_file = open(manifest_path, "a", encoding="utf-8")

    batch_waveforms = []
    batch_rows = []
    total_processed = 0
    total_skipped = 0
    total_failed = 0
    start_time = time.time()

    pbar = tqdm(df.iterrows(), total=len(df), desc="Preprocessing")
    for idx, row in pbar:
        uid = row[id_col]
        if uid in existing_ids:
            total_skipped += 1
            continue

        audio_path = row["audio_path"]
        if pd.isna(audio_path) or not Path(audio_path).exists():
            total_failed += 1
            continue

        waveform = load_audio(audio_path)
        if waveform is None:
            total_failed += 1
            continue

        batch_waveforms.append(waveform)
        batch_rows.append(row)

        if len(batch_waveforms) >= args.batch_size:
            _process_batch(
                batch_waveforms, batch_rows, model, feature_extractor,
                hubert_dir, manifest_file, id_col, text_col, args, prosodic_pool,
            )
            total_processed += len(batch_waveforms)
            batch_waveforms.clear()
            batch_rows.clear()

            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            remaining = (len(df) - total_processed - total_skipped - total_failed)
            eta_min = remaining / rate / 60 if rate > 0 else 0
            pbar.set_postfix(
                ok=total_processed, skip=total_skipped, fail=total_failed,
                rate=f"{rate:.0f}/s", eta=f"{eta_min:.0f}m",
            )

    # Flush remaining
    if batch_waveforms:
        _process_batch(
            batch_waveforms, batch_rows, model, feature_extractor,
            hubert_dir, manifest_file, id_col, text_col, args, prosodic_pool,
        )
        total_processed += len(batch_waveforms)

    manifest_file.close()
    if prosodic_pool:
        prosodic_pool.shutdown()

    elapsed = time.time() - start_time
    log.info("=" * 50)
    log.info("Preprocessing complete")
    log.info("  Processed: %d", total_processed)
    log.info("  Skipped:   %d (already existed)", total_skipped)
    log.info("  Failed:    %d", total_failed)
    log.info("  Time:      %.1f min (%.1f samples/sec)",
             elapsed / 60, total_processed / max(elapsed, 1))
    log.info("  Output:    %s", output_dir)


def _process_batch(
    waveforms, rows, model, feature_extractor, hubert_dir, manifest_file,
    id_col, text_col, args, prosodic_pool,
):
    # --- HuBERT (GPU, batched) ---
    if model is not None and not args.skip_hubert:
        hubert_outputs = extract_hubert_batch(
            waveforms, model, feature_extractor, fp16=args.fp16
        )
    else:
        hubert_outputs = [None] * len(waveforms)

    # --- Prosodic (CPU, parallel) ---
    if not args.skip_prosodic:
        audio_paths = [row["audio_path"] for row in rows]
        if prosodic_pool is not None:
            prosodic_results = list(prosodic_pool.map(_extract_one_prosodic, audio_paths))
        else:
            prosodic_results = [_extract_one_prosodic(p) for p in audio_paths]
    else:
        prosodic_results = [None] * len(rows)

    # --- Write manifest ---
    for i, row in enumerate(rows):
        uid = row[id_col]

        if hubert_outputs[i] is not None:
            pt_path = hubert_dir / f"{uid}.pt"
            torch.save(hubert_outputs[i], pt_path)

        entry = {
            "id": uid,
            "audio_path": row["audio_path"],
            "duration": float(row["duration"]),
            "emotion": row.get("emotion"),
            "label_id": int(row.get("label_id", -1)),
            "text": row.get(text_col) if pd.notna(row.get(text_col)) else None,
            "prosodic": prosodic_results[i],
            "hubert_path": str(hubert_dir / f"{uid}.pt") if hubert_outputs[i] is not None else None,
        }

        if "valence" in row.index:
            entry["valence"] = float(row["valence"])
            entry["arousal"] = float(row["arousal"])
            entry["dominance"] = float(row["dominance"])
        if "agreement" in row.index:
            entry["agreement"] = float(row["agreement"])
            entry["n_annotators"] = int(row["n_annotators"])
        if "source_id" in row.index and pd.notna(row.get("source_id")):
            entry["source_id"] = str(row["source_id"])

        manifest_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        manifest_file.flush()


def main():
    args = parse_args()
    seed_everything()

    if args.dataset == "dusha":
        if args.subset is None:
            log.error("--subset required for dusha")
            sys.exit(1)
        log.info("Loading Dusha %s...", args.subset)
        df = load_dusha(args.subset, require_audio=True)
        log.info("Loaded %d utterances", len(df))
        output_dir = PREPROCESSED_DIR / "dusha" / args.subset
        preprocess_dataset(df, output_dir, args)

    elif args.dataset == "iemocap":
        sessions = args.sessions or [1, 2, 3, 4, 5]
        log.info("Loading IEMOCAP sessions %s...", sessions)
        df = load_iemocap(sessions, require_audio=True)
        log.info("Loaded %d utterances", len(df))
        output_dir = PREPROCESSED_DIR / "iemocap"
        preprocess_dataset(df, output_dir, args)


if __name__ == "__main__":
    main()
