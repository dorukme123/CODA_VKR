"""
Dusha Dataset Loader
"""

from collections import Counter
from pathlib import Path
from typing import Literal

import pandas as pd

from src.config import (
    DUSHA_CROWD_TRAIN,
    DUSHA_CROWD_TEST,
    DUSHA_PODCAST_TRAIN,
    DUSHA_PODCAST_TEST,
    DUSHA_EMOTIONS,
    DUSHA_LABEL2ID,
    DUSHA_DROP_LABELS,
)

Subset = Literal["crowd_train", "crowd_test", "podcast_train", "podcast_test"]

_SUBSET_DIRS = {
    "crowd_train": DUSHA_CROWD_TRAIN,
    "crowd_test": DUSHA_CROWD_TEST,
    "podcast_train": DUSHA_PODCAST_TRAIN,
    "podcast_test": DUSHA_PODCAST_TEST,
}


def _majority_vote(emotions: list[str]) -> str | None:
    filtered = [e for e in emotions if e not in DUSHA_DROP_LABELS]
    if not filtered:
        return None
    counts = Counter(filtered)
    max_count = max(counts.values())
    # Break ties alphabetically for reproducibility
    candidates = sorted(e for e, c in counts.items() if c == max_count)
    return candidates[0]


def load_dusha_raw(subset: Subset) -> pd.DataFrame:
    """
    Columns: hash_id, audio_path, duration, annotator_emo, golden_emo,
             annotator_id, speaker_text, speaker_emo, source_id
    """
    subset_dir = _SUBSET_DIRS[subset]
    tsv_path = subset_dir / f"raw_{subset}.tsv"

    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")

    return pd.read_csv(tsv_path, sep="\t")


def load_dusha(
    subset: Subset,
    require_audio: bool = True,
    require_text: bool = False,
    min_annotators: int = 1,
    min_agreement: float = 0.0,
) -> pd.DataFrame:
    """
    Returns:
        DataFrame with columns:
            hash_id, audio_path (absolute), duration, emotion, label_id,
            speaker_text, n_annotators, agreement
    """
    raw = load_dusha_raw(subset)
    subset_dir = _SUBSET_DIRS[subset]

    # Group by hash_id and aggregate
    grouped = raw.groupby("hash_id")
    records = []

    for hash_id, group in grouped:
        emotions = group["annotator_emo"].dropna().tolist()
        if len(emotions) < min_annotators:
            continue

        emotion = _majority_vote(emotions)
        if emotion is None:
            continue

        counts = Counter(e for e in emotions if e not in DUSHA_DROP_LABELS)
        agreement = counts[emotion] / len(emotions) if emotions else 0.0
        if agreement < min_agreement:
            continue

        audio_rel = group["audio_path"].iloc[0]

        audio_abs = subset_dir / audio_rel

        texts = group["speaker_text"].dropna().unique()
        speaker_text = texts[0] if len(texts) > 0 else None

        duration = group["duration"].iloc[0]

        # Speaker (source) ID — needed for speaker-independent splits
        source_id = group["source_id"].iloc[0] if "source_id" in group.columns else None

        records.append({
            "hash_id": hash_id,
            "audio_path": str(audio_abs),
            "duration": duration,
            "emotion": emotion,
            "label_id": DUSHA_LABEL2ID[emotion],
            "speaker_text": speaker_text,
            "source_id": source_id,
            "n_annotators": len(emotions),
            "agreement": agreement,
        })

    df = pd.DataFrame(records)

    if require_audio:
        df = df[df["audio_path"].apply(lambda p: Path(p).exists())]

    if require_text:
        df = df[df["speaker_text"].notna()]

    df = df.reset_index(drop=True)
    return df


def load_dusha_splits(
    require_audio: bool = True,
    require_text: bool = False,
    min_agreement: float = 0.0,
) -> dict[str, pd.DataFrame]:
    splits = {}
    for subset in ["crowd_train", "crowd_test", "podcast_train", "podcast_test"]:
        splits[subset] = load_dusha(
            subset,
            require_audio=require_audio,
            require_text=require_text,
            min_agreement=min_agreement,
        )
    return splits


def get_dusha_stats(df: pd.DataFrame) -> dict:
    stats = {
        "total": len(df),
        "emotions": df["emotion"].value_counts().to_dict(),
        "has_text": df["speaker_text"].notna().sum(),
        "duration_mean": df["duration"].mean(),
        "duration_std": df["duration"].std(),
        "duration_total_hours": df["duration"].sum() / 3600,
        "agreement_mean": df["agreement"].mean(),
        "annotators_mean": df["n_annotators"].mean(),
    }
    return stats
