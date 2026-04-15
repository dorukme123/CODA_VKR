"""
add source_id (speaker ID) to existing manifest.jsonl files.

Usage:
    python -m src.data.patch_manifest_speaker_ids
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DUSHA_CROWD_TRAIN, DUSHA_CROWD_TEST, PREPROCESSED_DIR


def build_speaker_map(tsv_path: Path) -> dict[str, str]:
    df = pd.read_csv(tsv_path, sep="\t", usecols=["hash_id", "source_id"])
    # Each hash_id can appear multiple times (multi-annotator), source_id is the same
    mapping = df.drop_duplicates("hash_id").set_index("hash_id")["source_id"].to_dict()
    return {k: str(v) for k, v in mapping.items()}


def patch_manifest(manifest_path: Path, speaker_map: dict[str, str]):
    if not manifest_path.exists():
        print(f"  SKIP (not found): {manifest_path}")
        return

    backup = manifest_path.with_suffix(".jsonl.bak")
    shutil.copy2(manifest_path, backup)

    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            uid = entry["id"]
            if uid in speaker_map:
                entry["source_id"] = speaker_map[uid]
            entries.append(entry)

    matched = sum(1 for e in entries if "source_id" in e)
    print(f"  {manifest_path.name}: {matched}/{len(entries)} entries matched with speaker IDs")

    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  Backup saved to {backup.name}")


def main():
    subsets = [
        ("crowd_train", DUSHA_CROWD_TRAIN),
        ("crowd_test", DUSHA_CROWD_TEST),
    ]

    for subset_name, subset_dir in subsets:
        tsv_path = subset_dir / f"raw_{subset_name}.tsv"
        manifest_path = PREPROCESSED_DIR / "dusha" / subset_name / "manifest.jsonl"

        print(f"\n--- {subset_name} ---")
        if not tsv_path.exists():
            print(f"  SKIP (no TSV): {tsv_path}")
            continue

        speaker_map = build_speaker_map(tsv_path)
        print(f"  Built speaker map: {len(speaker_map)} utterances → speakers")
        patch_manifest(manifest_path, speaker_map)

    print("\nDone. Run training scripts — they will now use speaker-independent splits.")


if __name__ == "__main__":
    main()
