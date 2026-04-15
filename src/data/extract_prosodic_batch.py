"""
Usage:
    python -m src.data.extract_prosodic_batch --manifest C:\\Users\\Meric\\Desktop\\VKR\\Code\\results\\preprocessed\\dusha\\crowd_train\\manifest.jsonl --workers 8
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_DIR = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_CODE_DIR))

from src.data.prosodic_features import extract_prosodic_features  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prosodic_batch")


def _extract_one(audio_path: str) -> list[float]:
    try:
        pf = extract_prosodic_features(audio_path)
        return pf.to_array().tolist()
    except Exception as e:
        return None  # Return None so we can count failures


def main():
    from multiprocessing import Pool

    parser = argparse.ArgumentParser(description="Parallel prosodic extraction")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of CPU processes (default: 8)")
    parser.add_argument("--chunksize", type=int, default=100,
                        help="Files per worker batch (default: 100)")
    parser.add_argument("--reextract-zeros", action="store_true", default=True,
                        help="Re-extract entries with all-zero prosodic (from failed runs)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        return

    # Read all entries
    log.info("Reading manifest: %s", manifest_path)
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    log.info("Total entries: %d", len(entries))

    _ALL_ZEROS = [0.0, 0.0, 0.0, 0.0, 0.0]
    needs_prosodic = []
    needs_indices = []
    for i, entry in enumerate(entries):
        p = entry.get("prosodic")
        needs_it = (p is None) or (args.reextract_zeros and p == _ALL_ZEROS)
        if needs_it:
            needs_prosodic.append(entry["audio_path"])
            needs_indices.append(i)

    if not needs_prosodic:
        log.info("All entries already have prosodic features. Nothing to do.")
        return

    log.info("Entries needing prosodic: %d / %d", len(needs_prosodic), len(entries))
    log.info("Using %d worker processes, chunksize=%d", args.workers, args.chunksize)

    log.info("Sanity check: extracting one file...")
    test_result = _extract_one(needs_prosodic[0])
    if test_result is None:
        log.error("Sanity check FAILED — prosodic extraction returned None for: %s", needs_prosodic[0])
        log.error("Check that the audio file exists and parselmouth is installed.")
        return
    log.info("Sanity check OK: %s", test_result)

    # Extract in parallel
    start = time.time()
    n_failed = 0
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap(_extract_one, needs_prosodic, chunksize=args.chunksize),
            total=len(needs_prosodic),
            desc="Prosodic extraction",
        ))

    # Update entries, count failures
    for idx, prosodic in zip(needs_indices, results):
        if prosodic is None:
            entries[idx]["prosodic"] = [0.0] * 5
            n_failed += 1
        else:
            entries[idx]["prosodic"] = prosodic

    elapsed = time.time() - start

    # Rewrite manifest
    log.info("Rewriting manifest...")
    tmp_path = manifest_path.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    tmp_path.replace(manifest_path)

    rate = len(needs_prosodic) / max(elapsed, 1)
    log.info("Done! %d files in %.1f min (%.0f files/sec) | %d failed",
             len(needs_prosodic), elapsed / 60, rate, n_failed)


if __name__ == "__main__":
    main()
