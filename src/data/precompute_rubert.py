"""
Pre-compute ruBERT embeddings for all samples in a manifest.

Usage:
    # Dusha crowd_train (~180K samples, ~20-30 min on RTX 4080)
    python -m src.data.precompute_rubert --manifest results/preprocessed/dusha/crowd_train/manifest.jsonl

    # Dusha crowd_test
    python -m src.data.precompute_rubert --manifest results/preprocessed/dusha/crowd_test/manifest.jsonl

    # Quick verification
    python -m src.data.precompute_rubert --manifest results/preprocessed/dusha/crowd_train/manifest.jsonl --limit 100
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DEVICE, RUBERT_BASE, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("precompute_rubert")


def parse_args():
    p = argparse.ArgumentParser(description="Pre-compute ruBERT embeddings")
    p.add_argument("--manifest", type=str, required=True,
                   help="Path to manifest.jsonl from preprocessing")
    p.add_argument("--model-name", type=str, default=RUBERT_BASE,
                   help=f"HuggingFace model name (default: {RUBERT_BASE})")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for BERT inference (default: 64)")
    p.add_argument("--max-length", type=int, default=128,
                   help="Max token length (default: 128)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N samples (for verification)")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip samples that already have cached embeddings")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(42)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    # Output directory: same parent as manifest / rubert/
    out_dir = manifest_path.parent / "rubert"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    # Load manifest entries that have text
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("text"):
                entries.append(entry)

    if args.limit:
        entries = entries[:args.limit]
    log.info("Loaded %d entries with text", len(entries))

    # Filter out already-computed
    if args.skip_existing:
        before = len(entries)
        entries = [e for e in entries if not (out_dir / f"{e['id']}.pt").exists()]
        skipped = before - len(entries)
        if skipped > 0:
            log.info("Skipping %d already-computed samples", skipped)

    if not entries:
        log.info("Nothing to compute. All done!")
        # Update manifest with rubert_path pointers
        _update_manifest(manifest_path, out_dir)
        return

    log.info("Processing %d samples...", len(entries))

    # Load model
    log.info("Loading %s...", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    log.info("Model loaded on %s", DEVICE)

    # Process in batches
    start = time.time()
    processed = 0

    for batch_start in tqdm(range(0, len(entries), args.batch_size),
                            desc="ruBERT extraction"):
        batch_entries = entries[batch_start:batch_start + args.batch_size]
        texts = [e["text"] for e in batch_entries]
        uids = [e["id"] for e in batch_entries]

        tokens = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(DEVICE)
        attention_mask = tokens["attention_mask"].to(DEVICE)

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # (B, L_padded, 768)

        # Save per sample — trim padding, store as fp16
        for i, uid in enumerate(uids):
            mask_i = attention_mask[i].bool()
            actual_len = mask_i.sum().item()
            hs = hidden_states[i, :actual_len].cpu().half()  # (L_actual, 768) fp16

            save_path = out_dir / f"{uid}.pt"
            torch.save({"hidden_states": hs}, save_path)

        processed += len(batch_entries)

    elapsed = time.time() - start
    log.info("Extracted %d samples in %.1f min (%.0f samples/sec)",
             processed, elapsed / 60, processed / elapsed)

    # Storage stats
    total_bytes = sum(f.stat().st_size for f in out_dir.glob("*.pt"))
    log.info("Total storage: %.2f GB", total_bytes / 1e9)

    # Update manifest with rubert_path pointers
    _update_manifest(manifest_path, out_dir)


def _update_manifest(manifest_path: Path, out_dir: Path):
    log.info("Updating manifest with rubert_path pointers...")
    updated_lines = []
    n_updated = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            uid = entry["id"]
            rubert_file = out_dir / f"{uid}.pt"
            if rubert_file.exists():
                entry["rubert_path"] = str(rubert_file)
                n_updated += 1
            updated_lines.append(json.dumps(entry, ensure_ascii=False))

    with open(manifest_path, "w", encoding="utf-8") as f:
        for line in updated_lines:
            f.write(line + "\n")

    log.info("Updated %d entries with rubert_path", n_updated)


if __name__ == "__main__":
    main()
