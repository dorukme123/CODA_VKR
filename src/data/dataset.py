"""
Usage:
    dataset = CODADataset("manifest.jsonl")
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Subset

log = logging.getLogger(__name__)


class CODADataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        max_samples: int | None = None,
        require_text: bool = False,
        require_hubert: bool = True,
        stratified: bool = False,
        num_classes: int = 4,
        seed: int = 42,
        cache_in_ram: bool = False,
        cache_max_frames: int = 250,
    ):
        self.entries = []

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if require_text and not entry.get("text"):
                    continue
                if require_hubert and not entry.get("hubert_path"):
                    continue
                self.entries.append(entry)

        if stratified and max_samples:
            self.entries = self._stratified_sample(max_samples, num_classes, seed)
        elif max_samples:
            self.entries = self.entries[:max_samples]

    def _stratified_sample(
        self, n: int, num_classes: int, seed: int
    ) -> list[dict]:
        rng = np.random.RandomState(seed)
        per_class = n // num_classes

        by_class: dict[int, list[dict]] = {}
        for entry in self.entries:
            lid = entry["label_id"]
            by_class.setdefault(lid, []).append(entry)

        sampled = []
        for lid in sorted(by_class.keys()):
            items = by_class[lid]
            rng.shuffle(items)
            sampled.extend(items[:per_class])

        rng.shuffle(sampled)
        return sampled

    def get_labels(self) -> list[int]:
        return [e["label_id"] for e in self.entries]

    def get_speaker_ids(self) -> list[str]:
        return [e.get("source_id", f"unknown_{i}") for i, e in enumerate(self.entries)]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]

        # HuBERT embeddings
        hubert_path = entry.get("hubert_path")
        if hubert_path and Path(hubert_path).exists():
            hubert = torch.load(hubert_path, weights_only=True).float()
        else:
            hubert = torch.zeros(1, 768)

        # Prosodic features
        prosodic = entry.get("prosodic")
        if prosodic is not None:
            prosodic = torch.tensor(prosodic, dtype=torch.float32)
        else:
            prosodic = torch.zeros(5, dtype=torch.float32)

        # Text
        text = entry.get("text") or ""

        # Pre-computed ruBERT embeddings
        rubert_path = entry.get("rubert_path")
        if rubert_path and Path(rubert_path).exists():
            data = torch.load(rubert_path, weights_only=True)
            rubert = data["hidden_states"].float()
        else:
            rubert = None

        # Label
        label = entry["label_id"]

        return {
            "hubert": hubert,
            "prosodic": prosodic,
            "text": text,
            "rubert": rubert,
            "label": label,
            "uid": entry["id"],
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        huberts = [item["hubert"] for item in batch]
        lengths = [h.shape[0] for h in huberts]

        hubert_padded = pad_sequence(huberts, batch_first=True)

        max_len = hubert_padded.shape[1]
        hubert_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        for i, length in enumerate(lengths):
            hubert_mask[i, :length] = True

        # Pre-computed ruBERT embeddings (if all samples have them)
        has_rubert = all(item["rubert"] is not None for item in batch)
        if has_rubert:
            ruberts = [item["rubert"] for item in batch]
            rubert_lengths = [r.shape[0] for r in ruberts]
            rubert_padded = pad_sequence(ruberts, batch_first=True)
            rubert_max = rubert_padded.shape[1]
            rubert_mask = torch.zeros(len(batch), rubert_max, dtype=torch.bool)
            for i, length in enumerate(rubert_lengths):
                rubert_mask[i, :length] = True
        else:
            rubert_padded = None
            rubert_mask = None

        return {
            "hubert": hubert_padded,
            "hubert_mask": hubert_mask,
            "prosodic": torch.stack([item["prosodic"] for item in batch]),
            "text": [item["text"] for item in batch],
            "rubert": rubert_padded,
            "rubert_mask": rubert_mask,
            "label": torch.tensor(
                [item["label"] for item in batch], dtype=torch.long
            ),
        }


def speaker_independent_split(
    dataset: "CODADataset",
    test_size: float = 0.1,
    seed: int = 42,
    dry_run: bool = False,
) -> tuple[Subset, Subset]:
    
    speaker_ids = np.array(dataset.get_speaker_ids())
    has_speakers = not any(s.startswith("unknown_") for s in speaker_ids[:10])

    if has_speakers and not dry_run:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, val_idx = next(gss.split(np.arange(len(dataset)), groups=speaker_ids))
        train_ds = Subset(dataset, train_idx.tolist())
        val_ds = Subset(dataset, val_idx.tolist())

        train_speakers = set(speaker_ids[train_idx])
        val_speakers = set(speaker_ids[val_idx])
        overlap = train_speakers & val_speakers
        log.info(
            "Speaker-independent split: %d train speakers, %d val speakers, %d overlap",
            len(train_speakers), len(val_speakers), len(overlap),
        )
        assert len(overlap) == 0, f"Speaker leakage: {len(overlap)} shared speakers!"
    else:
        if not has_speakers:
            log.warning("No source_id in manifest — falling back to index split. "
                        "Run: python -m src.data.patch_manifest_speaker_ids")
        n_val = max(int(len(dataset) * test_size), 1)
        n_train = len(dataset) - n_val
        train_ds = Subset(dataset, list(range(n_train)))
        val_ds = Subset(dataset, list(range(n_train, n_train + n_val)))

    return train_ds, val_ds
