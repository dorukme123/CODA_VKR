"""
Usage:
    python -m src.training.eval_dissonance \
        --checkpoint results/checkpoints/coda_full/epoch_XXX.pt \
        --train-manifest results/preprocessed/dusha/crowd_train/manifest.jsonl \
        --test-manifest results/preprocessed/dusha/crowd_test/manifest.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DEVICE,
    METRICS_DIR,
    RUBERT_BASE,
    ModelConfig,
    seed_everything,
)
from src.data.dataset import CODADataset
from src.models.coda_pipeline import CODAPipeline
from src.models.dissonance_detector import DissonanceDetector


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate dissonance detection")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--train-manifest", type=str, required=True,
                   help="Training manifest — used to fit Isolation Forest")
    p.add_argument("--test-manifest", type=str, required=True,
                   help="Test manifest — used for evaluation")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-train-samples", type=int, default=20000,
                   help="Max training samples for IF fitting (saves time/memory)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--contamination", type=float, default=0.05)
    return p.parse_args()


def build_model(device):
    mc = ModelConfig()
    model = CODAPipeline(
        variant="full",
        hubert_dim=mc.hubert_dim,
        bert_model_name=RUBERT_BASE,
        prosodic_dim=mc.prosodic_dim,
        cross_attn_dim=mc.cross_attn_dim,
        cross_attn_heads=mc.cross_attn_heads,
        cross_attn_dropout=mc.cross_attn_dropout,
        context_hidden=mc.context_hidden,
        context_layers=mc.context_layers,
        context_dropout=mc.context_dropout,
        classifier_hidden=mc.classifier_hidden,
        classifier_dropout=mc.classifier_dropout,
        num_classes=mc.num_classes,
        freeze_bert=True,
        dissonance_contamination=mc.dissonance_contamination,
    )
    return model.to(device)


def _extract_features(output, mask, rubert_mask):
    """
    Returns (B, 4) features:
        [H(a2t), H(t2a), prediction_entropy, max_softmax_prob]
    """
    a2t_w = output["a2t_weights"]  # (B, T_a, T_t)
    t2a_w = output["t2a_weights"]  # (B, T_t, T_a)

    # Attention entropy
    h_a2t = DissonanceDetector.compute_attention_entropy(a2t_w, mask)
    h_t2a = DissonanceDetector.compute_attention_entropy(t2a_w, rubert_mask)

    # Prediction confidence features
    logits = output["logits"]
    probs = torch.softmax(logits, dim=-1)
    pred_entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)  # (B,)
    max_prob = probs.max(dim=-1).values  # (B,)

    features = torch.stack([h_a2t, h_t2a, pred_entropy, max_prob], dim=-1)  # (B, 4)
    return features


@torch.no_grad()
def collect_features(model, loader, device, desc="Collecting features"):
    model.eval()
    all_features = []
    all_scores = []
    all_labels = []

    for batch in tqdm(loader, desc=desc):
        hubert = batch["hubert"].to(device)
        mask = batch["hubert_mask"].to(device)
        prosodic = batch["prosodic"].to(device)
        labels = batch["label"]
        texts = batch["text"]
        rubert_emb = batch["rubert"].to(device) if batch["rubert"] is not None else None
        rubert_mask = batch["rubert_mask"].to(device) if batch["rubert_mask"] is not None else None

        output = model(
            hubert=hubert, hubert_mask=mask, prosodic=prosodic,
            texts=texts, rubert_emb=rubert_emb, rubert_mask=rubert_mask,
        )

        features = _extract_features(output, mask, rubert_mask)
        all_features.append(features.cpu().numpy())
        all_scores.append(output["dissonance_score"].cpu().numpy())
        all_labels.extend(labels.tolist())

    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_scores, axis=0),
        np.array(all_labels),
    )


class MismatchedDataset(torch.utils.data.Dataset):
    """
    Wraps a CODADataset to create emotion-aware mismatches.
    """

    def __init__(self, base_dataset: CODADataset, seed: int = 42):
        self.base = base_dataset
        rng = np.random.RandomState(seed)

        # Group indices by emotion label
        by_label: dict[int, list[int]] = {}
        for i, entry in enumerate(base_dataset.entries):
            lid = entry["label_id"]
            by_label.setdefault(lid, []).append(i)

        # For each sample, pick a random sample with a DIFFERENT label
        all_labels = [e["label_id"] for e in base_dataset.entries]
        other_labels = sorted(by_label.keys())
        self.text_donor_idx = []

        for i, my_label in enumerate(all_labels):
            candidates = [l for l in other_labels if l != my_label]
            donor_label = rng.choice(candidates)
            donor_idx = rng.choice(by_label[donor_label])
            self.text_donor_idx.append(donor_idx)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # Audio + prosodic from original sample
        item = self.base[idx]
        # Text from a different-emotion donor
        donor = self.base[self.text_donor_idx[idx]]
        item["text"] = donor["text"]
        item["rubert"] = donor["rubert"]
        return item


@torch.no_grad()
def collect_mismatched_features(model, dataset, device, batch_size=32, num_workers=4):
    """
    Each sample's audio is paired with text from a different emotion class.
    """
    model.eval()
    mismatched_ds = MismatchedDataset(dataset)
    loader = DataLoader(
        mismatched_ds, batch_size=batch_size, shuffle=False,
        collate_fn=CODADataset.collate_fn, num_workers=num_workers,
        pin_memory=True,
    )

    all_features = []
    all_scores = []

    for batch in tqdm(loader, desc="Collecting mismatched features"):
        hubert = batch["hubert"].to(device)
        mask = batch["hubert_mask"].to(device)
        prosodic = batch["prosodic"].to(device)
        texts = batch["text"]
        rubert_emb = batch["rubert"].to(device) if batch["rubert"] is not None else None
        rubert_mask = batch["rubert_mask"].to(device) if batch["rubert_mask"] is not None else None

        output = model(
            hubert=hubert, hubert_mask=mask, prosodic=prosodic,
            texts=texts, rubert_emb=rubert_emb, rubert_mask=rubert_mask,
        )

        features = _extract_features(output, mask, rubert_mask)
        all_features.append(features.cpu().numpy())
        all_scores.append(output["dissonance_score"].cpu().numpy())

    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_scores, axis=0),
    )


def evaluate_binary(y_true, y_pred, method_name):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1
    )
    print(f"\n{'=' * 50}")
    print(f"Dissonance Detection — {method_name}")
    print(f"{'=' * 50}")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"\nDetailed report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["matched", "dissonant"],
    ))
    return {"precision": p, "recall": r, "f1": f1}


def main():
    args = parse_args()
    seed_everything(args.seed)

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model = build_model(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded weights from epoch {ckpt.get('epoch', '?')}")

    # --- Load datasets ---
    print(f"\nLoading training data for IF fitting: {args.train_manifest}")
    train_dataset = CODADataset(
        args.train_manifest,
        max_samples=args.max_train_samples,
        require_hubert=True,
    )
    print(f"Training samples (for IF): {len(train_dataset)}")

    print(f"Loading test data: {args.test_manifest}")
    test_dataset = CODADataset(args.test_manifest, require_hubert=True)
    print(f"Test samples: {len(test_dataset)}")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=CODADataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    feat_names = ["H(a2t)", "H(t2a)", "pred_entropy", "max_prob"]

    print("\n--- Step 1: Collecting features from training set ---")
    train_feats, train_scores, train_labels = collect_features(
        model, train_loader, DEVICE, desc="Train features"
    )
    print(f"Training feature stats:")
    for i, name in enumerate(feat_names):
        print(f"  {name}: mean={train_feats[:, i].mean():.4f}, std={train_feats[:, i].std():.4f}")
    print(f"  MLP score: mean={train_scores.mean():.4f}, std={train_scores.std():.4f}")

    print("\n--- Step 2: Fitting Isolation Forest ---")
    model.dissonance_head.contamination = args.contamination
    model.dissonance_head.fit_isolation_forest(train_feats)
    print(f"IF fitted on {len(feat_names)} features with contamination={args.contamination}")

    print("\n--- Step 3: Collecting matched test features ---")
    test_matched_feats, test_matched_scores, test_labels = collect_features(
        model, test_loader, DEVICE, desc="Matched test"
    )
    n_matched = len(test_matched_scores)
    print(f"Matched test samples: {n_matched}")

    print("\n--- Step 4: Collecting emotion-aware mismatched test features ---")
    test_mismatched_feats, test_mismatched_scores = collect_mismatched_features(
        model, test_dataset, DEVICE,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    n_mismatched = len(test_mismatched_scores)
    print(f"Mismatched test samples: {n_mismatched}")

    print("\n--- Step 5: Evaluation ---")

    all_feats = np.concatenate([test_matched_feats, test_mismatched_feats], axis=0)
    all_mlp_scores = np.concatenate([test_matched_scores, test_mismatched_scores], axis=0)
    y_true = np.array([0] * n_matched + [1] * n_mismatched)

    if_preds_raw = model.dissonance_head.predict_anomaly(all_feats)
    # IF returns -1 for anomaly, 1 for normal → convert to 0/1 (1 = dissonant)
    if_preds = (if_preds_raw == -1).astype(int)
    if_metrics = evaluate_binary(y_true, if_preds, "Isolation Forest (4 features)")

    print("\n--- MLP Dissonance Score Analysis ---")
    print(f"  Matched scores:    mean={test_matched_scores.mean():.4f}, std={test_matched_scores.std():.4f}")
    print(f"  Mismatched scores: mean={test_mismatched_scores.mean():.4f}, std={test_mismatched_scores.std():.4f}")

    best_f1 = 0
    best_threshold = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        mlp_preds = (all_mlp_scores >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, mlp_preds, average="binary", pos_label=1, zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"  Best threshold: {best_threshold:.2f}")
    mlp_preds = (all_mlp_scores >= best_threshold).astype(int)
    mlp_metrics = evaluate_binary(y_true, mlp_preds, f"MLP Head (threshold={best_threshold:.2f})")

    print("\n--- Prediction Confidence Analysis ---")
    matched_pred_ent = test_matched_feats[:, 2]
    mismatch_pred_ent = test_mismatched_feats[:, 2]
    print(f"  Matched pred_entropy:    mean={matched_pred_ent.mean():.4f}, std={matched_pred_ent.std():.4f}")
    print(f"  Mismatched pred_entropy: mean={mismatch_pred_ent.mean():.4f}, std={mismatch_pred_ent.std():.4f}")

    matched_max_prob = test_matched_feats[:, 3]
    mismatch_max_prob = test_mismatched_feats[:, 3]
    print(f"  Matched max_prob:        mean={matched_max_prob.mean():.4f}, std={matched_max_prob.std():.4f}")
    print(f"  Mismatched max_prob:     mean={mismatch_max_prob.mean():.4f}, std={mismatch_max_prob.std():.4f}")

    all_pred_ent = np.concatenate([matched_pred_ent, mismatch_pred_ent])
    best_pe_f1 = 0
    best_pe_thresh = 0.5
    for t in np.linspace(all_pred_ent.min(), all_pred_ent.max(), 200):
        preds = (all_pred_ent >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", pos_label=1, zero_division=0
        )
        if f1 > best_pe_f1:
            best_pe_f1 = f1
            best_pe_thresh = t

    pe_preds = (all_pred_ent >= best_pe_thresh).astype(int)
    pe_metrics = evaluate_binary(y_true, pe_preds, f"Prediction Entropy (threshold={best_pe_thresh:.4f})")

    print("\n--- Feature Summary (matched vs mismatched) ---")
    header = f"{'':>20s}" + "".join(f"  {n:>14s}" for n in feat_names)
    print(header)
    row_m = f"{'Matched mean':>20s}" + "".join(f"  {test_matched_feats[:, i].mean():>14.4f}" for i in range(4))
    row_ms = f"{'Matched std':>20s}" + "".join(f"  {test_matched_feats[:, i].std():>14.4f}" for i in range(4))
    row_mm = f"{'Mismatched mean':>20s}" + "".join(f"  {test_mismatched_feats[:, i].mean():>14.4f}" for i in range(4))
    row_mms = f"{'Mismatched std':>20s}" + "".join(f"  {test_mismatched_feats[:, i].std():>14.4f}" for i in range(4))
    print(row_m)
    print(row_ms)
    print(row_mm)
    print(row_mms)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "dissonance_detection.json"

    def _stats(arr, col):
        return {"mean": float(arr[:, col].mean()), "std": float(arr[:, col].std())}

    results = {
        "model": "coda_full",
        "checkpoint": args.checkpoint,
        "features": feat_names,
        "train_samples_for_if": len(train_dataset),
        "test_matched": n_matched,
        "test_mismatched": n_mismatched,
        "isolation_forest": {
            "contamination": args.contamination,
            **{k: float(v) for k, v in if_metrics.items()},
        },
        "mlp_head": {
            "best_threshold": float(best_threshold),
            **{k: float(v) for k, v in mlp_metrics.items()},
        },
        "prediction_entropy": {
            "best_threshold": float(best_pe_thresh),
            **{k: float(v) for k, v in pe_metrics.items()},
        },
        "feature_stats": {
            name: {
                "train": _stats(train_feats, i),
                "matched": _stats(test_matched_feats, i),
                "mismatched": _stats(test_mismatched_feats, i),
            }
            for i, name in enumerate(feat_names)
        },
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
