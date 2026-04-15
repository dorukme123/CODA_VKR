"""
Usage:
    python -m src.training.train_hf_baseline --dry-run
    python -m src.training.train_hf_baseline --max-samples 8000
    python -m src.training.train_hf_baseline  # full dataset
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    CHECKPOINTS_DIR,
    DEVICE,
    DUSHA_EMOTIONS,
    LOGS_DIR,
    METRICS_DIR,
    PREPROCESSED_DIR,
    ModelConfig,
    TrainConfig,
    seed_everything,
)
from src.data.dataset import CODADataset, speaker_independent_split
from src.models.baselines import AudioOnlyBaseline
from src.training.losses import WeightedCELoss, compute_class_weights
from src.utils.logger import setup_logger
from src.utils.metrics import compute_metrics, format_metrics

MODEL_NAME = "hf_audio_baseline"


def parse_args():
    p = argparse.ArgumentParser(description=f"Train {MODEL_NAME}")
    p.add_argument("--dry-run", action="store_true",
                   help="Quick check: 4 samples, 2 epochs")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit training data (None = all available)")
    p.add_argument("--stratified", action="store_true", default=True,
                   help="Balance classes when using --max-samples")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override max epochs")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch size")
    p.add_argument("--lr", type=float, default=None,
                   help="Override learning rate")
    p.add_argument("--no-prosodic", action="store_true",
                   help="Disable prosodic features")
    p.add_argument("--cache-in-ram", action="store_true",
                   help="Pre-load all HuBERT features into RAM (~14 GB)")
    p.add_argument("--weight-method", type=str, default="sqrt_inverse",
                   choices=["inverse", "sqrt_inverse", "effective", "none"],
                   help="Class weighting method (default: sqrt_inverse)")
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader workers (default: from config)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, device, epoch, log, verbose,
):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch_idx, batch in enumerate(pbar):
        hubert = batch["hubert"].to(device)
        mask = batch["hubert_mask"].to(device)
        prosodic = batch["prosodic"].to(device)
        labels = batch["label"].to(device)

        logits = model(hubert=hubert, hubert_mask=mask, prosodic=prosodic)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if verbose and (batch_idx + 1) % 50 == 0:
            grad_norm = sum(
                p.grad.norm().item() for p in model.parameters() if p.grad is not None
            )
            lr_now = optimizer.param_groups[0]["lr"]
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1e9
                log.debug(
                    "  batch %d | loss %.4f | grad_norm %.4f | lr %.2e | GPU %.2f GB",
                    batch_idx + 1, loss.item(), grad_norm, lr_now, mem,
                )

    avg_loss = total_loss / max(len(loader), 1)
    metrics = compute_metrics(all_labels, all_preds, DUSHA_EMOTIONS)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        hubert = batch["hubert"].to(device)
        mask = batch["hubert_mask"].to(device)
        prosodic = batch["prosodic"].to(device)
        labels = batch["label"].to(device)

        logits = model(hubert=hubert, hubert_mask=mask, prosodic=prosodic)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    metrics = compute_metrics(all_labels, all_preds, DUSHA_EMOTIONS)
    return avg_loss, metrics


def save_checkpoint(model, optimizer, epoch, metrics, config, ckpt_dir):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"epoch_{epoch:03d}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }, path)
    return path


def cleanup_checkpoints(ckpt_dir, keep_top_k=3):
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    if len(ckpts) <= keep_top_k:
        return

    # Load metrics and sort
    scored = []
    for p in ckpts:
        data = torch.load(p, weights_only=False, map_location="cpu")
        ua = data.get("metrics", {}).get("unweighted_accuracy", 0)
        scored.append((ua, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    for _, p in scored[keep_top_k:]:
        p.unlink()


def main():
    args = parse_args()
    seed_everything(args.seed)
    tc = TrainConfig()
    mc = ModelConfig()

    if args.dry_run:
        max_samples = tc.dry_run_samples
        max_epochs = tc.dry_run_epochs
        batch_size = min(tc.dry_run_samples, 4)
        num_workers = 0
    else:
        max_samples = args.max_samples
        max_epochs = args.epochs or tc.max_epochs
        batch_size = args.batch_size or tc.batch_size
        num_workers = args.num_workers if args.num_workers is not None else tc.num_workers

    lr = args.lr or tc.learning_rate
    use_prosodic = not args.no_prosodic

    # Logging
    log = setup_logger(MODEL_NAME)
    log.info("=" * 60)
    log.info("Training: %s", MODEL_NAME)
    log.info("Device: %s", DEVICE)
    log.info("Seed: %d", args.seed)
    log.info("Dry run: %s", args.dry_run)
    log.info("Max samples: %s", max_samples)
    log.info("Epochs: %d | Batch: %d | LR: %.2e", max_epochs, batch_size, lr)
    log.info("Prosodic: %s", use_prosodic)
    log.info("=" * 60)

    # --- Dataset ---
    manifest = PREPROCESSED_DIR / "dusha" / "crowd_train" / "manifest.jsonl"
    if not manifest.exists():
        log.error("No manifest at %s — run preprocessing first", manifest)
        sys.exit(1)

    dataset = CODADataset(
        manifest,
        max_samples=max_samples,
        stratified=args.stratified,
        require_hubert=True,
        cache_in_ram=args.cache_in_ram and not args.dry_run,
    )
    log.info("Dataset: %d samples", len(dataset))

    # Speaker-independent split (90/10)
    train_ds, val_ds = speaker_independent_split(dataset, test_size=0.1, seed=args.seed, dry_run=args.dry_run)
    log.info("Train: %d | Val: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=CODADataset.collate_fn,
        num_workers=num_workers,
        pin_memory=tc.pin_memory and not args.dry_run,
        persistent_workers=tc.persistent_workers and num_workers > 0,
        prefetch_factor=tc.prefetch_factor if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tc.eval_batch_size if not args.dry_run else batch_size,
        shuffle=False,
        collate_fn=CODADataset.collate_fn,
        num_workers=num_workers,
        pin_memory=tc.pin_memory and not args.dry_run,
        persistent_workers=tc.persistent_workers and num_workers > 0,
        prefetch_factor=tc.prefetch_factor if num_workers > 0 else None,
    )

    # --- Class weights ---
    all_labels = dataset.get_labels()
    class_weights = compute_class_weights(all_labels, mc.num_classes, method=args.weight_method).to(DEVICE)
    log.info("Class weights (%s): %s", args.weight_method, class_weights.cpu().tolist())
    criterion = WeightedCELoss(weight=class_weights)

    # --- Model ---
    model = AudioOnlyBaseline(
        hubert_dim=mc.hubert_dim,
        prosodic_dim=mc.prosodic_dim,
        hidden_dim=mc.classifier_hidden,
        num_classes=mc.num_classes,
        dropout=mc.classifier_dropout,
        use_prosodic=use_prosodic,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model params: %s", f"{n_params:,}")

    # --- Optimizer + Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=tc.weight_decay
    )
    total_steps = max_epochs * len(train_loader)
    warmup_steps = int(total_steps * tc.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=tc.warmup_ratio,
        anneal_strategy="cos",
    )

    # --- Training loop ---
    ckpt_dir = CHECKPOINTS_DIR / MODEL_NAME
    best_ua = 0.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            DEVICE, epoch, log, args.verbose,
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, DEVICE,
        )

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        eta = elapsed / epoch * (max_epochs - epoch)

        log.info(
            "Epoch %d/%d | train_loss %.4f | val_loss %.4f | "
            "train_UA %.4f | val_UA %.4f | val_F1 %.4f | "
            "time %.1fs | ETA %.1fs",
            epoch, max_epochs,
            train_loss, val_loss,
            train_metrics["unweighted_accuracy"],
            val_metrics["unweighted_accuracy"],
            val_metrics["macro_f1"],
            epoch_time, eta,
        )

        # Checkpoint
        config_snapshot = {
            "model_name": MODEL_NAME,
            "seed": args.seed,
            "max_samples": max_samples,
            "batch_size": batch_size,
            "lr": lr,
            "use_prosodic": use_prosodic,
            "weight_method": args.weight_method,
        }
        save_checkpoint(model, optimizer, epoch, val_metrics, config_snapshot, ckpt_dir)
        cleanup_checkpoints(ckpt_dir, keep_top_k=tc.keep_top_k)

        # Early stopping
        if val_metrics["unweighted_accuracy"] > best_ua:
            best_ua = val_metrics["unweighted_accuracy"]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= tc.patience and not args.dry_run:
                log.info("Early stopping at epoch %d (patience=%d)", epoch, tc.patience)
                break

    total_time = time.time() - start_time
    log.info("=" * 60)
    log.info("Training complete in %.1f min", total_time / 60)
    log.info("Best val UA: %.4f", best_ua)

    # Reload best checkpoint
    best_ckpt = max(
        ckpt_dir.glob("epoch_*.pt"),
        key=lambda p: torch.load(p, weights_only=False, map_location="cpu")
        .get("metrics", {}).get("unweighted_accuracy", 0),
    )
    log.info("Reloading best checkpoint: %s", best_ckpt.name)
    best_state = torch.load(best_ckpt, weights_only=False, map_location=DEVICE)
    model.load_state_dict(best_state["model_state_dict"])
    _, best_val_metrics = evaluate(model, val_loader, criterion, DEVICE)

    log.info("\nBest checkpoint validation metrics:")
    log.info(format_metrics(best_val_metrics))

    # Save final metrics
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / f"{MODEL_NAME}_seed{args.seed}.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "seed": args.seed,
            "best_val_ua": best_ua,
            "final_val_metrics": best_val_metrics,
            "total_time_sec": total_time,
            "config": config_snapshot,
        }, f, indent=2)
    log.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
