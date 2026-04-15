"""
Usage:
    python -m src.training.train_hf_rubert_baseline --dry-run
    python -m src.training.train_hf_rubert_baseline --max-samples 8000
    python -m src.training.train_hf_rubert_baseline # Full
"""

import argparse
import json
import sys
import time
from pathlib import Path

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
    RUBERT_BASE,
    ModelConfig,
    TrainConfig,
    seed_everything,
)
from src.data.dataset import CODADataset, speaker_independent_split
from src.models.baselines import TextOnlyBaseline
from src.training.losses import WeightedCELoss, compute_class_weights
from src.utils.logger import setup_logger
from src.utils.metrics import compute_metrics, format_metrics

MODEL_NAME = "hf_rubert_baseline"


def parse_args():
    p = argparse.ArgumentParser(description=f"Train {MODEL_NAME}")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--stratified", action="store_true", default=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=2e-5,
                   help="LR for BERT fine-tuning (default 2e-5)")
    p.add_argument("--freeze-bert", action="store_true",
                   help="Freeze BERT weights, train only classifier")
    p.add_argument("--weight-method", type=str, default="sqrt_inverse",
                   choices=["inverse", "sqrt_inverse", "effective", "none"],
                   help="Class weighting method (default: sqrt_inverse)")
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader workers (default: from config)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, log, verbose):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch_idx, batch in enumerate(pbar):
        texts = batch["text"]
        labels = batch["label"].to(device)

        logits = model(texts=texts)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if verbose and (batch_idx + 1) % 50 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1e9
                log.debug("  batch %d | loss %.4f | lr %.2e | GPU %.2f GB",
                          batch_idx + 1, loss.item(), lr_now, mem)

    avg_loss = total_loss / max(len(loader), 1)
    metrics = compute_metrics(all_labels, all_preds, DUSHA_EMOTIONS)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        texts = batch["text"]
        labels = batch["label"].to(device)

        logits = model(texts=texts)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        all_preds.extend(logits.argmax(-1).cpu().tolist())
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

    lr = args.lr

    log = setup_logger(MODEL_NAME)
    log.info("=" * 60)
    log.info("Training: %s", MODEL_NAME)
    log.info("Device: %s | Seed: %d | Dry run: %s", DEVICE, args.seed, args.dry_run)
    log.info("Epochs: %d | Batch: %d | LR: %.2e", max_epochs, batch_size, lr)
    log.info("Freeze BERT: %s", args.freeze_bert)
    log.info("=" * 60)

    # --- Dataset ---
    manifest = PREPROCESSED_DIR / "dusha" / "crowd_train" / "manifest.jsonl"
    if not manifest.exists():
        log.error("No manifest at %s", manifest)
        sys.exit(1)

    dataset = CODADataset(
        manifest,
        max_samples=max_samples,
        stratified=args.stratified,
        require_text=True,
        require_hubert=False,
    )
    log.info("Dataset: %d samples (text required)", len(dataset))

    # Speaker-independent split (90/10)
    train_ds, val_ds = speaker_independent_split(dataset, test_size=0.1, seed=args.seed, dry_run=args.dry_run)
    log.info("Train: %d | Val: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=CODADataset.collate_fn, num_workers=num_workers,
        pin_memory=tc.pin_memory and not args.dry_run,
        persistent_workers=tc.persistent_workers and num_workers > 0,
        prefetch_factor=tc.prefetch_factor if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tc.eval_batch_size if not args.dry_run else batch_size,
        shuffle=False, collate_fn=CODADataset.collate_fn, num_workers=num_workers,
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
    model = TextOnlyBaseline(
        model_name=RUBERT_BASE,
        hidden_dim=mc.classifier_hidden,
        num_classes=mc.num_classes,
        dropout=mc.classifier_dropout,
        freeze_bert=args.freeze_bert,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Total params: %s | Trainable: %s", f"{n_params:,}", f"{n_train_params:,}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=tc.weight_decay,
    )
    total_steps = max_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=tc.warmup_ratio, anneal_strategy="cos",
    )

    # --- Training loop ---
    ckpt_dir = CHECKPOINTS_DIR / MODEL_NAME
    best_ua = 0.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        train_loss, train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            DEVICE, epoch, log, args.verbose,
        )
        val_loss, val_m = evaluate(model, val_loader, criterion, DEVICE)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        eta = elapsed / epoch * (max_epochs - epoch)

        log.info(
            "Epoch %d/%d | train_loss %.4f | val_loss %.4f | "
            "train_UA %.4f | val_UA %.4f | val_F1 %.4f | "
            "time %.1fs | ETA %.1fs",
            epoch, max_epochs, train_loss, val_loss,
            train_m["unweighted_accuracy"], val_m["unweighted_accuracy"],
            val_m["macro_f1"], epoch_time, eta,
        )

        config_snapshot = {
            "model_name": MODEL_NAME, "seed": args.seed,
            "max_samples": max_samples, "batch_size": batch_size,
            "lr": lr, "freeze_bert": args.freeze_bert,
            "weight_method": args.weight_method,
        }
        save_checkpoint(model, optimizer, epoch, val_m, config_snapshot, ckpt_dir)
        cleanup_checkpoints(ckpt_dir, keep_top_k=tc.keep_top_k)

        if val_m["unweighted_accuracy"] > best_ua:
            best_ua = val_m["unweighted_accuracy"]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= tc.patience and not args.dry_run:
                log.info("Early stopping at epoch %d", epoch)
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
    _, best_val_m = evaluate(model, val_loader, criterion, DEVICE)

    log.info("\nBest checkpoint validation metrics:")
    log.info(format_metrics(best_val_m))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / f"{MODEL_NAME}_seed{args.seed}.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "model": MODEL_NAME, "seed": args.seed,
            "best_val_ua": best_ua, "final_val_metrics": best_val_m,
            "total_time_sec": total_time, "config": config_snapshot,
        }, f, indent=2)
    log.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
