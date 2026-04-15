"""
Usage:
    python -m src.training.train_coda --variant biattn --dry-run
    python -m src.training.train_coda --variant base --max-samples
    python -m src.training.train_coda --variant biattn --max-samples
    python -m src.training.train_coda --variant biattn_context --max-samples
    python -m src.training.train_coda --variant full --max-samples
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
    METRICS_DIR,
    PREPROCESSED_DIR,
    RUBERT_BASE,
    ModelConfig,
    TrainConfig,
    seed_everything,
)
from src.data.dataset import CODADataset, speaker_independent_split
from src.models.coda_pipeline import VARIANTS, CODAPipeline
from src.training.losses import WeightedCELoss, compute_class_weights
from src.utils.logger import setup_logger
from src.utils.metrics import compute_metrics, format_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Train CODA Pipeline")
    p.add_argument("--variant", choices=list(VARIANTS), required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--stratified", action="store_true", default=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--freeze-bert", action="store_true", default=True,
                   help="Freeze ruBERT weights (default: True)")
    p.add_argument("--no-freeze-bert", dest="freeze_bert", action="store_false")
    p.add_argument("--no-prosodic", action="store_true")
    p.add_argument("--cache-in-ram", action="store_true",
                   help="Pre-load all HuBERT features into RAM (~14 GB)")
    p.add_argument("--weight-method", type=str, default="sqrt_inverse",
                   choices=["inverse", "sqrt_inverse", "effective", "none"],
                   help="Class weighting method (default: sqrt_inverse)")
    p.add_argument("--amp", action="store_true", default=True,
                   help="Use mixed precision training (default: True)")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader workers (default: from config)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from best checkpoint in model's checkpoint dir")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Resume from a specific checkpoint file path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, log, verbose, scaler=None):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    use_amp = scaler is not None

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch_idx, batch in enumerate(pbar):
        hubert = batch["hubert"].to(device)
        mask = batch["hubert_mask"].to(device)
        prosodic = batch["prosodic"].to(device)
        labels = batch["label"].to(device)
        texts = batch["text"]

        # Pre-computed ruBERT embeddings (if available from preprocessing)
        rubert_emb = batch["rubert"].to(device) if batch["rubert"] is not None else None
        rubert_mask = batch["rubert_mask"].to(device) if batch["rubert_mask"] is not None else None

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            output = model(
                hubert=hubert, hubert_mask=mask,
                prosodic=prosodic, texts=texts,
                rubert_emb=rubert_emb, rubert_mask=rubert_mask,
            )
            logits = output["logits"]
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
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
            grad_norm = sum(
                p.grad.norm().item()
                for p in model.parameters()
                if p.grad is not None
            )
            lr_now = optimizer.param_groups[0]["lr"]
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log.debug(
                "  batch %d | loss %.4f | grad %.4f | lr %.2e | GPU %.2f GB",
                batch_idx + 1, loss.item(), grad_norm, lr_now, mem,
            )

    avg_loss = total_loss / max(len(loader), 1)
    metrics = compute_metrics(all_labels, all_preds, DUSHA_EMOTIONS)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        hubert = batch["hubert"].to(device)
        mask = batch["hubert_mask"].to(device)
        prosodic = batch["prosodic"].to(device)
        labels = batch["label"].to(device)
        texts = batch["text"]

        rubert_emb = batch["rubert"].to(device) if batch["rubert"] is not None else None
        rubert_mask = batch["rubert_mask"].to(device) if batch["rubert_mask"] is not None else None

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            output = model(
                hubert=hubert, hubert_mask=mask,
                prosodic=prosodic, texts=texts,
                rubert_emb=rubert_emb, rubert_mask=rubert_mask,
            )
            logits = output["logits"]
            loss = criterion(logits, labels)

        total_loss += loss.item()
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    metrics = compute_metrics(all_labels, all_preds, DUSHA_EMOTIONS)
    return avg_loss, metrics


def save_checkpoint(model, optimizer, epoch, metrics, config, ckpt_dir,
                    scheduler=None, scaler=None):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"epoch_{epoch:03d}.pt"
    data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    if scheduler is not None:
        data["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        data["scaler_state_dict"] = scaler.state_dict()
    torch.save(data, path)
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

    variant = args.variant
    model_name = f"coda_{variant}"
    if args.no_prosodic:
        model_name += "_noprosodic"

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

    log = setup_logger(model_name)
    log.info("=" * 60)
    log.info("Training: CODA-%s", variant.upper())
    log.info("Device: %s | Seed: %d | Dry run: %s", DEVICE, args.seed, args.dry_run)
    log.info("Epochs: %d | Batch: %d | LR: %.2e", max_epochs, batch_size, lr)
    log.info("Freeze BERT: %s | Prosodic: %s | AMP: %s", args.freeze_bert, not args.no_prosodic, args.amp)
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
        require_hubert=True,
        cache_in_ram=args.cache_in_ram and not args.dry_run,
    )
    log.info("Dataset: %d samples", len(dataset))

    # Speaker-independent split: no speaker in both train and val
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
    log.info("Class weights (%s): %s", args.weight_method, [f"{w:.3f}" for w in class_weights.cpu().tolist()])
    criterion = WeightedCELoss(weight=class_weights)

    # --- Model ---
    model = CODAPipeline(
        variant=variant,
        hubert_dim=mc.hubert_dim,
        bert_model_name=RUBERT_BASE,
        prosodic_dim=mc.prosodic_dim if not args.no_prosodic else 0,
        cross_attn_dim=mc.cross_attn_dim,
        cross_attn_heads=mc.cross_attn_heads,
        cross_attn_dropout=mc.cross_attn_dropout,
        context_hidden=mc.context_hidden,
        context_layers=mc.context_layers,
        context_dropout=mc.context_dropout,
        classifier_hidden=mc.classifier_hidden,
        classifier_dropout=mc.classifier_dropout,
        num_classes=mc.num_classes,
        freeze_bert=args.freeze_bert,
        dissonance_contamination=mc.dissonance_contamination,
    ).to(DEVICE)

    n_total = sum(p.numel() for p in model.parameters())
    n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Total params: %s | Trainable: %s", f"{n_total:,}", f"{n_train_p:,}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=tc.weight_decay,
    )
    total_steps = max_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=tc.warmup_ratio, anneal_strategy="cos",
    )

    # --- AMP scaler ---
    use_amp = args.amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler() if use_amp else None

    # --- Resume from checkpoint ---
    ckpt_dir = CHECKPOINTS_DIR / model_name
    start_epoch = 1
    best_ua = 0.0
    patience_counter = 0

    if args.resume or args.resume_from:
        if args.resume_from:
            resume_path = Path(args.resume_from)
        else:
            # Find best checkpoint by UA in the model's dir
            ckpts = list(ckpt_dir.glob("epoch_*.pt"))
            if not ckpts:
                log.error("--resume: no checkpoints found in %s", ckpt_dir)
                sys.exit(1)
            resume_path = max(
                ckpts,
                key=lambda p: torch.load(p, weights_only=False, map_location="cpu")
                .get("metrics", {}).get("unweighted_accuracy", 0),
            )

        if not resume_path.exists():
            log.error("Checkpoint not found: %s", resume_path)
            sys.exit(1)

        log.info("Resuming from checkpoint: %s", resume_path)
        ckpt_data = torch.load(resume_path, weights_only=False, map_location=DEVICE)
        model.load_state_dict(ckpt_data["model_state_dict"])
        optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
        start_epoch = ckpt_data["epoch"] + 1
        best_ua = ckpt_data.get("metrics", {}).get("unweighted_accuracy", 0.0)

        # Fast-forward scheduler to the correct step
        steps_done = ckpt_data["epoch"] * len(train_loader)
        if "scheduler_state_dict" in ckpt_data:
            scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])
        else:
            for _ in range(steps_done):
                scheduler.step()
            log.info("Fast-forwarded scheduler by %d steps", steps_done)

        if scaler is not None and "scaler_state_dict" in ckpt_data:
            scaler.load_state_dict(ckpt_data["scaler_state_dict"])

        log.info("Resumed at epoch %d | best_ua so far: %.4f", start_epoch, best_ua)

    # --- Training loop ---
    start_time = time.time()

    for epoch in range(start_epoch, max_epochs + 1):
        epoch_start = time.time()

        train_loss, train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            DEVICE, epoch, log, args.verbose, scaler=scaler,
        )
        val_loss, val_m = evaluate(model, val_loader, criterion, DEVICE, use_amp=use_amp)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        epochs_done = epoch - start_epoch + 1
        eta = elapsed / epochs_done * (max_epochs - epoch)

        log.info(
            "Epoch %d/%d | train_loss %.4f | val_loss %.4f | "
            "train_UA %.4f | val_UA %.4f | val_F1 %.4f | "
            "time %.1fs | ETA %.1fs",
            epoch, max_epochs, train_loss, val_loss,
            train_m["unweighted_accuracy"], val_m["unweighted_accuracy"],
            val_m["macro_f1"], epoch_time, eta,
        )

        config_snap = {
            "model_name": model_name, "variant": variant,
            "seed": args.seed, "max_samples": max_samples,
            "batch_size": batch_size, "lr": lr,
            "freeze_bert": args.freeze_bert,
            "weight_method": args.weight_method,
        }
        save_checkpoint(model, optimizer, epoch, val_m, config_snap, ckpt_dir,
                       scheduler=scheduler, scaler=scaler)
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

    # Reload best checkpoint and re-evaluate for final metrics
    best_ckpt = max(
        ckpt_dir.glob("epoch_*.pt"),
        key=lambda p: torch.load(p, weights_only=False, map_location="cpu")
        .get("metrics", {}).get("unweighted_accuracy", 0),
    )
    log.info("Reloading best checkpoint: %s", best_ckpt.name)
    best_state = torch.load(best_ckpt, weights_only=False, map_location=DEVICE)
    model.load_state_dict(best_state["model_state_dict"])
    _, best_val_m = evaluate(model, val_loader, criterion, DEVICE, use_amp=use_amp)

    log.info("\nBest checkpoint validation metrics:")
    log.info(format_metrics(best_val_m))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / f"{model_name}_seed{args.seed}.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "model": model_name, "variant": variant, "seed": args.seed,
            "best_val_ua": best_ua, "final_val_metrics": best_val_m,
            "total_time_sec": total_time, "config": config_snap,
        }, f, indent=2)
    log.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
