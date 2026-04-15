"""
Usage:
    python -m src.utils.pipeline_check --model audio_only
    python -m src.utils.pipeline_check --model text_only
    python -m src.utils.pipeline_check --model multimodal_concat
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DEVICE,
    DUSHA_EMOTIONS,
    PREPROCESSED_DIR,
    ModelConfig,
    TrainConfig,
    seed_everything,
)
from src.data.dataset import CODADataset
from src.models.baselines import (
    AudioOnlyBaseline,
    MultimodalConcatBaseline,
    TextOnlyBaseline,
)
from src.training.losses import WeightedCELoss, compute_class_weights
from src.utils.export import save_torchinfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline_check")


def build_model(model_type: str, device: torch.device):
    mc = ModelConfig()
    if model_type == "audio_only":
        model = AudioOnlyBaseline(
            hubert_dim=mc.hubert_dim,
            prosodic_dim=mc.prosodic_dim,
            hidden_dim=mc.classifier_hidden,
            num_classes=mc.num_classes,
            dropout=mc.classifier_dropout,
        )
    elif model_type == "text_only":
        model = TextOnlyBaseline(
            hidden_dim=mc.classifier_hidden,
            num_classes=mc.num_classes,
            dropout=mc.classifier_dropout,
            freeze_bert=False,
        )
    elif model_type == "multimodal_concat":
        model = MultimodalConcatBaseline(
            hubert_dim=mc.hubert_dim,
            prosodic_dim=mc.prosodic_dim,
            hidden_dim=mc.classifier_hidden,
            num_classes=mc.num_classes,
            dropout=mc.classifier_dropout,
            freeze_bert=True,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model.to(device)


def run_check(model_type: str):
    seed_everything()
    tc = TrainConfig()

    log.info("=" * 60)
    log.info("PIPELINE CHECK: %s", model_type)
    log.info("Device: %s", DEVICE)
    log.info("=" * 60)

    # --- 1. Dataset ---
    manifest = PREPROCESSED_DIR / "dusha" / "crowd_train" / "manifest.jsonl"
    if not manifest.exists():
        log.error("Manifest not found: %s", manifest)
        log.error("Run preprocessing first: python -m src.data.preprocessing --dataset dusha --subset crowd_train --limit 500")
        return False

    log.info("[1/8] Loading dataset from %s", manifest)
    dataset = CODADataset(
        manifest, max_samples=tc.dry_run_samples, require_text=True
    )
    log.info("  Loaded %d samples", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=min(tc.dry_run_samples, len(dataset)),
        collate_fn=CODADataset.collate_fn,
        num_workers=0,
    )
    batch = next(iter(loader))
    log.info("  hubert shape:      %s", batch["hubert"].shape)
    log.info("  hubert_mask shape: %s", batch["hubert_mask"].shape)
    log.info("  prosodic shape:    %s", batch["prosodic"].shape)
    log.info("  label shape:       %s", batch["label"].shape)
    log.info("  texts:             %s", batch["text"][:2])

    # --- 2. Model ---
    log.info("[2/8] Building model: %s", model_type)
    model = build_model(model_type, DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("  Total params:     %s", f"{n_params:,}")
    log.info("  Trainable params: %s", f"{n_train:,}")

    # --- 3. Forward pass ---
    log.info("[3/8] Forward pass")
    model.train()
    batch_gpu = {
        k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    logits = model(
        hubert=batch_gpu.get("hubert"),
        hubert_mask=batch_gpu.get("hubert_mask"),
        prosodic=batch_gpu.get("prosodic"),
        texts=batch_gpu.get("text"),
    )
    log.info("  logits shape: %s", logits.shape)
    log.info("  logits finite: %s", torch.isfinite(logits).all().item())

    # --- 4. Loss ---
    log.info("[4/8] Loss computation")
    labels = dataset.get_labels()
    weights = compute_class_weights(labels, num_classes=4).to(DEVICE)
    criterion = WeightedCELoss(weight=weights)
    loss = criterion(logits, batch_gpu["label"])
    log.info("  loss value: %.6f", loss.item())
    log.info("  loss finite: %s", torch.isfinite(loss).item())

    # --- 5. Backward ---
    log.info("[5/8] Backward pass")
    loss.backward()
    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            grad_norms.append(gn)
            if not torch.isfinite(torch.tensor(gn)):
                log.warning("  Non-finite gradient in %s", name)
    log.info("  Gradients computed for %d parameters", len(grad_norms))
    log.info("  Mean grad norm: %.6f", sum(grad_norms) / max(len(grad_norms), 1))
    log.info("  Max grad norm:  %.6f", max(grad_norms) if grad_norms else 0)

    # --- 6. Optimizer step ---
    log.info("[6/8] Optimizer step")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay
    )
    optimizer.step()
    optimizer.zero_grad()
    log.info("  Optimizer step completed")

    # --- 7. Second forward pass ---
    log.info("[7/8] Second forward pass (verify weights updated)")
    logits2 = model(
        hubert=batch_gpu.get("hubert"),
        hubert_mask=batch_gpu.get("hubert_mask"),
        prosodic=batch_gpu.get("prosodic"),
        texts=batch_gpu.get("text"),
    )
    loss2 = criterion(logits2, batch_gpu["label"])
    log.info("  loss after step: %.6f (was %.6f)", loss2.item(), loss.item())

    # --- 8. torchinfo summary ---
    log.info("[8/8] torchinfo summary")
    try:
        summary_text = save_torchinfo(
            model,
            input_data={
                "hubert": batch_gpu.get("hubert"),
                "hubert_mask": batch_gpu.get("hubert_mask"),
                "prosodic": batch_gpu.get("prosodic"),
                "texts": batch_gpu.get("text"),
            },
            model_name=model_type,
        )
        # Print first 30 lines
        for line in summary_text.split("\n")[:30]:
            log.info("  %s", line)
        log.info("  ... (full summary saved to results/torchinfo/%s.txt)", model_type)
    except Exception as e:
        log.warning("  torchinfo failed (non-critical): %s", e)

    log.info("=" * 60)
    log.info("PIPELINE CHECK PASSED: %s", model_type)
    log.info("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Pipeline Check")
    parser.add_argument(
        "--model",
        choices=["audio_only", "text_only", "multimodal_concat", "all"],
        default="all",
    )
    args = parser.parse_args()

    models = (
        ["audio_only", "text_only", "multimodal_concat"]
        if args.model == "all"
        else [args.model]
    )

    results = {}
    for m in models:
        ok = run_check(m)
        results[m] = ok

    log.info("")
    log.info("SUMMARY:")
    for m, ok in results.items():
        status = "PASS" if ok else "FAIL"
        log.info("  %-25s %s", m, status)


if __name__ == "__main__":
    main()
