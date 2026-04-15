"""
Usage:
    python -m src.training.evaluate --checkpoint results/checkpoints/coda_biattn/epoch_006.pt --test-manifest results/preprocessed/dusha/crowd_test/manifest.jsonl --variant biattn
    python -m src.training.evaluate --checkpoint results/checkpoints/hf_audio_baseline/epoch_021.pt --test-manifest results/preprocessed/dusha/crowd_test/manifest.jsonl --model-type audio_only
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DEVICE,
    DUSHA_EMOTIONS,
    IEMOCAP_EMOTIONS,
    METRICS_DIR,
    RUBERT_BASE,
    ModelConfig,
    seed_everything,
)
from src.data.dataset import CODADataset
from src.models.baselines import AudioOnlyBaseline, MultimodalConcatBaseline, TextOnlyBaseline
from src.models.coda_pipeline import CODAPipeline
from src.utils.metrics import compute_metrics, format_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained model")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test-manifest", type=str, required=True)

    p.add_argument("--model-type", choices=["audio_only", "text_only", "multimodal_concat", "coda"],
                   default="coda")
    p.add_argument("--variant", choices=["base", "uniattn", "biattn", "biattn_context", "full"],
                   default="biattn")

    p.add_argument("--no-prosodic", action="store_true",
                   help="Use prosodic_dim=0 (for no-prosodic ablation checkpoint)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dataset-name", choices=["dusha", "iemocap"], default="dusha",
                   help="Which label set to use for metrics")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_model(args, device):
    mc = ModelConfig()
    if args.model_type == "audio_only":
        model = AudioOnlyBaseline(
            hubert_dim=mc.hubert_dim, prosodic_dim=mc.prosodic_dim,
            hidden_dim=mc.classifier_hidden, num_classes=mc.num_classes,
            dropout=mc.classifier_dropout,
        )
    elif args.model_type == "text_only":
        model = TextOnlyBaseline(
            model_name=RUBERT_BASE, hidden_dim=mc.classifier_hidden,
            num_classes=mc.num_classes, dropout=mc.classifier_dropout,
        )
    elif args.model_type == "multimodal_concat":
        model = MultimodalConcatBaseline(
            hubert_dim=mc.hubert_dim, prosodic_dim=mc.prosodic_dim,
            text_model_name=RUBERT_BASE, hidden_dim=mc.classifier_hidden,
            num_classes=mc.num_classes, dropout=mc.classifier_dropout,
        )
    elif args.model_type == "coda":
        prosodic_dim = 0 if args.no_prosodic else mc.prosodic_dim
        model = CODAPipeline(
            variant=args.variant,
            hubert_dim=mc.hubert_dim, bert_model_name=RUBERT_BASE,
            prosodic_dim=prosodic_dim, cross_attn_dim=mc.cross_attn_dim,
            cross_attn_heads=mc.cross_attn_heads,
            cross_attn_dropout=mc.cross_attn_dropout,
            context_hidden=mc.context_hidden, context_layers=mc.context_layers,
            context_dropout=mc.context_dropout,
            classifier_hidden=mc.classifier_hidden,
            classifier_dropout=mc.classifier_dropout,
            num_classes=mc.num_classes, freeze_bert=True,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, device, is_coda=False):
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Evaluating"):
        hubert = batch["hubert"].to(device)
        mask = batch["hubert_mask"].to(device)
        prosodic = batch["prosodic"].to(device)
        labels = batch["label"].to(device)
        texts = batch["text"]

        rubert_emb = batch["rubert"].to(device) if batch["rubert"] is not None else None
        rubert_mask = batch["rubert_mask"].to(device) if batch["rubert_mask"] is not None else None

        if is_coda:
            output = model(
                hubert=hubert, hubert_mask=mask, prosodic=prosodic,
                texts=texts, rubert_emb=rubert_emb, rubert_mask=rubert_mask,
            )
            logits = output["logits"]
        else:
            logits = model(hubert=hubert, hubert_mask=mask, prosodic=prosodic, texts=texts)

        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return all_labels, all_preds


def main():
    args = parse_args()
    seed_everything(args.seed)

    label_names = DUSHA_EMOTIONS if args.dataset_name == "dusha" else IEMOCAP_EMOTIONS

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")

    print(f"Loading test data: {args.test_manifest}")
    dataset = CODADataset(args.test_manifest, require_hubert=True)
    print(f"Test samples: {len(dataset)}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=CODADataset.collate_fn, num_workers=4,
        pin_memory=True,
    )

    print(f"Building model: {args.model_type}" + (f" ({args.variant})" if args.model_type == "coda" else ""))
    model = build_model(args, DEVICE)

    # Load weights
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded weights from epoch {ckpt.get('epoch', '?')}")

    is_coda = args.model_type == "coda"
    all_labels, all_preds = evaluate(model, loader, DEVICE, is_coda)
    metrics = compute_metrics(all_labels, all_preds, label_names)

    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(format_metrics(metrics))
    print()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = f"coda_{args.variant}" if is_coda else f"hf_{args.model_type}"
    if is_coda and args.no_prosodic:
        model_name += "_noprosodic"
    out_path = METRICS_DIR / f"{model_name}_test_{args.dataset_name}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": model_name,
            "checkpoint": args.checkpoint,
            "test_manifest": args.test_manifest,
            "test_samples": len(dataset),
            "metrics": metrics,
        }, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
