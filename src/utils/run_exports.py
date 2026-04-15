import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    CHECKPOINTS_DIR,
    DEVICE,
    NETRON_DIR,
    RUBERT_BASE,
    TORCHINFO_DIR,
    ModelConfig,
    seed_everything,
)
from src.models.baselines import AudioOnlyBaseline, TextOnlyBaseline
from src.models.coda_pipeline import CODAPipeline
from src.utils.export import save_torchinfo

seed_everything(42)
mc = ModelConfig()

# (batch=2, seq_len=50 audio frames, 10 text tokens)
B, T_A, T_T = 2, 50, 10
DUMMY_HUBERT = torch.randn(B, T_A, 768)
DUMMY_MASK = torch.ones(B, T_A, dtype=torch.bool)
DUMMY_PROSODIC = torch.randn(B, 5)
DUMMY_RUBERT_EMB = torch.randn(B, T_T, 768)
DUMMY_RUBERT_MASK = torch.ones(B, T_T, dtype=torch.bool)
DUMMY_INPUT_IDS = torch.randint(0, 1000, (B, T_T))
DUMMY_ATTN_MASK = torch.ones(B, T_T, dtype=torch.long)


# ── ONNX wrappers  ──


class AudioBaselineONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hubert, hubert_mask, prosodic):
        return self.model(hubert=hubert, hubert_mask=hubert_mask, prosodic=prosodic)


class TextBaselineONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


class CODAOnnx(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hubert, hubert_mask, prosodic, rubert_emb, rubert_mask):
        out = self.model(
            hubert=hubert, hubert_mask=hubert_mask, prosodic=prosodic,
            rubert_emb=rubert_emb, rubert_mask=rubert_mask,
        )
        return out["logits"]


class CODANoProsOnnx(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hubert, hubert_mask, rubert_emb, rubert_mask):
        out = self.model(
            hubert=hubert, hubert_mask=hubert_mask,
            rubert_emb=rubert_emb, rubert_mask=rubert_mask,
        )
        return out["logits"]


def _load_ckpt(model, ckpt_dir_name):
    ckpt_dir = CHECKPOINTS_DIR / ckpt_dir_name
    ckpts = list(ckpt_dir.glob("epoch_*.pt"))
    if not ckpts:
        print(f"  WARNING: no checkpoints in {ckpt_dir}")
        return False
    best = max(
        ckpts,
        key=lambda p: torch.load(p, weights_only=False, map_location="cpu")
        .get("metrics", {}).get("unweighted_accuracy", 0),
    )
    state = torch.load(best, weights_only=False, map_location="cpu")
    model.load_state_dict(state["model_state_dict"], strict=False)
    print(f"  Loaded {best.name} (epoch {state.get('epoch', '?')})")
    return True


def _export_onnx(wrapper, dummy_inputs, model_name, input_names, dynamic_axes):
    NETRON_DIR.mkdir(parents=True, exist_ok=True)
    out_path = NETRON_DIR / f"{model_name}.onnx"
    wrapper.eval()
    try:
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(out_path),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
        )
        print(f"  ONNX saved: {out_path}")
    except Exception as e:
        print(f"  ONNX FAILED: {e}")


def export_audio_baseline():
    name = "hf_audio_baseline"
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    model = AudioOnlyBaseline(
        hubert_dim=mc.hubert_dim, prosodic_dim=mc.prosodic_dim,
        hidden_dim=mc.classifier_hidden, num_classes=mc.num_classes,
        dropout=mc.classifier_dropout,
    ).to(DEVICE)
    _load_ckpt(model, name)

    # torchinfo
    dummy = {
        "hubert": DUMMY_HUBERT.to(DEVICE),
        "hubert_mask": DUMMY_MASK.to(DEVICE),
        "prosodic": DUMMY_PROSODIC.to(DEVICE),
    }
    text = save_torchinfo(model, input_data=dummy, model_name=name)
    print(f"  torchinfo saved ({len(text)} chars)")

    # ONNX
    wrapper = AudioBaselineONNX(model).to(DEVICE)
    inputs = (DUMMY_HUBERT.to(DEVICE), DUMMY_MASK.to(DEVICE), DUMMY_PROSODIC.to(DEVICE))
    _export_onnx(wrapper, inputs, name,
                 input_names=["hubert", "hubert_mask", "prosodic"],
                 dynamic_axes={"hubert": {0: "batch", 1: "time"}, "hubert_mask": {0: "batch", 1: "time"},
                               "prosodic": {0: "batch"}, "logits": {0: "batch"}})


def export_text_baseline():
    name = "hf_rubert_baseline"
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    model = TextOnlyBaseline(
        model_name=RUBERT_BASE, hidden_dim=mc.classifier_hidden,
        num_classes=mc.num_classes, dropout=mc.classifier_dropout,
    ).to(DEVICE)
    _load_ckpt(model, name)

    # torchinfo — pass tokenized inputs
    dummy = {
        "input_ids": DUMMY_INPUT_IDS.to(DEVICE),
        "attention_mask": DUMMY_ATTN_MASK.to(DEVICE),
    }
    text = save_torchinfo(model, input_data=dummy, model_name=name)
    print(f"  torchinfo saved ({len(text)} chars)")

    # ONNX
    wrapper = TextBaselineONNX(model).to(DEVICE)
    inputs = (DUMMY_INPUT_IDS.to(DEVICE), DUMMY_ATTN_MASK.to(DEVICE))
    _export_onnx(wrapper, inputs, name,
                 input_names=["input_ids", "attention_mask"],
                 dynamic_axes={"input_ids": {0: "batch", 1: "seq_len"},
                               "attention_mask": {0: "batch", 1: "seq_len"},
                               "logits": {0: "batch"}})


def export_coda_variant(variant, no_prosodic=False):
    name = f"coda_{variant}"
    if no_prosodic:
        name += "_noprosodic"
    print(f"\n{'='*60}\n{name}\n{'='*60}")

    prosodic_dim = 0 if no_prosodic else mc.prosodic_dim
    model = CODAPipeline(
        variant=variant,
        hubert_dim=mc.hubert_dim, bert_model_name=RUBERT_BASE,
        prosodic_dim=prosodic_dim, cross_attn_dim=mc.cross_attn_dim,
        cross_attn_heads=mc.cross_attn_heads, cross_attn_dropout=mc.cross_attn_dropout,
        context_hidden=mc.context_hidden, context_layers=mc.context_layers,
        context_dropout=mc.context_dropout, classifier_hidden=mc.classifier_hidden,
        classifier_dropout=mc.classifier_dropout, num_classes=mc.num_classes,
        freeze_bert=True,
        dissonance_contamination=mc.dissonance_contamination,
    ).to(DEVICE)
    _load_ckpt(model, name)

    # torchinfo — use pre-computed ruBERT embeddings
    dummy = {
        "hubert": DUMMY_HUBERT.to(DEVICE),
        "hubert_mask": DUMMY_MASK.to(DEVICE),
        "prosodic": DUMMY_PROSODIC.to(DEVICE),
        "rubert_emb": DUMMY_RUBERT_EMB.to(DEVICE),
        "rubert_mask": DUMMY_RUBERT_MASK.to(DEVICE),
    }
    if no_prosodic:
        dummy.pop("prosodic")
    text = save_torchinfo(model, input_data=dummy, model_name=name)
    print(f"  torchinfo saved ({len(text)} chars)")

    # ONNX
    if no_prosodic:
        wrapper = CODANoProsOnnx(model).to(DEVICE)
        inputs = (
            DUMMY_HUBERT.to(DEVICE), DUMMY_MASK.to(DEVICE),
            DUMMY_RUBERT_EMB.to(DEVICE), DUMMY_RUBERT_MASK.to(DEVICE),
        )
        _export_onnx(wrapper, inputs, name,
                     input_names=["hubert", "hubert_mask", "rubert_emb", "rubert_mask"],
                     dynamic_axes={"hubert": {0: "batch", 1: "time_a"},
                                   "hubert_mask": {0: "batch", 1: "time_a"},
                                   "rubert_emb": {0: "batch", 1: "time_t"},
                                   "rubert_mask": {0: "batch", 1: "time_t"},
                                   "logits": {0: "batch"}})
    else:
        wrapper = CODAOnnx(model).to(DEVICE)
        inputs = (
            DUMMY_HUBERT.to(DEVICE), DUMMY_MASK.to(DEVICE),
            DUMMY_PROSODIC.to(DEVICE),
            DUMMY_RUBERT_EMB.to(DEVICE), DUMMY_RUBERT_MASK.to(DEVICE),
        )
        _export_onnx(wrapper, inputs, name,
                     input_names=["hubert", "hubert_mask", "prosodic", "rubert_emb", "rubert_mask"],
                     dynamic_axes={"hubert": {0: "batch", 1: "time_a"},
                                   "hubert_mask": {0: "batch", 1: "time_a"},
                                   "prosodic": {0: "batch"},
                                   "rubert_emb": {0: "batch", 1: "time_t"},
                                   "rubert_mask": {0: "batch", 1: "time_t"},
                                   "logits": {0: "batch"}})


def main():
    print("Exporting all models: torchinfo + ONNX")
    print(f"Device: {DEVICE}")

    export_audio_baseline()
    export_text_baseline()

    for variant in ["base", "uniattn", "biattn", "biattn_context", "full"]:
        export_coda_variant(variant)

    export_coda_variant("full", no_prosodic=True)

    print("\n" + "=" * 60)
    print("ALL EXPORTS DONE")
    print(f"  torchinfo → {TORCHINFO_DIR}")
    print(f"  ONNX      → {NETRON_DIR}")


if __name__ == "__main__":
    main()
