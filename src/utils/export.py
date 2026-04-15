from pathlib import Path

import torch
import torch.nn as nn
from torchinfo import summary

from src.config import NETRON_DIR, TORCHINFO_DIR


def save_torchinfo(
    model: nn.Module,
    input_data: dict | tuple | None = None,
    input_size: tuple | list | None = None,
    model_name: str = "model",
    output_dir: Path = TORCHINFO_DIR,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)

    info = summary(
        model,
        input_data=input_data,
        input_size=input_size,
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )
    text = str(info)

    out_path = output_dir / f"{model_name}.txt"
    out_path.write_text(text, encoding="utf-8")
    return text


def export_onnx(
    model: nn.Module,
    dummy_input: tuple,
    model_name: str = "model",
    output_dir: Path = NETRON_DIR,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_name}.onnx"

    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=input_names,
        output_names=output_names or ["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
    )
    return out_path
