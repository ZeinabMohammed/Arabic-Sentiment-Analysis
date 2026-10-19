"""Export the sentiment model to ONNX.

This is the serving-acceleration step the course calls out as "TensorRT FP16" —
TensorRT itself needs an NVIDIA GPU, which this project was built without. ONNX
Runtime is the portable substitute: it runs the same graph-optimized model on
CPU here, and on a GPU box the identical .onnx file is also what you'd feed to
`trtexec --fp16` to build a real TensorRT engine (command printed below).

Run as: python -m src.optimization.export_onnx [--model-path <dir-or-hub-id>]
"""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

from src.config import BASE_MODEL_NAME, MAX_SEQ_LENGTH, MODELS_DIR
from src.models.sentiment_model import ArabicSentimentModel


def export(model_path: str, output_path: str) -> str:
    wrapper = ArabicSentimentModel(model_path)
    tokenizer: AutoTokenizer = wrapper.tokenizer
    model = wrapper.model.cpu().eval()

    dummy = tokenizer(
        "عينة نص للتصدير", return_tensors="pt", padding="max_length", max_length=MAX_SEQ_LENGTH
    )

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    # Sanity-check: ONNX Runtime output should match the PyTorch output.
    with torch.inference_mode():
        torch_logits = model(**dummy).logits.numpy()

    session = ort.InferenceSession(output_path)
    onnx_logits = session.run(
        None,
        {
            "input_ids": dummy["input_ids"].numpy(),
            "attention_mask": dummy["attention_mask"].numpy(),
        },
    )[0]
    assert np.allclose(torch_logits, onnx_logits, atol=1e-3), "ONNX output diverged from PyTorch"

    print(f"Exported {output_path} (verified against PyTorch, atol=1e-3)")
    print(
        "On a GPU box, build a real TensorRT FP16 engine from this file with:\n"
        f"  trtexec --onnx={output_path} --fp16 --saveEngine={output_path.replace('.onnx', '.trt')}"
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=BASE_MODEL_NAME)
    parser.add_argument("--output", type=str, default=str(MODELS_DIR / "model_fp32.onnx"))
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    export(args.model_path, args.output)
