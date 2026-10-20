"""Dynamic INT8 quantization of an exported ONNX model.

Run as: python -m src.optimization.quantize --input models/model_fp32.onnx
"""

from __future__ import annotations

import argparse

from onnxruntime.quantization import QuantType, quantize_dynamic

from src.config import MODELS_DIR


def quantize(input_path: str, output_path: str) -> str:
    quantize_dynamic(input_path, output_path, weight_type=QuantType.QInt8)
    print(f"Quantized {input_path} -> {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(MODELS_DIR / "model_fp32.onnx"))
    parser.add_argument("--output", type=str, default=str(MODELS_DIR / "model_int8.onnx"))
    args = parser.parse_args()
    quantize(args.input, args.output)
