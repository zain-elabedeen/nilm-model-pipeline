#!/usr/bin/env python
"""Run ONNX NILM inference from a single aggregate window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _parse_window_values(raw: str) -> np.ndarray:
    """Parse comma/space/newline separated floats into a 1D numpy array."""
    normalized = raw.replace("\n", " ").replace(",", " ")
    tokens = [token for token in normalized.split(" ") if token.strip()]
    if not tokens:
        raise ValueError("No numeric values found in input window.")
    try:
        values = np.array([float(token) for token in tokens], dtype=np.float32)
    except ValueError as exc:
        raise ValueError("Input window contains non-numeric values.") from exc
    return values


def _load_window(args: argparse.Namespace, expected_window_size: int) -> np.ndarray:
    """Load input window from file or inline CLI value."""
    if bool(args.window_file) == bool(args.window_values):
        raise ValueError("Provide exactly one of --window-file or --window-values.")

    if args.window_file:
        raw = Path(args.window_file).read_text()
    else:
        raw = args.window_values

    values = _parse_window_values(raw)
    if values.shape[0] != expected_window_size:
        raise ValueError(
            f"Expected {expected_window_size} values, got {values.shape[0]}."
        )
    return values


def _denormalize_outputs(
    y_norm: np.ndarray,
    output_meta: dict,
    appliance_order: list[str],
) -> dict[str, float]:
    """Map normalized output vector to denormalized appliance predictions."""
    output = {}
    for i, name in enumerate(appliance_order):
        params = output_meta[name]
        output[name] = float(y_norm[i] * params["std"] + params["mean"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ONNX NILM inference for one window")
    parser.add_argument("--onnx", required=True, help="Path to exported .onnx model")
    parser.add_argument("--metadata", required=True, help="Path to *_metadata.json")
    parser.add_argument(
        "--window-file",
        type=str,
        help="Text file containing exactly window_size values (comma/space/newline separated)",
    )
    parser.add_argument(
        "--window-values",
        type=str,
        help="Inline values string (comma/space/newline separated)",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        help="Optional path to save denormalized predictions as JSON",
    )
    args = parser.parse_args()

    metadata = json.loads(Path(args.metadata).read_text())
    window_size = int(metadata["window_size"])
    input_meta = metadata["input"]
    output_meta = metadata["outputs"]
    appliance_order = list(output_meta.keys())

    raw_window = _load_window(args, expected_window_size=window_size)

    x_norm = ((raw_window - input_meta["mean"]) / input_meta["std"]).astype(np.float32)
    x_norm = x_norm.reshape(1, window_size)

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    y_norm = session.run([output_name], {input_name: x_norm})[0][0]

    denorm = _denormalize_outputs(y_norm, output_meta, appliance_order)

    print(f"input_shape={tuple(x_norm.shape)} output_shape={(1, len(appliance_order))}")
    print("\nDenormalized predictions:")
    for name in appliance_order:
        print(f"  {name:<14} {denorm[name]:>12.4f}")

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.write_text(json.dumps(denorm, indent=2))
        print(f"\nSaved predictions JSON to {output_path}")


if __name__ == "__main__":
    main()
