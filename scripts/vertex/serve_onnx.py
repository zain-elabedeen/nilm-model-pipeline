#!/usr/bin/env python3
"""Vertex custom prediction server for ONNX NILM models."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


def _download_artifact_dir(gcs_uri: str, local_dir: Path) -> None:
    """Download files from a gs://bucket/prefix path into local_dir."""
    from google.cloud import storage

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gcs_uri}")

    path = gcs_uri.removeprefix("gs://")
    bucket_name, _, prefix = path.partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for blob in client.list_blobs(bucket, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        rel_name = blob.name[len(prefix) :].lstrip("/") if prefix else blob.name
        target = local_dir / rel_name
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(target)


def _resolve_artifact_paths() -> tuple[Path, Path]:
    """Resolve local paths for ONNX model and metadata."""
    onnx_path_env = os.getenv("MODEL_ONNX_PATH")
    metadata_path_env = os.getenv("MODEL_METADATA_PATH")
    storage_uri = os.getenv("AIP_STORAGE_URI")

    if onnx_path_env and metadata_path_env:
        return Path(onnx_path_env), Path(metadata_path_env)

    if storage_uri:
        local_dir = Path(tempfile.mkdtemp(prefix="onnx-model-"))
        _download_artifact_dir(storage_uri, local_dir)
    else:
        local_dir = Path(os.getenv("MODEL_LOCAL_DIR", "/app/model"))

    onnx_name = os.getenv("MODEL_ONNX_FILENAME", "nilm_colab.onnx")
    metadata_name = os.getenv("MODEL_METADATA_FILENAME", "nilm_colab_metadata.json")
    return local_dir / onnx_name, local_dir / metadata_name


def _load_model_bundle() -> tuple[ort.InferenceSession, dict[str, Any], list[str]]:
    onnx_path, metadata_path = _resolve_artifact_paths()
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    appliance_order = list(metadata["outputs"].keys())
    providers_env = os.getenv("ONNX_PROVIDERS", "CPUExecutionProvider")
    providers = [provider.strip() for provider in providers_env.split(",") if provider.strip()]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    return session, metadata, appliance_order


def _normalize_window(values: list[float], metadata: dict[str, Any]) -> np.ndarray:
    window_size = int(metadata["window_size"])
    if len(values) != window_size:
        raise ValueError(f"Expected {window_size} values, got {len(values)}.")
    input_meta = metadata["input"]
    x = np.asarray(values, dtype=np.float32)
    x_norm = ((x - input_meta["mean"]) / input_meta["std"]).astype(np.float32)
    return x_norm.reshape(1, window_size)


def _denormalize_output(
    y_norm: np.ndarray, metadata: dict[str, Any], appliance_order: list[str]
) -> dict[str, float]:
    outputs = metadata["outputs"]
    result: dict[str, float] = {}
    for idx, appliance in enumerate(appliance_order):
        params = outputs[appliance]
        result[appliance] = float(y_norm[idx] * params["std"] + params["mean"])
    return result


class PredictRequest(BaseModel):
    instances: list[Any]


app = FastAPI(title="NILM ONNX Vertex Predictor")
SESSION, METADATA, APPLIANCE_ORDER = _load_model_bundle()
INPUT_NAME = SESSION.get_inputs()[0].name
OUTPUT_NAME = SESSION.get_outputs()[0].name


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    if not request.instances:
        raise HTTPException(status_code=400, detail="instances cannot be empty")

    predictions: list[dict[str, float]] = []
    for instance in request.instances:
        if isinstance(instance, dict):
            values = (
                instance.get("window")
                or instance.get("inputs")
                or instance.get("values")
            )
        else:
            values = instance

        if not isinstance(values, list):
            raise HTTPException(
                status_code=400,
                detail="Each instance must be a list[float] or dict with window/inputs/values.",
            )

        try:
            x = _normalize_window(values, METADATA)
            y_norm = SESSION.run([OUTPUT_NAME], {INPUT_NAME: x})[0][0]
            prediction = _denormalize_output(y_norm, METADATA, APPLIANCE_ORDER)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        predictions.append(prediction)

    return {"predictions": predictions}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
