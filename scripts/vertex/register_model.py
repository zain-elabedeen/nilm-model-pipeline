#!/usr/bin/env python3
"""Upload ONNX artifacts to GCS and optionally register a Vertex AI model.

This script is intended for Google Colab / Colab Enterprise workflows.

Example:
    python scripts/vertex/register_model.py \
      --project-id my-project \
      --region europe-west3 \
      --bucket my-vertex-artifacts \
      --onnx-path exports/nilm_colab.onnx \
      --metadata-path exports/nilm_colab_metadata.json \
      --display-name nilm-atcn \
      --serving-container-image-uri europe-west3-docker.pkg.dev/my-project/edge-serving/onnx-runtime:latest
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path


def _load_storage_lib() -> object:
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise SystemExit(
            "Missing Google Cloud Storage dependency. In Colab run:\n"
            "  pip install google-cloud-storage"
        ) from exc
    return storage


def _load_aiplatform_lib() -> object:
    try:
        from google.cloud import aiplatform
    except ImportError as exc:
        raise SystemExit(
            "Missing Vertex AI dependency. In Colab run:\n"
            "  pip install google-cloud-aiplatform"
        ) from exc
    return aiplatform


def _resolve_project_id(project_id: str | None) -> str:
    if project_id:
        return project_id

    for env_name in ("GOOGLE_CLOUD_PROJECT", "GCP_PROJECT", "PROJECT_ID"):
        env_value = os.getenv(env_name)
        if env_value:
            return env_value

    raise SystemExit(
        "Could not resolve project id. Pass --project-id or set GOOGLE_CLOUD_PROJECT."
    )


def _normalize_bucket_name(bucket: str) -> str:
    return bucket.removeprefix("gs://").strip("/")


def _parse_labels(raw_labels: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for raw in raw_labels:
        if "=" not in raw:
            raise SystemExit(f"Invalid label '{raw}'. Use key=value format.")
        key, value = raw.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise SystemExit(f"Invalid label '{raw}'. Empty key or value.")
        labels[key] = value
    return labels


def _safe_name(name: str) -> str:
    lowered = name.lower().strip()
    safe = re.sub(r"[^a-z0-9-]+", "-", lowered)
    safe = re.sub(r"-+", "-", safe).strip("-")
    return safe or "model"


def _upload_artifacts(
    storage: object,
    *,
    project_id: str,
    bucket_name: str,
    gcs_prefix: str,
    display_name: str,
    onnx_path: Path,
    metadata_path: Path | None,
) -> tuple[str, list[str]]:
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = f"{gcs_prefix.strip('/')}/{_safe_name(display_name)}/{timestamp}"

    uploaded_uris: list[str] = []

    files_to_upload: list[Path] = [onnx_path]
    if metadata_path is not None:
        files_to_upload.append(metadata_path)

    for local_file in files_to_upload:
        blob = bucket.blob(f"{run_dir}/{local_file.name}")
        blob.upload_from_filename(str(local_file))
        uploaded_uris.append(f"gs://{bucket_name}/{blob.name}")

    artifact_uri = f"gs://{bucket_name}/{run_dir}"
    return artifact_uri, uploaded_uris


def register_uploaded_model(
    *,
    project_id: str | None,
    region: str,
    artifact_uri: str,
    display_name: str,
    serving_container_image_uri: str,
    description: str = "",
    labels: dict[str, str] | None = None,
    predict_route: str = "/predict",
    health_route: str = "/health",
    serving_port: int = 8080,
    parent_model: str | None = None,
    version_aliases: list[str] | None = None,
    set_default_version: bool = False,
) -> object:
    """Register an already-uploaded model artifact in Vertex AI."""
    resolved_project_id = _resolve_project_id(project_id)
    aiplatform = _load_aiplatform_lib()
    aiplatform.init(project=resolved_project_id, location=region)

    upload_kwargs: dict[str, object] = {
        "display_name": display_name,
        "artifact_uri": artifact_uri,
        "serving_container_image_uri": serving_container_image_uri,
        "serving_container_predict_route": predict_route,
        "serving_container_health_route": health_route,
        "serving_container_ports": [serving_port],
    }

    if description:
        upload_kwargs["description"] = description
    if labels:
        upload_kwargs["labels"] = labels
    if parent_model:
        upload_kwargs["parent_model"] = parent_model
    if version_aliases:
        upload_kwargs["version_aliases"] = version_aliases
    if set_default_version:
        upload_kwargs["is_default_version"] = True

    return aiplatform.Model.upload(**upload_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload ONNX artifacts to GCS and register a Vertex AI model."
    )
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--bucket", default=None, help="GCS bucket name or gs://bucket")
    parser.add_argument(
        "--gcs-prefix",
        default="vertex-model-artifacts",
        help="Folder prefix inside the bucket",
    )

    parser.add_argument("--artifact-uri", default=None, help="Existing gs:// path with model files")
    parser.add_argument("--onnx-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)

    parser.add_argument("--display-name", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--label", action="append", default=[], help="key=value")
    parser.add_argument(
        "--setup-vertex-model",
        action="store_true",
        default=False,
        help="Also register the uploaded artifacts in Vertex AI Model Registry",
    )

    parser.add_argument(
        "--serving-container-image-uri",
        default=None,
        help="Container image used later for Vertex Endpoint deployment",
    )
    parser.add_argument("--serving-port", type=int, default=8080)
    parser.add_argument("--predict-route", default="/predict")
    parser.add_argument("--health-route", default="/health")

    parser.add_argument("--parent-model", default=None)
    parser.add_argument("--version-alias", action="append", default=[])
    parser.add_argument("--set-default-version", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_id = _resolve_project_id(args.project_id)
    labels = _parse_labels(args.label)

    storage = _load_storage_lib()

    if args.artifact_uri:
        if not args.artifact_uri.startswith("gs://"):
            raise SystemExit("--artifact-uri must start with gs://")
        artifact_uri = args.artifact_uri
        print(f"Using existing artifact URI: {artifact_uri}")
    else:
        if not args.bucket:
            raise SystemExit("Provide --bucket when --artifact-uri is not set.")
        if args.onnx_path is None:
            raise SystemExit("Provide --onnx-path when --artifact-uri is not set.")
        if not args.onnx_path.exists():
            raise SystemExit(f"ONNX file not found: {args.onnx_path}")
        if args.metadata_path is not None and not args.metadata_path.exists():
            raise SystemExit(f"Metadata file not found: {args.metadata_path}")

        bucket_name = _normalize_bucket_name(args.bucket)
        print("Uploading model artifacts to GCS...")
        artifact_uri, uploaded_uris = _upload_artifacts(
            storage,
            project_id=project_id,
            bucket_name=bucket_name,
            gcs_prefix=args.gcs_prefix,
            display_name=args.display_name,
            onnx_path=args.onnx_path,
            metadata_path=args.metadata_path,
        )
        for uri in uploaded_uris:
            print(f"  uploaded: {uri}")
        print(f"Artifact URI: {artifact_uri}")

    if not args.setup_vertex_model:
        print("Skipping Vertex AI Model Registry registration.")
        return

    if not args.serving_container_image_uri:
        raise SystemExit(
            "Provide --serving-container-image-uri when --setup-vertex-model is enabled."
        )

    print("Registering model in Vertex AI Model Registry...")
    model = register_uploaded_model(
        project_id=project_id,
        region=args.region,
        artifact_uri=artifact_uri,
        display_name=args.display_name,
        serving_container_image_uri=args.serving_container_image_uri,
        description=args.description,
        labels=labels,
        predict_route=args.predict_route,
        health_route=args.health_route,
        serving_port=args.serving_port,
        parent_model=args.parent_model,
        version_aliases=args.version_alias,
        set_default_version=args.set_default_version,
    )

    print("Model registered.")
    print(f"Model resource name: {model.resource_name}")
    print(
        "Vertex Console URL: "
        f"https://console.cloud.google.com/vertex-ai/locations/{args.region}/models?project={project_id}"
    )

if __name__ == "__main__":
    main()
