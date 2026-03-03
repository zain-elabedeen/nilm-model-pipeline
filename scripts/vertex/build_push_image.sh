#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${SCRIPT_DIR}/defaults.env" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/defaults.env"
fi

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "PROJECT_ID is not set. Export PROJECT_ID or run: gcloud config set project <project-id>" >&2
  exit 1
fi

REGION="${REGION:-${VERTEX_REGION:-us-central1}}"
AR_REPO="${AR_REPO:-vertex-training}"
IMAGE_NAME="${IMAGE_NAME:-edge-model}"
ENABLE_APIS="${ENABLE_APIS:-true}"

if [[ -n "${IMAGE_TAG:-}" ]]; then
  TAG="${IMAGE_TAG}"
elif git -C "${REPO_ROOT}" rev-parse --short HEAD >/dev/null 2>&1; then
  TAG="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
else
  TAG="$(date +%Y%m%d-%H%M%S)"
fi

IMAGE_REPO="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}"
IMAGE_URI="${IMAGE_REPO}:${TAG}"
LATEST_URI="${IMAGE_REPO}:latest"

if [[ "${ENABLE_APIS}" == "true" ]]; then
  gcloud services enable \
    aiplatform.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    --project "${PROJECT_ID}"
fi

if ! gcloud artifacts repositories describe "${AR_REPO}" \
  --location="${REGION}" \
  --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${AR_REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --description="Vertex AI training images"
fi

gcloud builds submit "${REPO_ROOT}" \
  --project="${PROJECT_ID}" \
  --config="${REPO_ROOT}/cloudbuild.vertex.yaml" \
  --substitutions="_IMAGE_REPO=${IMAGE_REPO},_IMAGE_TAG=${TAG}"
wha
echo "Pushed images:"
echo "  ${IMAGE_URI}"
echo "  ${LATEST_URI}"
