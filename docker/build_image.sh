#!/bin/bash
set -e

# Usage Examples:
#
# 1. Build with default values:
#    ./docker/build_image.sh
#
# 2. Build with custom image tag and base image:
#    ./docker/build_image.sh -t my-vllm-image -b my-base-image
#
# 3. Build with local vLLM source:
#    ./docker/build_image.sh -s /path/to/vllm/source
#
# To build the prerequisite torch-tpu base image:
#    cd ../torch_tpu
#    bash docker/build_image_multistage.sh
#    cd -

# Default values
IMAGE_TAG="torchtpu-vllm-local"
BASE_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torch-tpu-base:nightly-latest"
ARTIFACT_SOURCE="us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torch-tpu:nightly-latest"
VLLM_SOURCE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--image-tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    -b|--base-image)
      BASE_IMAGE="$2"
      shift 2
      ;;
    -a|--artifact-source)
      ARTIFACT_SOURCE="$2"
      shift 2
      ;;
    -s|--vllm-source)
      VLLM_SOURCE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "===> Building torchtpu-vllm image..."
echo "Image Tag: $IMAGE_TAG"
echo "Base Image: $BASE_IMAGE"
echo "Artifact Source: $ARTIFACT_SOURCE"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../" && pwd)"

# Handle local vllm source directory
CLEANUP_VLLM=false
if [ -n "$VLLM_SOURCE" ] && [ -d "$VLLM_SOURCE" ]; then
  echo "===> Copying local vllm source from $VLLM_SOURCE..."
  rm -rf "${REPO_ROOT}/vllm"
  cp -r "$VLLM_SOURCE" "${REPO_ROOT}/vllm"
  CLEANUP_VLLM=true
else
  echo "===> Creating empty vllm directory for build context..."
  mkdir -p "${REPO_ROOT}/vllm"
  touch "${REPO_ROOT}/vllm/.dummy"
  CLEANUP_VLLM=true
fi

# Run docker build
docker build \
  --progress=plain \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_TAG}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg ARTIFACT_SOURCE="${ARTIFACT_SOURCE}" \
  --build-arg VLLM_SOURCE="${VLLM_SOURCE}" \
  "${REPO_ROOT}"

if [ "$CLEANUP_VLLM" = true ]; then
  echo "===> Cleaning up copied vllm source..."
  rm -rf "${REPO_ROOT}/vllm"
fi

echo "===> Done!"
