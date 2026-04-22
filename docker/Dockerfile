# Base image: torch_tpu nightly from Google Artifact Registry.
# The registry requires authentication to pull:
#
#   TOKEN=$(gcloud auth print-access-token)
#   echo $TOKEN | docker login -u oauth2accesstoken --password-stdin us-docker.pkg.dev
#   docker build -f docker/Dockerfile -t <your-tag> .
#
# Meta-internal registry (used when we can't push to the public registry):
# ARG BASE_IMAGE="us-central1-docker.pkg.dev/meta-tpu-rbbq5/zongwei-repo/torchtpu:colin2328-latest"
ARG BASE_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torch-tpu:nightly-latest"
ARG VLLM_COMMIT_HASH="v0.17.1"

FROM $BASE_IMAGE

# Clear PYTHONPATH=/workspace inherited from base image — it causes
# /workspace dirs to shadow installed packages.
ENV PYTHONPATH=""

# System libraries needed by vLLM/torchtpu-vllm
RUN apt-get update && apt-get install -y \
    libopenblas0 libopenmpi-dev libomp-dev

RUN pip install uv

# Install vLLM from source with TPU support
WORKDIR /root/vllm
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_COMMIT_HASH
RUN git clone --depth 1 --branch $VLLM_COMMIT_HASH $VLLM_REPO /root/vllm
RUN sed -i '/tpu-inference/d' requirements/tpu.txt && \
    uv pip install --system -r requirements/tpu.txt
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VLLM_COMMIT_HASH#v} VLLM_TARGET_DEVICE="tpu" \
    uv pip install --system -e .

# Install tpu_inference and remaining dependencies via pyproject.toml.
# torch-tpu and vllm are already installed above; uv will skip them.
WORKDIR /root/tpu_inference
COPY . .
# --no-config skips [[tool.uv.index]] in pyproject.toml (private registry
# requires auth not available during docker build).
# SETUPTOOLS_SCM_PRETEND_VERSION is needed because .git is absent in the
# Docker context, so setuptools-scm cannot infer the version.
RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    uv pip install --system --no-cache --no-config -e ".[test,benchmarking]"

CMD ["/bin/bash"]
