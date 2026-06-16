# Dockerfile for torchtpu-vllm.
# This Dockerfile builds an image for running vLLM on Google Cloud TPUs.
# By default it installs torch-tpu from an artifact image.
# With USE_TORCH_TPU_REGISTRY set, it installs torch-tpu from the private registry,
# using TORCH_TPU_VERSION or the torch-tpu pin in pyproject.toml.

ARG ARTIFACT_SOURCE="us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torch-tpu:nightly-latest"
ARG BASE_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torch-tpu-base:nightly-latest"

# Stage 1: The Siphon
FROM $ARTIFACT_SOURCE AS artifact-source

# Stage 2: The Runtime
FROM $BASE_IMAGE AS base

ARG USE_TORCH_TPU_REGISTRY=""
ARG TORCH_TPU_VERSION=""

# Install curl for health checks (minimal for base)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Stage 2.5: The CI Environment (with tools)
FROM base AS ci

# Install tools needed for CI (git, gcloud, lsof, fuser)
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg git lsof psmisc lsb-release && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -c -s) main" | tee -a /etc/apt/sources.list.d/gcsfuse.list && \
    apt-get update && apt-get install -y google-cloud-cli gcsfuse && \
    rm -rf /var/lib/apt/lists/*

# Clear PYTHONPATH inherited from base image if any
ENV PYTHONPATH=""

WORKDIR /root/torchtpu-vllm

# Mark mounted directory as safe for git
RUN git config --global --add safe.directory /root/torchtpu-vllm
RUN mkdir -p /workspace && ln -sfnT /root/torchtpu-vllm /workspace/torchtpu-vllm

RUN pip install uv

# Copy pyproject.toml to install dependencies
COPY pyproject.toml .

# Hide torch-tpu to allow resolution of other dependencies.
RUN cp pyproject.toml /tmp/pyproject.toml && sed -i '/torch-tpu==/d' pyproject.toml

ARG VLLM_SOURCE
# Copy vllm source (might be empty if not provided)
COPY vllm /root/torchtpu-vllm/vllm

# Install vllm from source if requested
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=gcloud_token \
    if [ -n "$VLLM_SOURCE" ]; then \
      TOKEN=""; \
      if [ -f /run/secrets/gcloud_token ]; then TOKEN="$(cat /run/secrets/gcloud_token)"; fi; \
      sed -i 's/vllm==[^"]*/vllm/' pyproject.toml && \
      VLLM_TARGET_DEVICE="tpu" uv pip install --system \
        --extra-index-url "https://oauth2accesstoken:${TOKEN}@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" \
        -e /root/torchtpu-vllm/vllm ; \
    fi

# Install dependencies directly from pyproject.toml
# We use cache mount for uv to speed up downloads!
# This layer will be cached!
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    uv pip install --system --no-config --index-url https://pypi.org/simple -r pyproject.toml

# Install torch-tpu either from a fresh artifact image or the pinned registry version.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=gcloud_token \
    --mount=from=artifact-source,source=/,target=/artifact-source,ro \
    set -eu; \
    if [ -z "${USE_TORCH_TPU_REGISTRY}" ]; then \
        set -- /artifact-source/opt/wheels/*.whl; \
        if [ ! -e "$1" ]; then echo "No torch-tpu wheels found in ARTIFACT_SOURCE" >&2; exit 1; fi; \
        uv pip install --system --no-deps --no-config --index-url https://pypi.org/simple "$@"; \
    else \
        if [ -z "${TORCH_TPU_VERSION}" ]; then TORCH_TPU_VERSION="$(sed -nE 's/^[[:space:]]*"torch-tpu==([^"]+)".*/\1/p' /tmp/pyproject.toml | head -n1)"; fi; \
        if [ -z "${TORCH_TPU_VERSION}" ]; then echo "Could not resolve torch-tpu version from pyproject.toml" >&2; exit 1; fi; \
        TOKEN=""; \
        if [ -f /run/secrets/gcloud_token ]; then TOKEN="$(cat /run/secrets/gcloud_token)"; fi; \
        if [ -z "${TOKEN}" ]; then echo "Missing BuildKit secret gcloud_token for registry mode" >&2; exit 1; fi; \
        uv pip install --system --no-deps --no-config \
          --index-url https://pypi.org/simple \
          --extra-index-url "https://oauth2accesstoken:${TOKEN}@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" \
          "torch-tpu==${TORCH_TPU_VERSION}"; \
    fi


# Stage 3: Developer Image (Editable)
FROM ci AS dev
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    uv pip install --system --no-config --index-url https://pypi.org/simple -e ".[test,benchmarking]"


ENV PYTHONPATH="/root/torchtpu-vllm/src:/root/torchtpu-vllm"
CMD ["/bin/bash"]

# Stage 4: Production/Full Image (Regular Install)
FROM base AS prod
WORKDIR /root/torchtpu-vllm
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=gcloud_token \
    set -eu; \
    TOKEN=""; \
    if [ -f /run/secrets/gcloud_token ]; then TOKEN="$(cat /run/secrets/gcloud_token)"; fi; \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    uv pip install --system --no-config \
      --index-url https://pypi.org/simple \
      --extra-index-url "https://oauth2accesstoken:${TOKEN}@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" \
      .

RUN mkdir -p /workspace && ln -sfnT /root/torchtpu-vllm /workspace/torchtpu-vllm
ENV PYTHONPATH="/root/torchtpu-vllm/src:/root/torchtpu-vllm"
CMD ["/bin/bash"]
