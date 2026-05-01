# Dockerfile for torchtpu-vllm.
# This Dockerfile builds an image for running vLLM on Google Cloud TPUs.
# It uses a multi-stage build to "siphon" the torch-tpu wheel from a source image
# and install it on top of a base image with necessary dependencies.

ARG ARTIFACT_SOURCE="us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torch-tpu:nightly-latest"
ARG BASE_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torch-tpu-base:nightly-latest"

# Stage 1: The Siphon
FROM $ARTIFACT_SOURCE AS artifact-source

# Stage 2: The Runtime
FROM $BASE_IMAGE

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Clear PYTHONPATH inherited from base image if any
ENV PYTHONPATH=""

WORKDIR /root/tpu_inference

RUN pip install uv

# Copy pyproject.toml to install dependencies
COPY pyproject.toml .

# Hide torch-tpu to allow resolution of other dependencies
RUN sed -i '/torch-tpu==/d' pyproject.toml

ARG VLLM_SOURCE
# Copy vllm source (might be empty if not provided)
COPY vllm /root/tpu_inference/vllm

# Install vllm from source if requested
RUN if [ -n "$VLLM_SOURCE" ]; then \
      sed -i 's/vllm==[^"]*/vllm/' pyproject.toml && \
      VLLM_TARGET_DEVICE="tpu" uv pip install --system -e /root/tpu_inference/vllm ; \
    fi

# Install dependencies directly from pyproject.toml
# We use cache mount for uv to speed up downloads!
# This layer will be cached!
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    uv pip install --system --no-config --index-url https://pypi.org/simple .

# NOW pull the fresh wheel from the other project's image
# This is the cache-busting step!
COPY --from=artifact-source /opt/wheels/*.whl /tmp/

# Install the fresh torch-tpu wheel without dependencies (they are already installed)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps --no-config --index-url https://pypi.org/simple /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Copy the rest of the source code (restores pyproject.toml)
COPY . .

# Final install of the project
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    uv pip install --system --no-deps --no-config .

ENV PYTHONPATH="/root/tpu_inference"

CMD ["/bin/bash"]
