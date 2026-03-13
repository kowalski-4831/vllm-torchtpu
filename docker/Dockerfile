ARG BASE_IMAGE="us-central1-docker.pkg.dev/meta-tpu-rbbq5/zongwei-repo/torchtpu:colin2328-latest"
ARG VLLM_COMMIT_HASH="v0.17.1"

FROM $BASE_IMAGE

# Clear PYTHONPATH=/workspace inherited from base image — it causes
# /workspace dirs to shadow installed packages.
ENV PYTHONPATH=""

# System libraries needed by vLLM/torchtpu-vllm
RUN apt-get update && apt-get install -y \
    libopenblas0 libopenmpi-dev libomp-dev

RUN pip install uv

# Install vLLM under /root (not /workspace — base image sets PYTHONPATH=/workspace
# which would cause /workspace/vllm to shadow the installed vllm package)
WORKDIR /root/vllm
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_COMMIT_HASH
RUN git clone --depth 1 --branch $VLLM_COMMIT_HASH $VLLM_REPO /root/vllm
RUN sed -i '/tpu-inference/d' requirements/tpu.txt && \
    uv pip install --system -r requirements/tpu.txt
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VLLM_COMMIT_HASH#v} VLLM_TARGET_DEVICE="tpu" \
    uv pip install --system -e .

# Install test dependencies
RUN uv pip install --system -e tests/vllm_test_utils
RUN uv pip install --system --no-cache \
    git+https://github.com/thuml/depyf.git \
    pytest-asyncio \
    git+https://github.com/EleutherAI/lm-evaluation-harness.git@206b7722158f58c35b7ffcd53b035fdbdda5126d#egg=lm-eval[api] \
    pytest-cov tblib

# Install tpu_inference (this repo)
WORKDIR /workspace/tpu_inference
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
COPY requirements_benchmarking.txt .
RUN uv pip install --system -r requirements_benchmarking.txt
COPY . .
RUN uv pip install --system -e .

CMD ["/bin/bash"]
