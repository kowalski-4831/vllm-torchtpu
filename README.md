<h1 align="center">vLLM TPU</h1>

This repository contains the integration of **TorchTPU** and **vLLM**. The codebase references both vLLM main and `tpu-inference`, utilizing the vLLM framework for the server logic and custom kernels from `tpu-inference` for optimized performance.

---

## 🛠️ Installation

To run the **Qwen3** model on a single device, follow these installation steps:

### 1. Install TorchTPU-vLLM and dependencies

We recommend using `uv` for installing dependencies as it is significantly faster than standard `pip`.

#### Option A: Using `uv` (Recommended)

```bash
uv venv --python 3.12
source .venv/bin/activate

# Set up authentication for the Torch TPU virtual registry
export UV_INDEX_TORCH_TPU_REGISTRY_USERNAME="oauth2accesstoken"
export UV_INDEX_TORCH_TPU_REGISTRY_PASSWORD="$(gcloud auth print-access-token)"

# Clone vLLM to allow making local patches for debugging
git clone --depth 1 --branch v0.17.1 https://github.com/vllm-project/vllm.git ../vllm

# Patch vLLM's TPU requirements to accept our local workspace instead of overriding it
sed -i 's/tpu-inference==0.12.0/tpu-inference/' ../vllm/requirements/tpu.txt

# Install vLLM in editable mode (forcing the 0.17.1 base version to prevent .dev prerelease mismatch during dependency resolution)
SETUPTOOLS_SCM_PRETEND_VERSION=0.17.1 VLLM_TARGET_DEVICE="tpu" uv pip install -e ../vllm

# Install TorchTPU-vLLM and dependencies
uv pip install --pre -e .
```

#### Option B: Using `pip`

> **Note:** Currently, `vllm==0.17.1` is supported.

```bash
python3.12 -m venv vllm_env --symlinks
source ../vllm_env/bin/activate
pip install --upgrade pip

# Set up authentication for the Torch TPU virtual registry
ACCESS_TOKEN="$(gcloud auth print-access-token)"

# Set the PIP_INDEX_URL environment variable to point to the Torch TPU virtual registry
PIP_INDEX_URL="https://oauth2accesstoken:${ACCESS_TOKEN}@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/"

# Clone vLLM to allow making local patches for debugging
git clone --depth 1 --branch v0.17.1 https://github.com/vllm-project/vllm.git ../vllm

# Patch vLLM's TPU requirements to accept our local workspace instead of overriding it
sed -i 's/tpu-inference==0.12.0/tpu-inference/' ../vllm/requirements/tpu.txt

# Install vLLM in editable mode (forcing the 0.17.1 base version to prevent .dev prerelease mismatch during dependency resolution)
SETUPTOOLS_SCM_PRETEND_VERSION=0.17.1 VLLM_TARGET_DEVICE="tpu" pip install -e ../vllm

# Install TorchTPU-vLLM and dependencies
pip install --pre -e .
```

> **Note:** Prioritize compile mode for better performance. Add `--enforce-eager` if you want eager mode.
> On TPUv7, Qwen3-Coder-30B can fit on a single device. On v6, use a smaller model like Qwen3-4B or test with TP/EP.

---

## 🌐 Online Serving

Start the server with the following command:

```bash
vllm serve "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
  --tensor_parallel_size=1 \
  --max-model-len=256 \
  --max-num-batched-tokens=256
```

### Send a Request

Once the server is running, you can verify it by sending a request:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "prompt": "Hello, my name is",
        "max_tokens": 20,
        "temperature": 0
    }'
```

### 📉 Running Offline Inference

You can also run a simple offline inference script to verify the setup without starting a full server.

#### Single Device

```bash
python3 examples/offline_inference.py \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --max-model-len 256 \
  --max-num-batched-tokens 256
```

#### Tensor Parallelism (TP)

```bash
python3 examples/offline_inference.py \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --max-model-len 256 \
  --max-num-batched-tokens 256 \
  --tensor_parallel_size=2
```

#### Expert Parallelism (EP)

```bash
python3 examples/offline_inference.py \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --max-model-len 256 \
  --max-num-batched-tokens 256 \
  --tensor_parallel_size=2 \
  --enable-expert-parallel
```
