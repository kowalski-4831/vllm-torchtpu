<h1 align="center">vLLM TPU</h1>

This repository contains the integration of **TorchTPU** and **vLLM**. The codebase references both vLLM main and `tpu-inference`, utilizing the vLLM framework for the server logic and custom kernels from `tpu-inference` for optimized performance.

---

## 🛠️ Installation

To run the **Qwen3** model on a single device, follow these installation steps:

### 1. Install TorchTPU
Build the `torch_tpu` Python wheel from source by following the instructions here:
👉 [google-ml-infra/torch_tpu Installation Guide](https://github.com/google-ml-infra/torch_tpu?tab=readme-ov-fil#installation)

### 2. Install vLLM from Source
Install vLLM from source using the pinned upstream version used by CI.
> **Note:** Currently, `vllm==v0.17.1` is supported.

```bash
python3.12 -m venv ../vllm_env --symlinks
source ../vllm_env/bin/activate
pip install --upgrade pip

git clone --depth 1 --branch v0.17.1 https://github.com/vllm-project/vllm.git ../vllm

pip install -r ../vllm/requirements/tpu.txt
VLLM_TARGET_DEVICE="tpu" pip install -e ../vllm

ACCESS_TOKEN="$(gcloud auth print-access-token)"

# sometimes need to reinstall after vllm to make sure some dependencies are installed correctly
pip install --pre \
  --index-url "https://oauth2accesstoken:${ACCESS_TOKEN}@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" \
  torch_tpu

pip install -r requirements.txt
pip install -e .
```

---

## 🌐 Online Serving

To run the Qwen3 model on a single device, start the server with the following command:

### Start the Server

```bash
MODEL_IMPL_TYPE="vllm" vllm serve "Qwen/Qwen3-0.6B" \
     --download_dir /tmp \
     --disable-log-requests \
     --tensor_parallel_size=1 \
     --max-model-len=2048 \
     --enforce-eager
```

### Send a Request

Once the server is running, you can verify it by sending a request:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-0.6B",
        "prompt": "Hello, my name is",
        "max_tokens": 20,
        "temperature": 0.7
    }'
```

### 📉 Running Offline Inference

You can also run a simple offline inference script to verify the setup without starting a full server:

```bash
MODEL_IMPL_TYPE="vllm" python examples/offline_inference.py \
    --model Qwen/Qwen3-0.6B \
    --max-model-len 2048 \
    --enforce-eager
```
