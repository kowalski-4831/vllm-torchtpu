<h1 align="center">vLLM TPU</h1>

This repository contains the integration of **TorchTPU** and **vLLM**. The codebase references both vLLM main and `tpu-inference`, utilizing the vLLM framework for the server logic and custom kernels from `tpu-inference` for optimized performance.

---

## 🛠️ Installation

To run the **Qwen3** model on a single device, follow these installation steps:

### 1. Install TorchTPU
Build the `torch_tpu` Python wheel from source by following the instructions here:
👉 [google-ml-infra/torch_tpu Installation Guide](https://github.com/google-ml-infra/torch_tpu?tab=readme-ov-file#build-from-source)

### 2. Install vLLM from Source
Install vLLM from source using the official TPU documentation. 
> **Note:** Currently, `vllm==v0.13.0` is supported.

👉 [vLLM TPU Installation Guide](https://docs.vllm.ai/projects/tpu/en/latest/getting_started/installation/#install-from-source)

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
