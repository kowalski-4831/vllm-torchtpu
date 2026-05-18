# Local Reproduction and Debugging

If the CI (GitHub Actions) fails or you want to test changes locally before pushing, you can run the exact same evaluation flow on a development TPU VM.

## Prerequisites
- A Cloud TPU VM with Docker installed.
- Your code synced to the TPU VM (e.g., in `/mnt/pd/projects/torchtpu-vllm`).
- `gcloud` authenticated on the TPU VM or passed from host.
- An alias `tpu-vm-ssh` configured to SSH into your TPU VM (see below).

### Setting up `tpu-vm-ssh` Alias

```bash
alias tpu-vm-ssh="ssh <user_name>@tpu_ip_addr"
```

## Running the Full Flow in Docker

To simulate the CI environment as closely as possible, run the tests inside the designated CI Docker container on the TPU VM.

First, make sure you have the latest CI Docker image on your TPU VM. You can pull it by running:

```bash
ssh <TPU_VM_NAME> "gcloud auth configure-docker us-docker.pkg.dev && docker pull us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torchtpu-vllm-ci:latest"
```

### Testing Local Dockerfile Changes

If you make changes to the `Dockerfile` (e.g., adding packages like `gcsfuse`) and want to test them locally on the TPU VM:

1. **Build the image locally** on the TPU VM using the provided script:

   ```bash
   ssh <TPU_VM_NAME> "cd /mnt/pd/projects/torchtpu-vllm && ./docker/build_image.sh --target ci"
   ```

   This will build the `ci` stage and tag it as `torchtpu-vllm-local` by default.

> [!NOTE]
> Changes to the `Dockerfile` are only built and pushed to the remote registry during the nightly CI job. To test your changes immediately on the TPU VM, you must build the image locally and use it in the run command.

1. **Run the container** using your local image:
   In the `docker run` command below, replace the remote image `us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torchtpu-vllm-ci:latest` with your local image tag `torchtpu-vllm-local`.

Then, run the following command from your local machine (Cloudtop) to trigger the run on the TPU VM via SSH.

**Template:**

```bash
ssh <TPU_VM_NAME> "TOKEN=\$(gcloud auth print-access-token); docker run --rm --privileged --net=host --shm-size=16g \\
  -v /mnt/pd/projects/torchtpu-vllm:/root/tpu_inference \\
  -v /mnt/pd/.cache/huggingface:/root/.cache/huggingface \\
  -e UV_INDEX_TORCH_TPU_REGISTRY_USERNAME=oauth2accesstoken \\
  -e UV_INDEX_TORCH_TPU_REGISTRY_PASSWORD=\"\$TOKEN\" \\
  -e SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \\
  us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torchtpu-vllm-ci:latest \\
  bash -c 'cd /root/tpu_inference && uv pip install --system \".[test,benchmarking]\" && bash ./scripts/vllm/benchmarking/run_eval_flow.sh --config <CONFIG_NAME> --run-lm-eval'"
```

Replace `<TPU_VM_NAME>` with your TPU VM hostname and `<CONFIG_NAME>` with one of the available configs.

**Concrete Examples:**

**1. PR Guard: Qwen3-Coder-30B-A3B-Instruct-FP8**

```bash
tpu-vm-ssh "TOKEN=\$(gcloud auth print-access-token); docker run --rm --privileged --net=host --shm-size=16g \\
  -v /mnt/pd/projects/torchtpu-vllm:/root/tpu_inference \\
  -v /mnt/pd/.cache/huggingface:/root/.cache/huggingface \\
  -e UV_INDEX_TORCH_TPU_REGISTRY_USERNAME=oauth2accesstoken \\
  -e UV_INDEX_TORCH_TPU_REGISTRY_PASSWORD=\"\$TOKEN\" \\
  -e SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \\
  us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torchtpu-vllm-ci:latest \\
  bash -c 'cd /root/tpu_inference && uv pip install --system \".[test,benchmarking]\" && bash ./scripts/vllm/benchmarking/run_eval_flow.sh --config qwen3-coder-30b-fp8-tp8-ep --run-lm-eval'"
```

**2. PR Guard: Qwen3-Coder-30B-A3B-Instruct**

```bash
tpu-vm-ssh "TOKEN=\$(gcloud auth print-access-token); docker run --rm --privileged --net=host --shm-size=16g \\
  -v /mnt/pd/projects/torchtpu-vllm:/root/tpu_inference \\
  -v /mnt/pd/.cache/huggingface:/root/.cache/huggingface \\
  -e UV_INDEX_TORCH_TPU_REGISTRY_USERNAME=oauth2accesstoken \\
  -e UV_INDEX_TORCH_TPU_REGISTRY_PASSWORD=\"\$TOKEN\" \\
  -e SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \\
  us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torchtpu-vllm-ci:latest \\
  bash -c 'cd /root/tpu_inference && uv pip install --system \".[test,benchmarking]\" && bash ./scripts/vllm/benchmarking/run_eval_flow.sh --config qwen3-coder-30b-tp8-ep --run-lm-eval'"
```

**3. Nightly: Qwen3-Coder-480B-A35B-Instruct-FP8**

```bash
tpu-vm-ssh "TOKEN=\$(gcloud auth print-access-token); docker run --rm --privileged --net=host --shm-size=16g \\
  -v /mnt/pd/projects/torchtpu-vllm:/root/tpu_inference \\
  -v /mnt/pd/.cache/huggingface:/root/.cache/huggingface \\
  -e UV_INDEX_TORCH_TPU_REGISTRY_USERNAME=oauth2accesstoken \\
  -e UV_INDEX_TORCH_TPU_REGISTRY_PASSWORD=\"\$TOKEN\" \\
  -e SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \\
  -e EVALPLUS_DATASETS=\"humaneval mbpp\" \\
  -e EVALPLUS_PARALLEL=8 \\
  us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torchtpu-vllm-ci:latest \\
  bash -c 'cd /root/tpu_inference && uv pip install --system \".[test,benchmarking]\" && bash ./scripts/vllm/benchmarking/run_eval_flow.sh --config qwen3-coder-480b-fp8-tp8-ep --run-lm-eval --run-evalplus'"
```

**4. Unit Tests**

```bash
tpu-vm-ssh "TOKEN=\$(gcloud auth print-access-token); docker run --rm --privileged --net=host --shm-size=16g \\
  -v /mnt/pd/projects/torchtpu-vllm:/root/tpu_inference \\
  -v /mnt/pd/.cache/huggingface:/root/.cache/huggingface \\
  -e UV_INDEX_TORCH_TPU_REGISTRY_USERNAME=oauth2accesstoken \\
  -e UV_INDEX_TORCH_TPU_REGISTRY_PASSWORD=\"\$TOKEN\" \\
  -e SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \\
  us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torchtpu-vllm-ci:latest \\
  bash -c 'cd /root/tpu_inference && bash ./scripts/vllm/benchmarking/run_unit_test.sh false'"
```

## Script Details

### `run_eval_flow.sh`
This is the unified script used by both CI and manual runs. It handles:
1. Initial cleanup of any leaked servers.
2. Starting the vLLM server and running performance benchmarks.
3. Running `lm_eval` (if `--run-lm-eval` is passed).
4. Running `evalplus` (if `--run-evalplus` is passed).
5. Final cleanup.
6. Regression checks against baselines.

### `cleanup_server.sh`
A robust script to kill any zombie vLLM servers and free up TPU resources (`/dev/vfio/*`) and port 8000. It is called automatically by `run_eval_flow.sh` but can be run manually if needed. It includes a pure Bash fallback to scan `/proc` if tools like `lsof` or `fuser` are missing in the container.
