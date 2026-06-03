# Docker images

This folder contains the Dockerfile used to build the main image variants for torchtpu-vllm.

## Image targets

### `ci:latest`
CI image target built from `--target ci`.

Use this image for CI and test infrastructure. It includes extra tooling on top of the base runtime, such as:

- `git`
- `google-cloud-cli`
- `gcsfuse`
- `lsof`
- `psmisc` / `fuser`

> [!NOTE]
> This image does not install the `vllm` project itself, only the dependencies and tools needed for CI.

Example build using the standard helper script:

```bash
./docker/build_image.sh --target ci -t ci:latest
```

### `dev:latest`
Developer image target built from `--target dev`.

This image is based on the `ci` stage, copies the repository into the image, and installs the project in editable mode with test and benchmarking extras:

- editable install: `-e ".[test,benchmarking]"`
- intended for local development, debugging, and running tests

Example build using the standard helper script:

```bash
./docker/build_image.sh --target dev -t dev:latest
```

### `prod:latest`
Production image target built from `--target prod`.

This image is intended for regular runtime usage. It performs a standard install of the project and is the best target for production-style execution.

Example build using the standard helper script:

```bash
./docker/build_image.sh --target prod -t prod:latest
```

## Useful build arguments

The Dockerfile also supports a few build arguments that may be useful in advanced workflows:

- `ARTIFACT_SOURCE`: source image used to copy prebuilt `torch-tpu` wheels
- `BASE_IMAGE`: base runtime image
- `USE_TORCH_TPU_REGISTRY`: when set, installs `torch-tpu` from the registry instead of the artifact image
- `TORCH_TPU_VERSION`: explicit `torch-tpu` version override
- `VLLM_SOURCE`: when set, installs `vllm` from the copied local source tree

Example using the helper script with build arguments:

```bash
./docker/build_image.sh \
  --target prod \
  --torch-tpu-registry \
  -t prod:latest
```
