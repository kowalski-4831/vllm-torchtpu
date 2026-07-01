# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline embedding inference
on Qwen3-VL-Embedding models on TPU and optionally verify numerical accuracy
against a CPU reference baseline.

Example Command:
python examples/qwen3_vl_vllm_embedding.py \
  --model Qwen/Qwen3-VL-Embedding-8B \
  --tensor-parallel-size 1 \
  --verify-accuracy
"""

import multiprocessing as mp

import numpy as np
import torch

# --- BOOTSTRAP ---
# Satisfaction of transformers' torchvision dependency is handled via stubs
# if real torchvision is missing.
from vllm_torchtpu.utils import bootstrap_torchvision_stubs

bootstrap_torchvision_stubs()

# --- TPU MODEL REGISTRATION ---
from vllm.model_executor.models.registry import ModelRegistry  # noqa: E402

try:
    from vllm.model_executor.models.transformers import \
        TransformersMultiModalEmbeddingModel  # noqa: E402

    # Use generic TransformersMultiModalEmbeddingModel for embedding tasks
    # The TPU-specific Qwen3-VL patches are handled automatically in TpuPlatform.
    ModelRegistry.register_model("Qwen3VLForConditionalGeneration",
                                 TransformersMultiModalEmbeddingModel)
except ImportError:
    pass

# --- MAIN ---
from vllm import LLM  # noqa: E402
from vllm.config import PoolerConfig  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402


def parse_args():
    parser = FlexibleArgumentParser(
        description="Qwen3-VL Embedding Inference with vLLM on TPU")
    parser.add_argument("--model",
                        type=str,
                        default="Qwen/Qwen3-VL-Embedding-8B",
                        help="Model name or path")
    parser.add_argument("--tensor-parallel-size",
                        "-tp",
                        type=int,
                        default=1,
                        help="TP degree")
    parser.add_argument("--verify-accuracy",
                        action="store_true",
                        default=True,
                        help="Verify accuracy against CPU reference model")
    return parser.parse_args()


def _run_cpu_reference(model_name, prompt, return_dict):
    """Computes CPU reference embedding in an isolated process."""
    try:
        from transformers import AutoModel, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name,
                                                  trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name,
                                          trust_remote_code=True,
                                          torch_dtype=torch.bfloat16).to("cpu")
        model.eval()

        inputs = processor(text=[prompt], return_tensors="pt")
        inputs.pop("attention_mask", None)

        with torch.no_grad():
            outputs = model(**inputs,
                            output_hidden_states=True,
                            return_dict=True)

        hidden = (outputs.last_hidden_state if hasattr(
            outputs, "last_hidden_state") else outputs.hidden_states[-1])
        emb = hidden[0, -1, :].float().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return_dict["cpu_emb"] = emb
    except Exception as e:
        print(f"CPU Reference Error: {e}")


def compare_embeddings(emb_cpu, emb_tpu):
    print("\n--- Accuracy Verification Results (CPU vs TPU vLLM) ---")
    print(f"CPU First 10 dims: {emb_cpu[:10]}")
    print(f"TPU First 10 dims: {emb_tpu[:10]}")

    dot_product = np.dot(emb_cpu, emb_tpu)
    norm_cpu = np.linalg.norm(emb_cpu)
    norm_tpu = np.linalg.norm(emb_tpu)
    cosine_sim = dot_product / (norm_cpu * norm_tpu)

    l2_dist = np.linalg.norm(emb_cpu - emb_tpu)
    rel_l2_dist = l2_dist / norm_cpu

    print(f"Cosine Similarity: {cosine_sim:.6f}")
    print(f"L2 Distance (Abs): {l2_dist:.6f}")
    print(f"L2 Distance (Rel): {rel_l2_dist:.6f}")

    if cosine_sim > 0.99:
        print(
            "\n✅ SUCCESS: TPU embeddings are highly accurate (Similarity > 0.99)"
        )
    else:
        print("\n❌ WARNING: Significant numerical divergence detected.")


def main(args):
    prompt = ("<|im_start|>user\n"
              "What is the capital of France?"
              "<|im_end|>\n"
              "<|im_start|>assistant\n")

    print(f"Initializing vLLM on TPU for model {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        pooler_config=PoolerConfig(task="embed"),
        trust_remote_code=True,
        max_model_len=4096,
        max_num_batched_tokens=16384,
        enforce_eager=True,
        disable_log_stats=True,
        disable_chunked_mm_input=True,
    )

    print("Running vLLM TPU embedding inference...")
    outputs = llm.embed({"prompt": prompt})

    tpu_emb = np.array(outputs[0].outputs.embedding, dtype=np.float32)
    print("\nTPU Request 0 Embedding:")
    print(f"- Shape: {len(tpu_emb)}")
    print(f"- First 10 dims: {tpu_emb[:10]}")

    if args.verify_accuracy:
        print("\nComputing CPU reference embedding for accuracy check...")
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        return_dict = manager.dict()

        p_cpu = ctx.Process(target=_run_cpu_reference,
                            args=(args.model, prompt, return_dict))
        p_cpu.start()
        p_cpu.join()

        if "cpu_emb" in return_dict:
            compare_embeddings(return_dict["cpu_emb"], tpu_emb)
        else:
            print("Could not retrieve CPU reference embedding.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
