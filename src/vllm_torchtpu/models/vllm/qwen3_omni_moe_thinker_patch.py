# SPDX-License-Identifier: Apache-2.0
"""Model-specific patches for Qwen3-Omni-MoE Thinker on TPU."""

import functools
import importlib
import inspect
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.qwen3_vl_patch import _scoped_qwen3_vl_torch_ops

if TYPE_CHECKING:
    from vllm.config import ModelConfig
else:
    ModelConfig = None

logger = init_logger(__name__)


def _is_qwen3_omni_thinker_model(model_config: Optional["ModelConfig"]) -> bool:
    """Check if the provided ModelConfig corresponds to a Qwen3-Omni model."""
    if model_config is None:
        return False
    hf_config = model_config.hf_config
    if hf_config is not None:
        if "qwen3_omni" in str(getattr(hf_config, "model_type", "")).lower():
            return True
        archs = getattr(hf_config, "architectures", [])
        if any("qwen3omni" in str(a).lower() for a in archs):
            return True
    model_name = str(getattr(model_config, "model", "")).lower()
    return "qwen3-omni" in model_name or "qwen3omni" in model_name


def _patch_qwen3_omni_vision_transformer(modeling) -> None:
    """Patch Qwen3Omni_VisionTransformer.forward to scope TPU torch ops."""
    target_cls = getattr(modeling, "Qwen3Omni_VisionTransformer", None)
    orig_fn = getattr(target_cls, "forward", None) if target_cls else None
    if orig_fn is None or getattr(orig_fn, "_tpu_qwen3_omni_vision_patch", False):
        return

    @functools.wraps(orig_fn)
    def patched_vision_forward(self, *args, **kwargs):
        with _scoped_qwen3_vl_torch_ops():
            return orig_fn(self, *args, **kwargs)

    patched_vision_forward._tpu_qwen3_omni_vision_patch = True
    target_cls.forward = patched_vision_forward


def _patch_qwen3_omni_audio_attention(modeling) -> None:
    """Patch Qwen3OmniMoeAudioAttention to support arbitrary TP and TPU SDPA."""
    target_cls = getattr(modeling, "Qwen3OmniMoeAudioAttention", None)
    if target_cls is None or getattr(
        target_cls, "_tpu_qwen3_omni_audio_attn_patch", False
    ):
        return

    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear

    def tpu_audio_attention_init(self, config, quant_config=None, prefix: str = ""):
        nn.Module.__init__(self)
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        tp_size = get_tensor_model_parallel_world_size()
        disable_tp = self.num_heads % tp_size != 0
        self.tp_size = 1 if disable_tp else tp_size
        self.num_local_heads = self.num_heads // self.tp_size
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=disable_tp,
        )
        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
            disable_tp=disable_tp,
        )

    def tpu_audio_attention_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        seq_len, _ = hidden_states.size()
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = [
            t.view(seq_len, self.num_local_heads, self.head_dim)
            .permute(1, 0, 2)
            .unsqueeze(0)
            for t in qkv.chunk(3, dim=-1)
        ]

        attn_mask = None
        if cu_seqlens is not None and cu_seqlens.numel() > 2:
            cu = cu_seqlens.to(device=hidden_states.device, dtype=torch.long)
            pos = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long)
            seg_ids = torch.searchsorted(cu[1:], pos, right=True)
            valid = pos < cu[-1]
            same_seg = (
                (seg_ids.unsqueeze(1) == seg_ids.unsqueeze(0))
                & valid.unsqueeze(1)
                & valid.unsqueeze(0)
            )
            attn_mask = (
                torch.where(same_seg, 0.0, float("-inf"))
                .to(dtype=q.dtype)
                .unsqueeze(0)
                .unsqueeze(0)
            )

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=self.scaling
        )
        out = out.squeeze(0).permute(1, 0, 2).reshape(seq_len, -1)
        output, _ = self.out_proj(out)
        return output

    target_cls.__init__ = tpu_audio_attention_init
    target_cls.forward = tpu_audio_attention_forward
    target_cls.qkv = property(lambda self: self.qkv_proj)
    target_cls._tpu_qwen3_omni_audio_attn_patch = True


def _patch_qwen3_omni_audio_encoder(modeling) -> None:
    """Patch Qwen3OmniMoeAudioEncoder to align input_features with feature_lens."""
    target_cls = getattr(modeling, "Qwen3OmniMoeAudioEncoder", None)
    if target_cls is None:
        return

    for method_name in ("_forward_encoder", "forward"):
        orig_fn = getattr(target_cls, method_name, None)
        if orig_fn is None or getattr(
            orig_fn, "_tpu_qwen3_omni_audio_encoder_patch", False
        ):
            continue

        def _make_patched(fn):
            @functools.wraps(fn)
            def patched_audio_encoder(
                self, input_features, feature_lens, *args, **kwargs
            ):
                expected = int(feature_lens.sum().item())
                actual = input_features.shape[-1]
                if actual < expected:
                    input_features = F.pad(input_features, (0, expected - actual))
                elif actual > expected:
                    input_features = input_features[..., :expected]
                with _scoped_qwen3_vl_torch_ops():
                    return fn(self, input_features, feature_lens, *args, **kwargs)

            patched_audio_encoder._tpu_qwen3_omni_audio_encoder_patch = True
            return patched_audio_encoder

        setattr(target_cls, method_name, _make_patched(orig_fn))


def _patch_qwen3_omni_mrope(modeling) -> None:
    """Patch Qwen3OmniMoeThinkerForConditionalGeneration._get_mrope_input_positions."""
    target_cls = getattr(
        modeling, "Qwen3OmniMoeThinkerForConditionalGeneration", None
    )
    orig_fn = getattr(target_cls, "_get_mrope_input_positions", None) if target_cls else None
    if orig_fn is None or getattr(orig_fn, "_tpu_qwen3_omni_mrope_patch", False):
        return

    def _move(val, to_cpu: bool, dev_box: list):
        if isinstance(val, torch.Tensor):
            if to_cpu:
                if dev_box[0] is None and val.device.type != "cpu":
                    dev_box[0] = val.device
                return val.cpu()
            return val.to(dev_box[0]) if dev_box[0] is not None else val
        if isinstance(val, (tuple, list)):
            return type(val)(_move(x, to_cpu, dev_box) for x in val)
        if isinstance(val, dict):
            return {k: _move(v, to_cpu, dev_box) for k, v in val.items()}
        return val

    sig = inspect.signature(orig_fn)

    @functools.wraps(orig_fn)
    def patched_mrope(self, *args, **kwargs):
        try:
            bound = sig.bind(self, *args, **kwargs)
        except TypeError:
            bound = sig.bind_partial(self, *args, **kwargs)
        bound.apply_defaults()
        dev_box = [None]
        for k, v in list(bound.arguments.items()):
            if k != "self":
                bound.arguments[k] = _move(v, True, dev_box)
        call_args = bound.args[1:]
        call_kwargs = {k: v for k, v in bound.kwargs.items() if k != "self"}
        with _scoped_qwen3_vl_torch_ops():
            res = orig_fn(self, *call_args, **call_kwargs)
        return _move(res, False, dev_box)

    patched_mrope._tpu_qwen3_omni_mrope_patch = True
    target_cls._get_mrope_input_positions = patched_mrope


def apply_qwen3_omni_thinker_module_patches(modeling) -> None:
    """Apply Qwen3-Omni-MoE Thinker patches to a target module."""
    _patch_qwen3_omni_vision_transformer(modeling)
    _patch_qwen3_omni_audio_attention(modeling)
    _patch_qwen3_omni_audio_encoder(modeling)
    _patch_qwen3_omni_mrope(modeling)


def maybe_patch_qwen3_omni_moe_thinker(
    model_config: Optional["ModelConfig"] = None,
) -> bool | None:
    """Apply model-specific patches for Qwen3-Omni-MoE Thinker on TPU."""
    if not _is_qwen3_omni_thinker_model(model_config):
        return False

    for mod_path in (
        "vllm.model_executor.models.qwen3_omni_moe_thinker",
        "vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker",
    ):
        try:
            apply_qwen3_omni_thinker_module_patches(importlib.import_module(mod_path))
        except ImportError:
            pass
    return None
