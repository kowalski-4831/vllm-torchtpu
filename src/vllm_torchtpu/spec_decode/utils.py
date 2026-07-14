import torch
import torch.distributed as dist
from torch.nn import Parameter
from torch_tpu._internal import sync
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def gather_sharded_weight(
        sharded_module) -> tuple[torch.Tensor, int, int, torch.device]:
    """Gathers a sharded weight tensor across TP into a full replicated tensor."""
    tp_group = get_tp_group().cpu_group
    org_vocab = sharded_module.org_vocab_size
    dim = sharded_module.embedding_dim

    si = sharded_module.shard_indices
    num_added = si.added_vocab_end_index - si.added_vocab_start_index
    if num_added > 0:
        raise NotImplementedError(
            "Draft models on TPU do not support LoRA with vocabulary expansion."
        )

    num_org = si.org_vocab_end_index - si.org_vocab_start_index
    full = torch.zeros(org_vocab, dim, dtype=torch.float32)
    my_rows = sharded_module.weight.data[:num_org].to(torch.float32).cpu()
    full[si.org_vocab_start_index:si.org_vocab_end_index] = my_rows
    dist.all_reduce(full, group=tp_group)

    draft_dtype = sharded_module.weight.dtype
    draft_device = sharded_module.weight.device
    full_dev = full.to(draft_dtype).to(draft_device)
    return full_dev, org_vocab, dim, draft_device


def populate_draft_embed_from_target(draft_model, target_embed) -> None:
    """Fill the draft's own (tp=1, full-vocab) embed_tokens with the
    target's embedding, assembled on the host.
    """
    draft_embed = draft_model.model.embed_tokens
    assert draft_embed.org_vocab_size == target_embed.org_vocab_size, (
        f"draft embed org_vocab {draft_embed.org_vocab_size} != target "
        f"{target_embed.org_vocab_size}; cannot populate")

    full_dev, org_vocab, dim, draft_device = gather_sharded_weight(
        target_embed)

    new_embed = torch.nn.Embedding(org_vocab, dim, _weight=full_dev)
    new_embed.weight.requires_grad_(False)
    del draft_model.model.embed_tokens
    draft_model.model.embed_tokens = new_embed

    if draft_device.type == "tpu":
        sync.synchronize(new_embed.weight, wait=True)
    logger.info(
        "Draft embed_tokens replaced with full replicated nn.Embedding: "
        "%d x %d per worker.", org_vocab, dim)


def maybe_share_embeddings(draft_model,
                           target_model,
                           draft_replicated: bool,
                           force_share: bool = False) -> None:
    target_lm = target_model.get_language_model() if hasattr(
        target_model, "get_language_model") else target_model
    target_lm_model = getattr(target_lm, "model", None)
    target_embed = getattr(target_lm_model, "embed_tokens",
                           None) if target_lm_model is not None else None

    if force_share:
        share_embed = True
    elif hasattr(draft_model, "has_own_embed_tokens"):
        share_embed = not draft_model.has_own_embed_tokens
    else:
        logger.info("Draft model does not declare `has_own_embed_tokens`; "
                    "defaulting to share embed_tokens with the target.")
        share_embed = True

    if share_embed:
        if target_embed is None:
            raise RuntimeError(
                "Draft embedding sharing requires target_lm.model.embed_tokens"
            )
        if draft_replicated:
            logger.info(
                "Populating draft's own embed_tokens with a host-gathered copy of target embedding."
            )
            populate_draft_embed_from_target(draft_model, target_embed)
        else:
            logger.info(
                "Sharing the target's sharded embed_tokens with the sharded draft."
            )
            draft_model.model.embed_tokens = target_embed


def populate_draft_lm_head_from_target(draft_model, target_lm_head) -> None:
    draft_lm_head = draft_model.model.lm_head if hasattr(
        draft_model, "model") and hasattr(draft_model.model,
                                          "lm_head") else getattr(
                                              draft_model, "lm_head", None)
    if draft_lm_head is None:
        draft_lm_head = ParallelLMHead(target_lm_head.org_vocab_size,
                                       target_lm_head.embedding_dim)
        draft_model.lm_head = draft_lm_head

    full_dev, org_vocab, dim, draft_device = gather_sharded_weight(
        target_lm_head)

    draft_lm_head.weight = Parameter(full_dev, requires_grad=False)

    if draft_device.type == "tpu":
        sync.synchronize(draft_lm_head.weight, wait=True)
    logger.info(
        "Draft lm_head replaced with full replicated nn.Linear: "
        "%d x %d per worker.", org_vocab, dim)


def maybe_share_lm_head(draft_model,
                        target_model,
                        draft_replicated: bool,
                        force_share: bool = False) -> None:
    target_lm = target_model.get_language_model() if hasattr(
        target_model, "get_language_model") else target_model
    target_lm_head = getattr(target_lm, "lm_head", None)

    if target_lm_head is None:
        raise RuntimeError(
            "Draft expects the target model to have an lm_head.")

    if force_share:
        share_lm_head = True
    elif hasattr(draft_model, "has_own_lm_head"):
        share_lm_head = not draft_model.has_own_lm_head
    else:
        logger.info("Draft model does not declare `has_own_lm_head`; "
                    "defaulting to share lm_head with the target.")
        share_lm_head = True

    if share_lm_head:
        if draft_replicated:
            logger.info("Populating draft's own lm_head with a host-gathered, "
                        "per-worker replicated copy of the target lm_head.")
            populate_draft_lm_head_from_target(draft_model, target_lm_head)
        else:
            logger.info(
                "Sharing the target's sharded lm_head with the sharded draft.")
            if hasattr(draft_model, "lm_head"):
                draft_model.lm_head = target_lm_head
            elif hasattr(draft_model, "model") and hasattr(
                    draft_model.model, "lm_head"):
                draft_model.model.lm_head = target_lm_head
            else:
                draft_model.lm_head = target_lm_head

    # We must override _gather_logits for replicated draft models even if we don't share the lm_head,
    # because the draft model's logits_processor will still try to gather logits across TP=1.
    if draft_replicated:
        lp = getattr(draft_model, "logits_processor", None)
        if lp is not None:
            assert hasattr(lp, "_gather_logits"), (
                "draft logits_processor has no _gather_logits to override; "
                "vLLM may have renamed it.")
            lp._gather_logits = lambda logits: logits
