# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Donating slot-to-slot copy for Mamba recurrent state arrays."""

import torch

_mamba_state_copy_op = None


def _build_op():
    import jax
    from torch_tpu._internal import pallas

    def _state_copy_tpu(state: jax.Array, src: jax.Array,
                        dst: jax.Array) -> jax.Array:
        # Keep the scatter under cond so torch_tpu's aliasing validation accepts
        # the donated buffer. Block ids are non-negative in real calls.
        return jax.lax.cond(
            dst[0] >= 0,
            lambda s: s.at[dst].set(s[src]),
            lambda s: s,
            state,
        )

    op = pallas.jax_op("pallas::mamba_apc_state_copy",
                       _state_copy_tpu,
                       donate_argnums=(0, ))
    op.register_fake(lambda state, src, dst: torch.empty_like(state))
    return op


def ensure_op_built() -> None:
    """Force pallas op construction outside the hot path.

    `_build_op()` traces StableHLO on first call (multi-second stall on cold
    JAX). Call this during runner init so the first live request that crosses
    a Mamba block boundary doesn't pay that cost.
    """
    global _mamba_state_copy_op
    if _mamba_state_copy_op is None:
        _mamba_state_copy_op = _build_op()


def mamba_state_copy(state: torch.Tensor, src: torch.Tensor,
                     dst: torch.Tensor) -> None:
    """In-place `state[dst] = state[src]` along dim 0 with buffer donation."""
    ensure_op_built()
    state.copy_(_mamba_state_copy_op(state, src, dst))
