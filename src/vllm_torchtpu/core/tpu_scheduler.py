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
import time

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_torchtpu import envs

logger = init_logger(__name__)


class TpuDpScheduler(AsyncScheduler):
    """TpuDpScheduler that buffers prefill and flushes them in batches."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._buffer_prefill = envs.DP_SCHED_BUFFER_PREFILL
        self._buffer_timeout_ms = max(0,
                                      envs.DP_SCHED_BUFFER_PREFILL_TIMEOUT_MS)
        self._max_buffer_count = max(1,
                                     self.parallel_config.data_parallel_size)
        self._flushing = False
        logger.info(
            "TpuDpScheduler active: buffer_prefill=%s max_buffer_count=%d "
            "buffer_timeout_ms=%d", self._buffer_prefill,
            self._max_buffer_count, self._buffer_timeout_ms)

    def _oldest_waiting_age_ms(self) -> float:
        oldest = None
        for queue in (self.waiting, self.skipped_waiting):
            if queue:
                arrival = queue.peek_request().arrival_time
                if oldest is None or arrival < oldest:
                    oldest = arrival
        if oldest is None:
            return 0.0
        return (time.time() - oldest) * 1000.0

    def _should_flush_prefill(self) -> bool:
        num_waiting = len(self.waiting) + len(self.skipped_waiting)
        if num_waiting == 0:
            # No more pending. Flushing if any is completed.
            self._flushing = False
            return True
        if not self._flushing:
            self._flushing = (not self.running
                              or num_waiting >= self._max_buffer_count
                              or self._oldest_waiting_age_ms()
                              >= self._buffer_timeout_ms)
        return self._flushing

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        if not self._buffer_prefill or self._should_flush_prefill():
            return super().schedule(throttle_prefills)

        # Pause scheduling any new prefill req, which effectively holds and
        # buffers new reqs.
        prev_pause_state = self._pause_state
        if prev_pause_state == PauseState.UNPAUSED:
            self._pause_state = PauseState.PAUSED_NEW
        try:
            return super().schedule(throttle_prefills)
        finally:
            self._pause_state = prev_pause_state
