from unittest.mock import patch

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from vllm_torchtpu.core.tpu_scheduler import TpuDpScheduler


def test_schedule_forwards_throttle_prefills():
    scheduler = TpuDpScheduler.__new__(TpuDpScheduler)
    scheduler._buffer_prefill = False
    expected = object()

    with patch.object(AsyncScheduler, "schedule",
                      return_value=expected) as schedule:
        result = scheduler.schedule(throttle_prefills=True)

    assert result is expected
    schedule.assert_called_once_with(True)
