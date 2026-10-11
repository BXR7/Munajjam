"""Engine-owned replacements for PR #125's two explicitly unfinished stubs.

The server continues to own job state, upload lifetime and HTTP formatting.
This module owns model lifetime and cooperative cancellation. It never imports
FastAPI, modifies server jobs, or silently interprets repetition_attach.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

from munajjam.core.hybrid_aligner import HybridQuranAligner
from munajjam.hybrid_aligner.surah import SurahAlignmentResult


async def run_hybrid_alignment(
    job_id: str,
    audio_path: str,
    params: dict,
    progress_callback: Callable[[int, int], None],
) -> SurahAlignmentResult:
    """PR #125 runner signature, suitable for its existing asyncio.run caller.

    Cancellation requests stop between model calls/DP rows; teardown completes
    before the coroutine exits. Native inference itself cannot be interrupted.
    The server's chunk_duration is not a breath-segmentation control: physical
    breath segmentation uses the three millisecond parameters instead.
    """
    if params.get("method", "hybrid") != "hybrid":
        raise ValueError("method must be hybrid")
    cancel_event = threading.Event()

    def run() -> SurahAlignmentResult:
        engine = HybridQuranAligner.from_pretrained()
        try:
            result = engine.align_surah(
                audio_path,
                params["surah_id"],
                params.get("riwaya", "hafs"),
                params.get("min_silence_ms", 300),
                params.get("min_speech_ms", 100),
                params.get("pad_ms", 100),
                params.get("repetition_attach", False),
                progress_callback,
                cancel_event=cancel_event,
            )
        except BaseException as error:
            try:
                engine.close()
            except Exception as cleanup_error:
                # Preserve the inference/callback/cancellation failure as the
                # public error while retaining teardown evidence in its cause.
                raise error from cleanup_error
            raise
        else:
            engine.close()
            return result

    worker = asyncio.create_task(asyncio.to_thread(run))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancel_event.set()
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                # A second cancellation must not let cleanup escape to an
                # unobserved background thread or overlap the next GPU job.
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()  # observe any teardown/inference error after cancellation
        raise


def extract_breath_boundaries(alignment_result: SurahAlignmentResult) -> list[Any]:
    """PR #125 extraction boundary; its formatter already accepts BreathGroup."""
    return list(alignment_result.breath_groups)
