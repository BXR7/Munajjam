"""Semantic regressions found while comparing Manus 3d08e9f and Codex fa194ad."""

from __future__ import annotations

import asyncio
import gc
import sys
import threading
from concurrent.futures import CancelledError
from types import SimpleNamespace

import pytest

from munajjam.core.arabic import normalize_arabic
from munajjam.core.hybrid_aligner import HybridQuranAligner
from munajjam.exceptions import InvalidProviderOutputError
from munajjam.hybrid_aligner import PhonemeEmission, Wav2Vec2ForcedAligner
from munajjam.hybrid_aligner.canonical_zipformer import align_zipformer_to_reference
from munajjam.hybrid_aligner.server_adapter import run_hybrid_alignment
from munajjam.hybrid_aligner.surah import full_surah_reference
from tests.unit.test_surah_alignment import engine, mapped, observations, reference


def test_manus_sparse_anchors_cannot_restore_unrecited_word():
    ref = full_surah_reference(114, "hafs")
    full_text = " ".join(p.text for p in ref.parts if p.kind == "ayah")
    # Manus returned بِرَبِّ despite no character evidence for that word.
    observation = normalize_arabic(full_text).replace("برب", "", 1)
    pipeline, audio, _ = engine([observation])
    with pytest.raises(InvalidProviderOutputError):
        pipeline.align_surah(audio, 114)


def test_recorded_basmalah_observations_do_not_spread_across_surah():
    # Deterministic replay of real public Zipformer output; not a real-model test.
    with pytest.raises(InvalidProviderOutputError):
        mapped(full_surah_reference(1, "hafs"), "بسمللااهررحماانررح")


def test_clean_basmalah_is_accepted_without_changing_thresholds():
    ref = full_surah_reference(1, "hafs")
    text = next(p.text for p in ref.parts if p.kind == "ayah")
    mapping = mapped(ref, text)
    selected = [mapping.words[i] for i in mapping.group_word_indices[0]]
    assert len(selected) == 4
    assert {ref.parts[w.part_index].ayah_number for w in selected} == {1}
    assert mapping.diagnostics["normalized_cost"] == 0


def test_manus_low_level_memory_bound_retains_codex_quality_semantics():
    ref = reference("ابتج")
    table, groups = observations(["ابت"])
    with pytest.raises(InvalidProviderOutputError, match="cell budget"):
        align_zipformer_to_reference(groups[0], table, ref, max_dp_cells=1)
    result = align_zipformer_to_reference(groups[0], table, ref)
    assert result.total_cost == 1
    assert result.normalized_cost == 0.25  # deletions still count for explicit references
    assert result.match_coverage == 0.75
    assert [c.operation for c in result.cells].count("DELETE") == 1


def test_low_level_preserves_unknown_acoustic_scores():
    emissions = [PhonemeEmission(0, 0, 1, None)]
    result = align_zipformer_to_reference(emissions, ["ا"], reference("ا"))
    assert result.cells[0].observed.score is None


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_global_cell_budget(limit):
    with pytest.raises(ValueError, match="max_cells"):
        mapped(reference("سلام"), "سلام", max_cells=limit)


def test_packed_global_traceback_enforces_byte_budget():
    ref = reference("قل اعوذ برب الناس")
    with pytest.raises(InvalidProviderOutputError, match="memory budget"):
        mapped(ref, "قل اعوذ برب الناس", max_traceback_bytes=1)
    mapping = mapped(ref, "قل اعوذ برب الناس")
    chars = len("قلاعوذبربالناس")
    assert mapping.diagnostics["traceback_bytes"] < (chars + 1) ** 2
    assert mapping.group_word_indices == ((0, 1, 2, 3),)


@pytest.mark.parametrize("cuda_available", [True, False])
def test_shutdown_attempts_all_components_when_one_close_fails(monkeypatch, cuda_available):
    calls = []

    class Component:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def close(self):
            calls.append(self.name)
            if self.fail:
                raise RuntimeError("teardown failed")

    pipeline = HybridQuranAligner(
        segmenter=Component("segmenter", True),
        reference_aligner=Component("zipformer"),
        forced_aligner=Component("wav2vec2"),
    )
    monkeypatch.setattr(gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: cuda_available, empty_cache=lambda: calls.append("cuda")
            )
        ),
    )
    with pytest.raises(RuntimeError, match="teardown failed"):
        pipeline.close()
    assert calls == ["segmenter", "zipformer", "wav2vec2", "gc"] + (
        ["cuda"] if cuda_available else []
    )
    assert pipeline.segmenter is pipeline.reference_aligner is pipeline.forced_aligner is None
    pipeline.close()


def test_forced_provider_is_detached_even_if_its_shutdown_raises():
    class Provider:
        def close(self):
            raise RuntimeError("provider close failed")

    forced = Wav2Vec2ForcedAligner(logits_provider=Provider())
    with pytest.raises(RuntimeError):
        forced.close()
    assert forced._provider is None and forced._provider_factory is None
    forced.close()


def test_repeated_async_cancellation_waits_for_worker_cleanup(monkeypatch):
    started = threading.Event()
    cancellation_seen = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class CancellableEngine:
        def align_surah(self, *args, cancel_event, **kwargs):
            started.set()
            assert cancel_event.wait(5)
            cancellation_seen.set()
            assert release.wait(5)
            raise CancelledError()

        def close(self):
            finished.set()

    monkeypatch.setattr(HybridQuranAligner, "from_pretrained", lambda: CancellableEngine())

    async def scenario():
        task = asyncio.create_task(
            run_hybrid_alignment("id", "unused", {"surah_id": 114}, lambda *_: None)
        )
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            assert await asyncio.to_thread(cancellation_seen.wait, 5)
            task.cancel()
            await asyncio.sleep(0)  # deliver the second cancellation at the shield await
            assert not task.done() and not finished.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            assert finished.is_set()
        finally:
            release.set()

    asyncio.run(scenario())


def test_cuda_cleanup_error_does_not_restore_model_references(monkeypatch):
    pipeline, _, _ = engine(["ملك الناس"])

    def fail():
        raise RuntimeError("CUDA cleanup failed")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, empty_cache=fail)),
    )
    with pytest.raises(RuntimeError, match="CUDA cleanup failed"):
        pipeline.close()
    assert pipeline.segmenter is pipeline.reference_aligner is pipeline.forced_aligner is None
    pipeline.close()


def test_adapter_preserves_inference_error_when_cleanup_also_fails(monkeypatch):
    class FailingEngine:
        def align_surah(self, *args, **kwargs):
            raise ValueError("inference failed")

        def close(self):
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(HybridQuranAligner, "from_pretrained", lambda: FailingEngine())
    with pytest.raises(ValueError, match="inference failed") as failure:
        asyncio.run(run_hybrid_alignment("id", "unused", {"surah_id": 114}, lambda *_: None))
    assert isinstance(failure.value.__cause__, RuntimeError)
