"""Behavioral tests for full-Surah orchestration and the real consumer schema."""

from __future__ import annotations

import ast
import gc
import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import CancelledError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from munajjam.core.arabic import normalize_arabic
from munajjam.core.hybrid_aligner import HybridQuranAligner
from munajjam.exceptions import (
    InvalidProviderOutputError,
    ModelUnavailableError,
    NoBreathGroupsError,
    QuranDataError,
)
from munajjam.hybrid_aligner import (
    AudioBuffer,
    PhonemeEmission,
    QuranRecitationSegmenter,
    Wav2Vec2ForcedAligner,
)
from munajjam.hybrid_aligner.reference import CanonicalReference, CanonicalReferencePart
from munajjam.hybrid_aligner.surah import (
    ISTIADHAH_TEXT,
    aggregate_words,
    full_surah_reference,
    map_surah,
    reference_words,
)
from munajjam.hybrid_aligner.types import AlignmentSpan


def reference(*texts):
    parts = tuple(
        CanonicalReferencePart("ayah", text, f"test#{i}", "a" * 64, 1, i)
        for i, text in enumerate(texts, 1)
    )
    text = " ".join(texts)
    return CanonicalReference(
        "hafs", parts, "test", "a" * 64, text, hashlib.sha256(text.encode()).hexdigest()
    )


def observations(texts):
    table = tuple(sorted(set(normalize_arabic(" ".join(texts)).replace(" ", ""))))
    groups = [
        tuple(
            PhonemeEmission(table.index(c), i, i + 1, -0.1)
            for i, c in enumerate(normalize_arabic(text).replace(" ", ""))
        )
        for text in texts
    ]
    return table, groups


def mapped(ref, *texts, **kwargs):
    table, groups = observations(texts)
    return map_surah(ref, groups, table, **kwargs)


class SegmentBackend:
    def __init__(self, count):
        self.count = count
        self.controls = None
        self.closed = 0

    def configure_cleanup(self, *controls):
        self.controls = controls

    def __call__(self, audio):
        return [(float(i * 5), float(i * 5 + 4)) for i in range(self.count)]

    def close(self):
        self.closed += 1


class ReferenceAligner:
    def __init__(self, texts):
        self.table, self.groups = observations(texts)
        self.closed = 0

    def align(self, audio, group):
        return self.groups[int(group.start / 5)]

    def token_table(self):
        return self.table

    def close(self):
        self.closed += 1


class ForcedAligner:
    def __init__(self):
        self.texts = []
        self.closed = 0

    def encode_text(self, text):
        self.texts.append(text)
        tokens = list(text.replace(" ", "|"))
        return list(range(1, len(tokens) + 1)), tokens

    def align_group(self, audio, group, token_ids, token_texts, **kwargs):
        step = (group.end - group.start) / (len(token_texts) + 1)
        return tuple(
            AlignmentSpan(t, group.start + i * step, group.start + (i + 1) * step, 0.81)
            for i, t in enumerate(token_texts)
        )

    def close(self):
        self.closed += 1


def engine(texts):
    backend = SegmentBackend(len(texts))
    pipeline = HybridQuranAligner(
        segmenter=QuranRecitationSegmenter(backend_factory=lambda: backend),
        reference_aligner=ReferenceAligner(texts),
        forced_aligner=ForcedAligner(),
    )
    audio = AudioBuffer(np.zeros(max(len(texts), 1) * 5 * 16000, dtype=np.float32), 16000)
    return pipeline, audio, backend


@pytest.mark.parametrize("riwaya,count", [("hafs", 6), ("warsh", 6)])
def test_complete_reference_and_provenance(riwaya, count):
    ref = full_surah_reference(114, riwaya)
    assert len([p for p in ref.parts if p.kind == "ayah"]) == count
    assert ref.parts[0].kind == "istiadhah" and ref.parts[1].kind == "basmalah"
    assert len(ref.source_file_sha256) == 64
    assert ref.text_sha256 == hashlib.sha256(ref.text.encode()).hexdigest()
    assert all(p.source and p.source_sha256 for p in ref.parts)
    assert full_surah_reference(9, riwaya).parts[1].kind == "ayah"


@pytest.mark.parametrize(
    "surah,riwaya", [(0, "hafs"), (115, "hafs"), (True, "hafs"), (1, "unknown")]
)
def test_invalid_reference(surah, riwaya):
    with pytest.raises(ValueError):
        full_surah_reference(surah, riwaya)


def test_no_invented_empty_reference(monkeypatch):
    class Missing:
        def joinpath(self, _):
            return self

        def read_bytes(self):
            return b'{"1": ["valid", ""]}'

    monkeypatch.setattr("munajjam.hybrid_aligner.surah.resources.files", lambda _: Missing())
    with pytest.raises(QuranDataError, match="complete canonical"):
        full_surah_reference(1, "warsh")


def test_global_unequal_breath_ayah_boundaries():
    ref = reference("قل اعوذ", "برب الناس", "ملك الناس")
    mapping = mapped(ref, "قل اعوذ برب", "الناس ملك الناس")
    assert mapping.group_word_indices == ((0, 1, 2), (3, 4, 5))
    indices = [c["canonical_character_index"] for c in mapping.diagnostics["cells"]]
    assert indices == sorted(set(indices))


def test_order_cannot_restart():
    with pytest.raises(InvalidProviderOutputError):
        mapped(reference("قل اعوذ", "برب الناس"), "برب الناس", "قل اعوذ")


@pytest.mark.parametrize("text", ["سلام سلام", "zzzzzzzzzz"])
def test_repetition_or_low_quality_rejected(text):
    with pytest.raises(InvalidProviderOutputError):
        mapped(reference("سلام"), text)


def test_ambiguous_repeated_passage_rejected():
    with pytest.raises(InvalidProviderOutputError, match="ambiguous"):
        mapped(reference("سلام نور سلام"), "سلام")


def test_partial_word_and_internal_missing_word_rejected():
    for ref, text in [
        (reference("سلام نور"), "لام نور"),
        (reference("سلام عالمين نور"), "سلام نور"),
    ]:
        with pytest.raises(InvalidProviderOutputError):
            mapped(ref, text)


def test_partial_surah_word_boundary_has_explicit_omissions():
    pipeline, audio, _ = engine(["ملك الناس"])
    result = pipeline.align_surah(audio, 114)
    assert [a.ayah_number for a in result] == [2]
    assert result.metadata["partial"] and result.metadata["omitted_canonical_word_indices"]


def test_one_breath_multiple_ayahs_and_controls_and_progress():
    ref = full_surah_reference(114, "hafs")
    text = " ".join(p.text for p in ref.parts if p.kind == "ayah")
    pipeline, audio, backend = engine([text])
    progress = []
    result = pipeline.align_surah(
        audio,
        114,
        min_silence_ms=400,
        pad_ms=50,
        progress_callback=lambda c, t: progress.append((c, t)),
    )
    assert backend.controls == (400, 100, 50)
    assert progress == [(0, 1), (1, 1)]
    assert [a.ayah_number for a in result] == list(range(1, 7))
    assert not result.metadata["partial"]
    assert all(not a.is_breath_boundary for a in result.data[:-1])
    assert result.data[-1].is_breath_boundary
    assert result.breath_groups[0].duration == 4
    json.dumps(result.to_dict(), allow_nan=False)


def test_ayah_spans_breaths_and_pause_semantics():
    pipeline, audio, _ = engine(["قل اعوذ", "برب الناس", "ملك الناس"])
    result = pipeline.align_surah(audio, 114)
    assert [a.ayah_number for a in result] == [1, 2]
    first = result.data[0]
    assert len(first.words) == 4 and first.start_time == 0 and first.end_time > 5
    assert first.is_breath_boundary
    assert first.pause_duration == pytest.approx(result.data[1].start_time - first.end_time)
    assert first.words[2].start >= 5
    assert all(w.probability == pytest.approx(0.81) for w in first.words)


@pytest.mark.parametrize("riwaya", ["hafs", "warsh"])
@pytest.mark.parametrize("intro", ["", "istiadhah", "basmalah", "both"])
def test_optional_introductions_and_riwaya(riwaya, intro):
    ref = full_surah_reference(114, riwaya)
    ayahs = " ".join(p.text for p in ref.parts if p.kind == "ayah")
    phrases = []
    if intro in ("istiadhah", "both"):
        phrases.append(ISTIADHAH_TEXT)
    if intro in ("basmalah", "both"):
        phrases.append(ref.parts[1].text)
    pipeline, audio, _ = engine([" ".join(phrases + [ayahs])])
    result = pipeline.align_surah(audio, 114, riwaya)
    assert len(result.data) == 6
    assert [i.kind for i in result.introductions] == (
        ["istiadhah"]
        if intro == "istiadhah"
        else ["basmalah"]
        if intro == "basmalah"
        else ["istiadhah", "basmalah"]
        if intro == "both"
        else []
    )
    assert all(a.ayah_number > 0 for a in result)


@pytest.mark.parametrize("riwaya", ["hafs", "warsh"])
def test_fatiha_basmalah_is_canonical_ayah_text(riwaya):
    ref = full_surah_reference(1, riwaya)
    text = " ".join(p.text for p in ref.parts if p.kind == "ayah")
    pipeline, audio, _ = engine([text])
    result = pipeline.align_surah(audio, 1, riwaya)
    assert not result.introductions
    assert result.data[0].ayah_number == 1
    assert "بِسْمِ" in result.data[0].text
    assert len(result.data[0].words) == (4 if riwaya == "hafs" else 8)


def test_zero_breaths_fail_closed():
    pipeline, audio, _ = engine([])
    with pytest.raises(NoBreathGroupsError):
        pipeline.align_surah(audio, 1)


def test_word_crossing_breaths_rejected():
    with pytest.raises(InvalidProviderOutputError, match="across breaths"):
        mapped(reference("سلام نور"), "سل", "ام نور")


def test_geometric_character_confidence_and_canonical_text():
    words = reference_words(reference("قُلْ"))
    result = aggregate_words(
        words, [AlignmentSpan("ق", 0, 0.1, 0.25), AlignmentSpan("ل", 0.1, 0.2, 1)]
    )
    assert result[0].word == "قُلْ"
    assert result[0].probability == pytest.approx(0.5)
    assert (result[0].start, result[0].end) == (0, 0.2)


@pytest.mark.parametrize(
    "spans",
    [
        [],
        [AlignmentSpan("ن", 0, 0.1, 0.8)],
        [AlignmentSpan("ق", 0, 0.2, 0.8), AlignmentSpan("ل", 0.1, 0.3, 0.9)],
        [AlignmentSpan("ق", 0, 0.1, None), AlignmentSpan("ل", 0.1, 0.2, 0.9)],
    ],
)
def test_missing_reordered_or_unscored_characters_rejected(spans):
    with pytest.raises(InvalidProviderOutputError):
        aggregate_words(reference_words(reference("قل")), spans)


def test_ctc_confidence_uses_frame_posterior_not_path_score():
    forced = Wav2Vec2ForcedAligner()
    probs = np.array([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1]])
    result = forced._ctc(np.log(probs), [1])
    assert result == [(1, 0, 2, pytest.approx(0.9))]


def test_ctc_repeated_token_states_stay_distinct():
    probs = np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    result = Wav2Vec2ForcedAligner()._ctc(np.log(probs), [1, 1])
    assert len(result) == 2 and result[0][2] <= result[1][1]


def test_cancel_and_callback_errors():
    pipeline, audio, _ = engine(["ملك الناس"])
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(CancelledError):
        pipeline.align_surah(audio, 114, cancel_event=cancelled)

    def callback(c, t):
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        pipeline.align_surah(audio, 114, progress_callback=callback)

    def cancel_callback(c, t):
        cancelled.set()

    cancelled.clear()
    with pytest.raises(CancelledError):
        pipeline.align_surah(audio, 114, cancel_event=cancelled, progress_callback=cancel_callback)


def test_dp_budget_and_cancellation():
    with pytest.raises(InvalidProviderOutputError, match="budget"):
        mapped(reference("قل اعوذ"), "قل اعوذ", max_cells=5)

    def cancel():
        raise CancelledError()

    with pytest.raises(CancelledError):
        mapped(reference("قل اعوذ"), "قل اعوذ", check_cancelled=cancel)


@pytest.mark.parametrize(
    "kwargs",
    [{"repetition_attach": True}, {"min_silence_ms": 99}, {"min_speech_ms": True}, {"pad_ms": -1}],
)
def test_undefined_repetition_and_invalid_controls(kwargs):
    pipeline, audio, _ = engine(["ملك الناس"])
    with pytest.raises(ValueError):
        pipeline.align_surah(audio, 114, **kwargs)


def test_close_idempotent_cuda_cleanup_and_terminal_state(monkeypatch):
    pipeline, audio, backend = engine(["ملك الناس"])
    pipeline.align_surah(audio, 114)
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True, empty_cache=lambda: calls.append("cuda")
            )
        ),
    )
    monkeypatch.setattr(gc, "collect", lambda: calls.append("gc"))
    reference_aligner, forced_aligner = pipeline.reference_aligner, pipeline.forced_aligner
    pipeline.close()
    pipeline.close()
    assert calls == ["gc", "cuda"]
    assert backend.closed == reference_aligner.closed == forced_aligner.closed == 1
    with pytest.raises(ModelUnavailableError, match="closed"):
        pipeline.align_surah(audio, 114)


def test_cpu_only_import_and_close_isolation():
    code = """
import sys
class BlockML:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'torch','transformers','sherpa_onnx','onnxruntime','recitations_segmenter'}:
   raise ImportError('optional dependency blocked: ' + fullname)
sys.meta_path.insert(0, BlockML())
from munajjam.core.hybrid_aligner import HybridQuranAligner
engine = HybridQuranAligner.from_pretrained(allow_download=False)
engine.close()
engine.close()
assert 'torch' not in sys.modules
"""
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=os.environ.copy()
    )
    assert run.returncode == 0, run.stderr


def test_actual_pr125_formatter_contract():
    # Supplied from the isolated PR worktree; do not import FastAPI or run server jobs.
    path = os.environ.get("PR125_SERVER_PATH")
    if path is None:
        pytest.skip("set PR125_SERVER_PATH to exact PR #125 server.py for consumer verification")
    tree = ast.parse(Path(path).read_text())
    names = {
        "_safe_get",
        "_format_ayah_segment",
        "_format_breath_groups",
        "_extract_breath_boundaries",
        "_build_success_response",
    }
    namespace = {"Any": object}
    module = ast.Module(
        body=[
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
        ],
        type_ignores=[],
    )
    exec(compile(module, path, "exec"), namespace)
    pipeline, audio, _ = engine(["قل اعوذ برب الناس", "ملك الناس"])
    result = pipeline.align_surah(audio, 114)
    response = namespace["_build_success_response"](result)
    assert response["data"] == json.loads(json.dumps(result.to_dict()))["data"]
    assert response["breath_groups"] == []  # verified upstream stub, never claim it works
    formatted = namespace["_format_breath_groups"](result.breath_groups)
    assert formatted == result.to_dict()["breath_groups"]
    # Proposed adapter replaces only the explicitly unfinished extraction boundary.
    namespace["_extract_breath_boundaries"] = lambda result: result.breath_groups
    assert json.loads(json.dumps(namespace["_build_success_response"](result))) == json.loads(
        json.dumps(result.to_dict())
    )


def test_incomplete_bundled_warsh_surah_fails_closed():
    # Actual upstream slot 2:254 is empty: never borrow Hafs or renumber it.
    with pytest.raises(QuranDataError, match="complete canonical warsh Surah 2"):
        full_surah_reference(2, "warsh")


def test_small_repeated_word_cannot_hide_inside_good_long_passage():
    with pytest.raises(InvalidProviderOutputError):
        mapped(
            reference("قل اعوذ برب الناس ملك الناس اله الناس"),
            "قل اعوذ اعوذ برب الناس ملك الناس اله الناس",
        )


def test_multiple_optimal_breath_assignments_rejected():
    with pytest.raises(InvalidProviderOutputError, match="across breaths"):
        mapped(reference("سلام سلام نور"), "سلام سلا", "م نور")


def test_no_reference_aligner_fails_closed():
    pipeline, audio, _ = engine(["ملك الناس"])
    pipeline.reference_aligner = None
    with pytest.raises(ModelUnavailableError, match="Zipformer"):
        pipeline.align_surah(audio, 114)


def test_adapter_actual_job_parameters_and_cleanup(monkeypatch):
    import asyncio

    from munajjam.hybrid_aligner.server_adapter import run_hybrid_alignment

    pipeline, audio, backend = engine(["ملك الناس"])
    monkeypatch.setattr(HybridQuranAligner, "from_pretrained", lambda **_: pipeline)
    progress = []
    result = asyncio.run(
        run_hybrid_alignment(
            "job-test",
            audio,
            {
                "surah_id": 114,
                "riwaya": "hafs",
                "method": "hybrid",
                "chunk_duration": 30.0,
                "min_silence_ms": 500,
                "min_speech_ms": 200,
                "pad_ms": 50,
                "repetition_attach": False,
            },
            lambda c, t: progress.append((c, t)),
        )
    )
    assert [a.ayah_number for a in result] == [2]
    assert backend.controls == (500, 200, 50)
    assert pipeline._closed and progress == [(0, 1), (1, 1)]


def test_adapter_cleanup_on_failure(monkeypatch):
    import asyncio

    from munajjam.hybrid_aligner.server_adapter import run_hybrid_alignment

    pipeline, audio, _ = engine(["ملك الناس"])
    monkeypatch.setattr(HybridQuranAligner, "from_pretrained", lambda **_: pipeline)
    with pytest.raises(ValueError):
        asyncio.run(run_hybrid_alignment("job-test", audio, {"surah_id": 0}, lambda *_: None))
    assert pipeline._closed


def test_adapter_async_cancellation_waits_for_cleanup(monkeypatch):
    import asyncio

    from munajjam.hybrid_aligner.server_adapter import run_hybrid_alignment

    started = threading.Event()
    finished = threading.Event()

    class CancellableEngine:
        def align_surah(self, *args, cancel_event, **kwargs):
            started.set()
            assert cancel_event.wait(timeout=5), "adapter did not request cancellation"
            raise CancelledError()

        def close(self):
            finished.set()

    monkeypatch.setattr(HybridQuranAligner, "from_pretrained", lambda **_: CancellableEngine())

    async def scenario():
        task = asyncio.create_task(
            run_hybrid_alignment("job-test", "unused", {"surah_id": 114}, lambda *_: None)
        )
        await asyncio.to_thread(started.wait, 5)
        assert started.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize("repetition_attach,expected_status", [(False, "success"), (True, "error")])
def test_actual_pr125_job_lifecycle(monkeypatch, tmp_path, repetition_attach, expected_status):
    import asyncio
    import importlib.util
    import wave

    from munajjam.hybrid_aligner.server_adapter import (
        extract_breath_boundaries,
        run_hybrid_alignment,
    )

    path = os.environ.get("PR125_SERVER_PATH")
    if path is None:
        pytest.skip("set PR125_SERVER_PATH to exact PR #125 server.py")
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart")
    spec = importlib.util.spec_from_file_location("pr125_contract", path)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    pipeline, _, _ = engine(["ملك الناس"])
    monkeypatch.setattr(HybridQuranAligner, "from_pretrained", lambda **_: pipeline)
    monkeypatch.setattr(server, "_run_hybrid_alignment", run_hybrid_alignment)
    monkeypatch.setattr(server, "_extract_breath_boundaries", extract_breath_boundaries)
    audio_path = tmp_path / "job.wav"
    with wave.open(str(audio_path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0" * (5 * 16000 * 2))
    server.jobs["job-test"] = {
        "status": "queued",
        "progress": 0,
        "message": "queued",
        "data": None,
        "error": None,
    }
    try:
        server._process_alignment_job(
            "job-test",
            str(audio_path),
            114,
            "hybrid",
            "hafs",
            30.0,
            300,
            100,
            100,
            repetition_attach,
        )
        response = asyncio.run(server.get_job_status("job-test"))
        data = json.loads(response.body)
        assert data["status"] == expected_status
        if expected_status == "success":
            assert response.status_code == 200 and data["progress"] == 100
            assert data["data"][0]["ayah_number"] == 2
            assert data["data"][0]["words"][0]["word"] == "مَلِكِ"
            assert data["breath_groups"] == [
                {"start_sec": 0.0, "end_sec": 4.0, "duration_sec": 4.0}
            ]
        else:
            assert response.status_code == 500 and "repetition_attach" in data["error"]
        assert pipeline._closed and not audio_path.exists()
    finally:
        server._executor.shutdown(wait=True)
