from __future__ import annotations

import json

import numpy as np
import pytest
from munajjam.core.arabic import normalize_arabic
from munajjam.core.hybrid_aligner import (
    AudioBuffer,
    CanonicalQuranReferenceProvider,
    HybridQuranAligner,
    PhonemeEmission,
    QuranRecitationSegmenter,
    VerifiedSpecialPhrase,
    Wav2Vec2ForcedAligner,
    ZipformerEvidence,
    ZipformerNeuralAligner,
)
from munajjam.core.hybrid_aligner.reference import CanonicalReferenceRequest
from munajjam.data.quran import get_ayah, get_ayah_count
from munajjam.exceptions import InvalidProviderOutputError


class FakeSegmentationBackend:
    def __init__(self, intervals):
        self.intervals = intervals
        self.min_silence_duration_ms = 30
        self.min_speech_duration_ms = 30
        self.pad_duration_ms = 30

    def __call__(self, audio):
        return self.intervals

    def close(self):
        pass


class FakeZipformerBackend:
    def __init__(self, token_table, char_ids, cuts):
        self.token_table = token_table
        self.char_ids = char_ids
        self.cuts = cuts
        self.calls = 0

    def __call__(self, audio, group):
        group_index = 0 if group.start < 1 else 1
        text = self.cuts[group_index]
        emissions = [
            PhonemeEmission(self.char_ids[char], i, i + 1, 0.99)
            for i, char in enumerate(text)
        ]
        self.calls += 1
        return emissions

    def close(self):
        pass


class FakeWav2Vec2Provider:
    def __init__(self, char_ids):
        self.char_ids = char_ids

    def encode_text(self, text):
        tokens = list(normalize_arabic(text))
        return [self.char_ids[token] for token in tokens], tokens

    def __call__(self, samples, sample_rate):
        return np.zeros((800, 51), dtype=np.float32)

    def close(self):
        pass


def build_engine(groups):
    audio = AudioBuffer(np.ones(32_000, dtype=np.float32) * 0.01, 16_000)
    canonical_provider = CanonicalQuranReferenceProvider()
    reference = canonical_provider.get_reference(
        CanonicalReferenceRequest(114, 1, get_ayah_count(114), "hafs")
    )
    warsh_reference = canonical_provider.get_reference(
        CanonicalReferenceRequest(114, 1, get_ayah_count(114), "warsh")
    )
    canonical_text = "".join(
        normalize_arabic(part.text).replace(" ", "") for part in reference.parts
    )
    warsh_text = "".join(
        normalize_arabic(part.text).replace(" ", "") for part in warsh_reference.parts
    )
    basmalah = normalize_arabic(get_ayah(1, 1, "hafs").text).replace(" ", "")
    istiadhah = normalize_arabic("أعوذ بالله من الشيطان الرجيم").replace(" ", "")
    chars = sorted(
        set(canonical_text) | set(warsh_text) | set(basmalah) | set(istiadhah) | {" "}
    )
    char_ids = {char: index + 1 for index, char in enumerate(chars)}
    assert max(char_ids.values()) < 50
    token_table = [f"unused-{i}" for i in range(251)]
    for char, token_id in char_ids.items():
        token_table[token_id] = char
    token_table[250] = "<blank>"
    split = len(canonical_text) // 2
    cuts = (
        canonical_text if groups == 1 else canonical_text[:split],
        canonical_text[split:],
    )
    zip_backend = FakeZipformerBackend(tuple(token_table), char_ids, cuts)
    evidence = ZipformerEvidence(
        repository="Quran-Lab/zipformer_p-arabic-v3",
        revision="0123456789abcdef0123456789abcdef01234567",
        approved=True,
        vocabulary_size=251,
        blank_id=250,
        sample_rate=16_000,
        feature_kind="kaldi-fbank",
    )
    reference_aligner = ZipformerNeuralAligner(
        evidence, backend_factory=lambda: zip_backend
    )
    intervals = [(0.0, 0.9, None), (1.0, 1.9, None)][:groups]
    segmenter = QuranRecitationSegmenter(
        loader=lambda value, rate: value if isinstance(value, AudioBuffer) else audio,
        backend_factory=lambda: FakeSegmentationBackend(intervals),
    )
    engine = HybridQuranAligner(
        segmenter=segmenter,
        reference_aligner=reference_aligner,
        forced_aligner=Wav2Vec2ForcedAligner(
            logits_provider=FakeWav2Vec2Provider(char_ids)
        ),
        canonical_reference_provider=canonical_provider,
        require_canonical_references=True,
    )
    return engine, audio, zip_backend


def test_full_surah_global_mapping_and_pr125_serialization():
    engine, audio, backend = build_engine(groups=2)
    progress = []
    result = engine.align_surah(
        audio,
        114,
        progress_callback=lambda current, total: progress.append((current, total)),
    )
    assert (
        result.metadata["global_mapping"]["algorithm"]
        == "ordered-global-character-dp-v1"
    )
    assert result.metadata["global_mapping"]["match_coverage"] == 1.0
    assert len(result.breath_groups) == 2
    assert result.ayahs
    assert len({item.ayah_number for item in result.ayahs}) > 1
    assert all(word.start < word.end for ayah in result for word in ayah.words)
    assert all(0 <= word.probability <= 1 for ayah in result for word in ayah.words)
    assert all(
        word.probability == pytest.approx(1 / 51)
        for ayah in result
        for word in ayah.words
    )
    assert all(
        result.ayahs[i].start_time <= result.ayahs[i + 1].start_time
        for i in range(len(result.ayahs) - 1)
    )
    ranges = result.metadata["global_mapping"]["group_canonical_ranges"]
    assert ranges["0"][0] <= ranges["0"][1] <= ranges["1"][0] <= ranges["1"][1]
    assert all(isinstance(ayah.is_breath_boundary, bool) for ayah in result.ayahs)
    for ayah, next_ayah in zip(result.ayahs, result.ayahs[1:], strict=False):
        assert ayah.start_time == pytest.approx(min(word.start for word in ayah.words))
        assert ayah.end_time == pytest.approx(max(word.end for word in ayah.words))
        assert ayah.pause_duration == pytest.approx(
            max(0.0, next_ayah.start_time - ayah.end_time)
        )
    assert result.ayahs[-1].pause_duration is None
    assert all(
        0 <= group.start_sec < group.end_sec <= audio.duration
        for group in result.breath_boundaries
    )
    assert all(
        group.duration_sec == pytest.approx(group.end_sec - group.start_sec)
        for group in result.breath_boundaries
    )
    assert any(
        len(
            {
                index
                for index, group in enumerate(result.breath_groups)
                if any(
                    group.start <= word.start and word.end <= group.end
                    for word in ayah.words
                )
            }
        )
        > 1
        for ayah in result.ayahs
    )
    assert backend.calls == 2
    assert progress and progress[-1][0] == progress[-1][1]

    # Match exactly the fields consumed by PR #125's _build_success_response.
    serialized = {
        "data": [
            {
                "ayah_number": item.ayah_number,
                "start_time": item.start_time,
                "end_time": item.end_time,
                "text": item.text,
                "words": [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                    for word in item.words
                ],
                "pause_duration": item.pause_duration,
                "is_breath_boundary": item.is_breath_boundary,
            }
            for item in result
        ],
        "breath_groups": [
            {
                "start_sec": group.start_sec,
                "end_sec": group.end_sec,
                "duration_sec": group.duration_sec,
            }
            for group in result.breath_boundaries
        ],
    }
    json.dumps(serialized, ensure_ascii=False, allow_nan=False)
    assert set(serialized) == {"data", "breath_groups"}
    assert set(serialized["data"][0]) == {
        "ayah_number",
        "start_time",
        "end_time",
        "text",
        "words",
        "pause_duration",
        "is_breath_boundary",
    }
    assert set(serialized["data"][0]["words"][0]) == {
        "word",
        "start",
        "end",
        "probability",
    }
    assert set(serialized["breath_groups"][0]) == {
        "start_sec",
        "end_sec",
        "duration_sec",
    }


def test_no_breaths_returns_empty_without_fabricated_ayahs():
    engine, audio, _ = build_engine(groups=0)
    result = engine.align_surah(audio, 114)
    assert not result.ayahs
    assert not result.breath_groups
    assert result.metadata["status"] == "no_speech_groups"
    assert result.warnings


def test_one_breath_can_cover_multiple_ayahs():
    engine, audio, _ = build_engine(groups=1)
    result = engine.align_surah(audio, 114)
    assert len(result.breath_groups) == 1
    assert len(result.ayahs) > 1
    assert result.metadata["global_mapping"]["mapped_breath_groups"] == [0]


def test_explicit_optional_intro_basmalah_is_stripped_not_assigned_as_ayah():
    engine, audio, backend = build_engine(groups=1)
    reference = engine.canonical_reference_provider.get_reference(
        CanonicalReferenceRequest(114, 1, get_ayah_count(114), "hafs")
    )
    basmalah = normalize_arabic(get_ayah(1, 1, "hafs").text).replace(" ", "")
    canonical = "".join(
        normalize_arabic(part.text).replace(" ", "") for part in reference.parts
    )
    backend.cuts = (basmalah + canonical, "")
    result = engine.align_surah(audio, 114)
    assert result.metadata["special_phrases"]["introductory_basmalah_detected"] is True
    assert result.metadata["special_phrases"]["introductory_basmalah_source"].endswith(
        "#1:1"
    )
    assert all(
        item.ayah_number != 1 or item.text != get_ayah(1, 1, "hafs").text
        for item in result
    )


def test_explicit_basmalah_is_part_of_reference_but_not_ayah_output():
    engine, audio, backend = build_engine(groups=1)
    reference = engine.canonical_reference_provider.get_reference(
        CanonicalReferenceRequest(114, 1, get_ayah_count(114), "hafs")
    )
    basmalah = normalize_arabic(get_ayah(1, 1, "hafs").text).replace(" ", "")
    canonical = "".join(
        normalize_arabic(part.text).replace(" ", "") for part in reference.parts
    )
    backend.cuts = (basmalah + canonical, "")
    result = engine.align_surah(audio, 114, include_basmalah=True)
    assert result.metadata["special_phrases"]["basmalah_included"] is True
    assert result.metadata["special_phrases"]["introductory_basmalah_detected"] is False
    assert all("بِسْمِ ٱللَّهِ" not in item.text for item in result)


def test_verified_istiadhah_is_in_reference_but_never_reported_as_ayah_text():
    engine, audio, backend = build_engine(groups=1)
    reference = engine.canonical_reference_provider.get_reference(
        CanonicalReferenceRequest(114, 1, get_ayah_count(114), "hafs")
    )
    canonical = "".join(
        normalize_arabic(part.text).replace(" ", "") for part in reference.parts
    )
    phrase_text = "أعوذ بالله من الشيطان الرجيم"
    phrase = VerifiedSpecialPhrase.from_source(
        "istiadhah", phrase_text, "unit-test canonical source"
    )
    backend.cuts = (normalize_arabic(phrase_text).replace(" ", "") + canonical, "")
    result = engine.align_surah(audio, 114, istiadhah_phrase=phrase)
    assert result.metadata["special_phrases"]["istiadhah_included"] is True
    assert all(phrase_text not in item.text for item in result)


def test_warsh_alignment_uses_its_own_reference_provenance():
    engine, audio, backend = build_engine(groups=1)
    reference = engine.canonical_reference_provider.get_reference(
        CanonicalReferenceRequest(114, 1, get_ayah_count(114), "warsh")
    )
    backend.cuts = (
        "".join(
            normalize_arabic(part.text).replace(" ", "") for part in reference.parts
        ),
        "",
    )
    result = engine.align_surah(audio, 114, riwaya="warsh")
    metadata = result.metadata["canonical_reference"]
    assert metadata["riwaya"] == "warsh"
    assert metadata["source_file"].endswith("quran_warsh.json")
    assert len(metadata["source_file_sha256"]) == 64


def test_partial_global_match_below_quality_threshold_fails_closed():
    engine, audio, backend = build_engine(groups=1)
    backend.cuts = (backend.cuts[0][:3], "")
    with pytest.raises(InvalidProviderOutputError, match="quality below threshold"):
        engine.align_surah(audio, 114)


def test_clean_partial_recitation_returns_only_anchored_ayahs_and_marks_partial():
    engine, audio, backend = build_engine(groups=1)
    backend.cuts = (backend.cuts[0][:20], "")
    result = engine.align_surah(audio, 114)
    assert result.metadata["status"] == "partial"
    assert result.metadata["global_mapping"]["match_coverage"] < 0.95
    assert result.warnings
    assert result.ayahs
    assert len(result.ayahs) < get_ayah_count(114)


def test_invalid_surah_riwaya_and_undefined_repetition_fail_closed():
    engine, audio, _ = build_engine(groups=2)
    with pytest.raises(ValueError, match="surah_id"):
        engine.align_surah(audio, 115)
    with pytest.raises(ValueError, match="riwaya"):
        engine.align_surah(audio, 114, riwaya="qiraat")
    with pytest.raises(InvalidProviderOutputError, match="no defined semantics"):
        engine.align_surah(audio, 114, repetition_attach=True)


def test_close_is_idempotent_and_cpu_import_stays_lazy():
    engine, _, _ = build_engine(groups=1)
    engine.close()
    engine.close()
    with pytest.raises(Exception, match="closed"):
        engine.align_surah("ignored.wav", 114)


def test_close_clears_cuda_cache_only_through_an_already_loaded_torch(monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: calls.append("empty_cache"),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    engine, _, _ = build_engine(groups=1)
    engine.close()
    engine.close()
    assert calls == ["empty_cache"]


def test_segmenter_receives_pr125_cleanup_parameters():
    engine, audio, _ = build_engine(groups=2)
    engine.align_surah(audio, 114, min_silence_ms=450, min_speech_ms=140, pad_ms=80)
    backend = engine.segmenter._backend
    assert backend.min_silence_duration_ms == 450
    assert backend.min_speech_duration_ms == 140
    assert backend.pad_duration_ms == 80
