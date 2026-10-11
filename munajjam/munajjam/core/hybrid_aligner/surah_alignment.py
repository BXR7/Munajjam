"""Automatic full-Surah alignment and API result construction.

The Zipformer emissions are concatenated in breath order and aligned once to the
complete canonical Surah. Only exact global DP matches anchor breath boundaries;
unmatched groups never receive guessed ayahs. Wav2Vec2 then aligns canonical
words inferred from those global intervals.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from munajjam.core.arabic import normalize_arabic
from munajjam.core.hybrid_aligner.canonical_zipformer import align_zipformer_to_reference
from munajjam.core.hybrid_aligner.reference import (
    CanonicalReferenceRequest,
    VerifiedSpecialPhrase,
)
from munajjam.core.hybrid_aligner.types import (
    AyahAlignment,
    HybridAlignmentResult,
    WordAlignment,
)
from munajjam.data.quran import get_ayah, get_ayah_count
from munajjam.exceptions import (
    InvalidProviderOutputError,
    ModelUnavailableError,
    NoBreathGroupsError,
)

if TYPE_CHECKING:
    from munajjam.core.hybrid_aligner.hybrid_pipeline import HybridQuranAligner


def _word_records(reference: Any) -> tuple[list[dict[str, Any]], int]:
    """Return canonical word offsets in the same normalized-character space as DP."""
    records: list[dict[str, Any]] = []
    cursor = 0
    for part in reference.parts:
        if part.kind != "ayah":
            cursor += len(normalize_arabic(part.text).replace(" ", ""))
            continue
        for raw_word in part.text.split():
            normalized = normalize_arabic(raw_word).replace(" ", "")
            if not normalized:
                continue
            start, end = cursor, cursor + len(normalized)
            records.append(
                {
                    "start_index": start,
                    "end_index": end,
                    "center": (start + end - 1) // 2,
                    "word": raw_word,
                    "ayah_number": part.ayah_number,
                }
            )
            cursor = end
        # Normalized whitespace is omitted from canonical DP positions.
    return records, cursor


def _progress(callback: Callable[[int, int], None] | None, current: int, total: int) -> None:
    if callback is not None:
        callback(current, total)


def _exact_prefix_emission_count(
    emissions: tuple[Any, ...], token_table: tuple[str, ...], phrase: str
) -> int | None:
    """Return a whole-emission prefix length only when symbols exactly spell phrase."""
    target = normalize_arabic(phrase).replace(" ", "")
    if not target:
        return None
    seen = ""
    for index, emission in enumerate(emissions, start=1):
        symbol = normalize_arabic(token_table[emission.token_id]).replace(" ", "")
        if not symbol or not target.startswith(seen + symbol):
            return None
        seen += symbol
        if seen == target:
            return index
    return None


def _best_dense_anchor_window(positions: list[int], *, max_gap: int = 8) -> list[int]:
    """Keep the strongest locally coherent canonical match window for a breath."""
    ordered = sorted(set(positions))
    if not ordered:
        return []
    clusters: list[list[int]] = [[ordered[0]]]
    for position in ordered[1:]:
        if position - clusters[-1][-1] <= max_gap:
            clusters[-1].append(position)
        else:
            clusters.append([position])
    return max(clusters, key=lambda cluster: (len(cluster), -cluster[0]))


def align_surah(
    engine: HybridQuranAligner,
    audio_path: Any,
    surah_id: int,
    riwaya: str = "hafs",
    min_silence_ms: int = 300,
    min_speech_ms: int = 100,
    pad_ms: int = 100,
    repetition_attach: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
    *,
    include_basmalah: bool = False,
    istiadhah_phrase: VerifiedSpecialPhrase | None = None,
) -> HybridAlignmentResult:
    """Align a recording without caller-provided breath-to-ayah assignments."""
    if isinstance(surah_id, bool) or not isinstance(surah_id, int) or not 1 <= surah_id <= 114:
        raise ValueError("surah_id must be an integer in 1..114")
    if riwaya not in ("hafs", "warsh"):
        raise ValueError("riwaya must be 'hafs' or 'warsh'")
    for name, value, minimum in (
        ("min_silence_ms", min_silence_ms, 0),
        ("min_speech_ms", min_speech_ms, 0),
        ("pad_ms", pad_ms, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be a non-negative integer")
    if not isinstance(repetition_attach, bool):
        raise ValueError("repetition_attach must be bool")
    # The server contract carries this flag, but PR #125 does not define what
    # repeated readings should attach to. Refuse to invent that policy.
    if repetition_attach:
        raise InvalidProviderOutputError(
            "repetition_attach=true has no defined semantics in PR #125; refusing to guess"
        )
    if engine.reference_aligner is None or engine.canonical_reference_provider is None:
        raise ModelUnavailableError("automatic Surah alignment requires canonical Zipformer DP")

    ayah_count = get_ayah_count(surah_id)
    request = CanonicalReferenceRequest(
        surah_id=surah_id,
        ayah_start=1,
        ayah_end=ayah_count,
        riwaya=riwaya,
        include_basmalah=include_basmalah,
        include_istiadhah=istiadhah_phrase is not None,
        istiadhah_phrase=istiadhah_phrase,
    )
    reference = engine.canonical_reference_provider.get_reference(request)
    sample_rate = engine.segmenter.config.sample_rate
    buffer = engine.segmenter._loader(audio_path, sample_rate)

    # Apply PR #125's segmentation settings to the real segmenter postprocessor.
    backend = engine.segmenter._get_backend()
    if hasattr(backend, "min_silence_duration_ms"):
        backend.min_silence_duration_ms = min_silence_ms
    if hasattr(backend, "min_speech_duration_ms"):
        backend.min_speech_duration_ms = min_speech_ms
    if hasattr(backend, "pad_duration_ms"):
        backend.pad_duration_ms = pad_ms

    try:
        groups = engine.segmenter.segment(buffer)
    except NoBreathGroupsError:
        empty = HybridAlignmentResult(
            breath_groups=(),
            warnings=("No breath groups were detected; no ayah assignments were fabricated.",),
            metadata={
                "surah_id": surah_id,
                "riwaya": riwaya,
                "canonical_reference": engine._reference_metadata(reference),
                "status": "no_speech_groups",
            },
        )
        empty.validate(buffer.duration)
        return empty

    total = len(groups) + 2
    _progress(progress_callback, 0, total)
    emissions = []
    emission_groups: list[int] = []
    group_emissions: list[tuple[Any, ...]] = []
    for group_index, group in enumerate(groups):
        current = engine.reference_aligner.align(buffer, group)
        group_emissions.append(current)
        emissions.extend(current)
        emission_groups.extend([group_index] * len(current))
        _progress(progress_callback, group_index + 1, total)
    token_table = engine.reference_aligner.token_table()
    detected_basmalah = False
    basmalah_source = None
    # The Fatiha's Basmalah is canonical ayah 1; never strip it there. For
    # other Surahs, a fully recognized, source-verified prefix is introductory
    # and is excluded from ayah assignment unless the caller explicitly asked
    # for it as part of the reference.
    if surah_id != 1 and not include_basmalah and group_emissions:
        basmalah_ayah = get_ayah(1, 1, riwaya)
        if basmalah_ayah is not None:
            prefix_count = _exact_prefix_emission_count(
                group_emissions[0], token_table, basmalah_ayah.text
            )
            if prefix_count is not None:
                group_emissions[0] = group_emissions[0][prefix_count:]
                detected_basmalah = True
                basmalah_source = f"munajjam.data/quran_{riwaya}.json#1:1"
                emissions = [item for group in group_emissions for item in group]
                emission_groups = [
                    index for index, group in enumerate(group_emissions) for _ in group
                ]
    if not emissions:
        result = HybridAlignmentResult(
            breath_groups=tuple(groups),
            warnings=(
                "Zipformer produced no canonical emissions; no ayah assignments were fabricated.",
            ),
            metadata={
                "surah_id": surah_id,
                "riwaya": riwaya,
                "canonical_reference": engine._reference_metadata(reference),
                "introductory_basmalah_detected": detected_basmalah,
                "introductory_basmalah_source": basmalah_source,
                "status": "no_zipformer_emissions",
            },
        )
        result.validate(buffer.duration)
        return result

    canonical_length = sum(
        len(normalize_arabic(part.text).replace(" ", "")) for part in reference.parts
    )
    minimum_anchor_coverage = min(0.70, 8.0 / max(canonical_length, 1))
    dp = align_zipformer_to_reference(
        emissions,
        token_table,
        reference,
        quality_threshold=0.45,
        coverage_threshold=minimum_anchor_coverage,
    )
    raw_anchors: dict[int, list[int]] = defaultdict(list)
    for cell in dp.cells:
        if cell.operation != "MATCH" or cell.observed is None or cell.canonical is None:
            continue
        emission_index = cell.observed.emission_index
        group_index = emission_groups[emission_index]
        raw_anchors[group_index].append(cell.canonical.index)

    anchors = {
        group_index: _best_dense_anchor_window(positions)
        for group_index, positions in raw_anchors.items()
    }
    mapped_groups = sorted(anchors)
    if not mapped_groups:
        raise InvalidProviderOutputError("global Zipformer DP found no exact canonical matches")
    group_ranges: dict[int, tuple[int, int]] = {}
    left = min(anchors[mapped_groups[0]])
    for position, group_index in enumerate(mapped_groups):
        if position + 1 < len(mapped_groups):
            next_group = mapped_groups[position + 1]
            boundary = (max(anchors[group_index]) + min(anchors[next_group])) // 2 + 1
        else:
            boundary = max(anchors[group_index]) + 1
        group_ranges[group_index] = (left, max(left, boundary))
        left = max(left, boundary)

    word_records, observed_canonical_length = _word_records(reference)
    if observed_canonical_length != canonical_length:
        raise InvalidProviderOutputError(
            "canonical text/DP normalization disagreement; refusing word assignment"
        )
    words_by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for word_record in word_records:
        for group_index, (start, end) in group_ranges.items():
            if start <= word_record["center"] < end:
                words_by_group[group_index].append(word_record)
                break

    aligned_words: list[dict[str, Any]] = []
    spans = []
    for group_index in mapped_groups:
        selected = words_by_group.get(group_index, [])
        if not selected:
            continue
        group = groups[group_index]
        words = [item["word"] for item in selected]
        text = " ".join(words)
        token_ids, token_texts = engine.forced_aligner.encode_text(text)
        prefix_ids: list[int] = []
        token_ranges: list[tuple[int, int]] = []
        for word_index, _word in enumerate(words):
            prefix = " ".join(words[: word_index + 1])
            prefix_ids, _ = engine.forced_aligner.encode_text(prefix)
            end_token = len(prefix_ids)
            start_token = token_ranges[-1][1] if token_ranges else 0
            token_ranges.append((start_token, end_token))
        if len(prefix_ids) != len(token_ids) or prefix_ids != token_ids:
            raise InvalidProviderOutputError(
                "Wav2Vec2 tokenizer does not preserve canonical word-prefix boundaries"
            )
        next_start = groups[group_index + 1].start if group_index + 1 < len(groups) else None
        group_spans = engine.forced_aligner.align_group(
            buffer, group, token_ids, token_texts, next_group_start=next_start
        )
        spans.extend(group_spans)
        for item, (token_start, token_end) in zip(selected, token_ranges, strict=True):
            token_spans = [
                span
                for span in group_spans[token_start:token_end]
                if span.token not in {"|", "<pad>", "[PAD]"}
            ]
            if not token_spans:
                raise InvalidProviderOutputError("word has no aligned character spans")
            confidence_values = [
                span.confidence for span in token_spans if span.confidence is not None
            ]
            if not confidence_values:
                raise InvalidProviderOutputError("word has no CTC character confidence evidence")
            item_out = dict(item)
            item_out.update(
                {
                    "start": min(span.start for span in token_spans),
                    "end": max(span.end for span in token_spans),
                    # Arithmetic mean of character CTC scores; not calibrated.
                    "probability": float(np.mean(confidence_values)),
                    "group_index": group_index,
                }
            )
            aligned_words.append(item_out)

    by_ayah: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in aligned_words:
        ayah_number = item["ayah_number"]
        if ayah_number is not None:
            by_ayah[ayah_number].append(item)
    ordered_ayah_numbers = sorted(by_ayah)
    ayahs: list[AyahAlignment] = []
    for number in ordered_ayah_numbers:
        items = sorted(by_ayah[number], key=lambda item: item["start_index"])
        word_results = tuple(
            WordAlignment(item["word"], item["start"], item["end"], item["probability"])
            for item in items
        )
        start_time = min(item["start"] for item in items)
        end_time = max(item["end"] for item in items)
        last = max(items, key=lambda item: item["end"])
        last_group = groups[last["group_index"]]
        boundary = abs(last_group.end - last["end"]) <= max(0.35, pad_ms / 1000)
        ayahs.append(
            AyahAlignment(
                ayah_number=number,
                start_time=start_time,
                end_time=end_time,
                text=" ".join(item["word"] for item in items),
                words=word_results,
                is_breath_boundary=boundary,
            )
        )
    for index, ayah in enumerate(ayahs[:-1]):
        pause = max(0.0, ayahs[index + 1].start_time - ayah.end_time)
        ayahs[index] = AyahAlignment(
            ayah.ayah_number,
            ayah.start_time,
            ayah.end_time,
            ayah.text,
            ayah.words,
            pause,
            ayah.is_breath_boundary,
        )

    warnings = []
    omitted = sorted(set(range(len(groups))) - set(mapped_groups))
    if omitted:
        warnings.append(
            f"No confident canonical anchor for breath group indices {omitted}; omitted."
        )
    if dp.match_coverage < 0.95:
        warnings.append(
            f"Canonical match coverage is {dp.match_coverage:.1%}; result is partial and requires review."
        )
    result = HybridAlignmentResult(
        breath_groups=tuple(groups),
        spans=tuple(spans),
        warnings=tuple(warnings),
        ayahs=tuple(ayahs),
        metadata={
            "surah_id": surah_id,
            "riwaya": riwaya,
            "canonical_reference": engine._reference_metadata(reference),
            "global_mapping": {
                "algorithm": "ordered-global-character-dp-v1",
                "normalized_cost": dp.normalized_cost,
                "total_edit_cost": dp.total_cost,
                "match_coverage": dp.match_coverage,
                "observed_units": dp.observed_units,
                "canonical_units": dp.canonical_units,
                "quality_threshold": dp.quality_threshold,
                "coverage_threshold": dp.coverage_threshold,
                "mapped_breath_groups": mapped_groups,
                "group_canonical_ranges": {
                    str(key): list(value) for key, value in group_ranges.items()
                },
            },
            "word_probability_aggregation": "arithmetic_mean_of_character_ctc_scores_uncalibrated",
            "zipformer_vocabulary_size": 251,
            "wav2vec2_vocabulary_size": 51,
            "vocabularies_are_independent": True,
            "time_unit": "seconds",
            "repetition_attach": False,
            "special_phrases": {
                "basmalah_included": include_basmalah,
                "introductory_basmalah_detected": detected_basmalah,
                "introductory_basmalah_source": basmalah_source,
                "istiadhah_included": istiadhah_phrase is not None,
            },
            "status": "partial" if warnings else "complete",
        },
    )
    result.validate(buffer.duration)
    _progress(progress_callback, total, total)
    return result
