"""Opt-in orchestration for the tripartite hybrid aligner."""

from __future__ import annotations

import gc
import sys
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError
from pathlib import Path
from typing import Any

import numpy as np

from munajjam.exceptions import InvalidProviderOutputError, ModelUnavailableError
from munajjam.hybrid_aligner.canonical_zipformer import (
    align_zipformer_to_reference,
    alignment_evidence,
)
from munajjam.hybrid_aligner.model_manager import ModelManager, ModelSpec
from munajjam.hybrid_aligner.neural_aligner import ZipformerEvidence, ZipformerNeuralAligner
from munajjam.hybrid_aligner.recitation_segmenter import (
    QuranRecitationSegmenter,
    TransformersRecitationSegmenterBackend,
)
from munajjam.hybrid_aligner.reference import (
    CanonicalQuranReferenceProvider,
    CanonicalReference,
    CanonicalReferenceRequest,
)
from munajjam.hybrid_aligner.surah import (
    AyahResult,
    IntroductionResult,
    SurahAlignmentResult,
    aggregate_words,
    full_surah_reference,
    map_surah,
)
from munajjam.hybrid_aligner.types import AlignmentSpan, AudioBuffer, HybridAlignmentResult
from munajjam.hybrid_aligner.wav2vec2_aligner import (
    TransformersWav2Vec2LogitsProvider,
    Wav2Vec2ForcedAligner,
    load_audio_file,
)
from munajjam.hybrid_aligner.zipformer_backend import (
    PUBLIC_ZIPFORMER_MODEL,
    PUBLIC_ZIPFORMER_MODEL_SHA256,
    PUBLIC_ZIPFORMER_REPOSITORY,
    PUBLIC_ZIPFORMER_REVISION,
    PUBLIC_ZIPFORMER_TOKENS,
    ZIPFORMER_BLANK_ID,
    ZIPFORMER_FRAME_DURATION_SEC,
    ZIPFORMER_TOKEN_SHA256,
    SherpaZipformerBackend,
)


def _load_production_audio(audio: Any, sample_rate: int) -> AudioBuffer:
    """Load paths or wrap samples without forcing model imports at construction."""
    if isinstance(audio, AudioBuffer):
        return audio
    if isinstance(audio, (str, Path)):
        return load_audio_file(audio, sample_rate)
    try:
        samples = np.asarray(audio, dtype=np.float32)
    except Exception as exc:
        raise InvalidProviderOutputError(
            "audio must be a path, AudioBuffer, or mono samples"
        ) from exc
    return AudioBuffer(samples=samples, sample_rate=sample_rate)


class HybridQuranAligner:
    """Explicit, opt-in pipeline; existing Munajjam strategies remain unchanged.

    ``align_surah`` automatically maps detected breaths against the complete
    canonical Surah. ``align`` retains the explicit-reference low-level API.
    Production construction rejects legacy caller-supplied token pairs.
    """

    def __init__(
        self,
        *,
        segmenter: QuranRecitationSegmenter | None = None,
        reference_aligner: ZipformerNeuralAligner | None = None,
        forced_aligner: Wav2Vec2ForcedAligner | None = None,
        canonical_reference_provider: CanonicalQuranReferenceProvider | None = None,
        require_canonical_references: bool = False,
    ) -> None:
        self._closed = False
        self._operation_lock = threading.RLock()
        self.segmenter = segmenter or QuranRecitationSegmenter()
        self.reference_aligner = reference_aligner
        self.forced_aligner = forced_aligner or Wav2Vec2ForcedAligner()
        self.canonical_reference_provider = canonical_reference_provider
        self.require_canonical_references = require_canonical_references

    @classmethod
    def from_pretrained(
        cls,
        *,
        zipformer_evidence: ZipformerEvidence | None = None,
        zipformer_backend_factory: Callable[[], Any] | None = None,
        cache_dir: str | None = None,
        device: str | None = None,
        allow_download: bool = True,
    ) -> HybridQuranAligner:
        """Construct a production pipeline with pinned, lazy model factories.

        The method itself performs no downloads and imports no heavyweight ML
        runtime. Zipformer weights, the segmenter model, and the Wav2Vec2 model
        are loaded on the first inference that needs each component.
        """
        manager = ModelManager(cache_dir)
        zipformer_spec = ModelSpec(
            repository=PUBLIC_ZIPFORMER_REPOSITORY,
            revision=PUBLIC_ZIPFORMER_REVISION,
            files=(PUBLIC_ZIPFORMER_TOKENS, PUBLIC_ZIPFORMER_MODEL),
            hashes={
                PUBLIC_ZIPFORMER_TOKENS: ZIPFORMER_TOKEN_SHA256,
                PUBLIC_ZIPFORMER_MODEL: PUBLIC_ZIPFORMER_MODEL_SHA256,
            },
        )
        evidence = zipformer_evidence or ZipformerEvidence(
            repository=PUBLIC_ZIPFORMER_REPOSITORY,
            revision=PUBLIC_ZIPFORMER_REVISION,
            approved=True,
            vocabulary_size=251,
            blank_id=ZIPFORMER_BLANK_ID,
            sample_rate=16_000,
            feature_kind="kaldi-fbank",
            token_table_sha256=ZIPFORMER_TOKEN_SHA256,
        )
        if (
            evidence.repository != PUBLIC_ZIPFORMER_REPOSITORY
            or evidence.revision != PUBLIC_ZIPFORMER_REVISION
        ):
            raise ModelUnavailableError(
                "Zipformer evidence does not match the pinned production artifact"
            )

        def make_zipformer_backend() -> Any:
            if zipformer_backend_factory is not None:
                return zipformer_backend_factory()
            zip_dir = manager.resolve(zipformer_spec, allow_download=allow_download)
            return SherpaZipformerBackend(zip_dir)

        segmenter = QuranRecitationSegmenter(
            loader=_load_production_audio,
            backend_factory=lambda: TransformersRecitationSegmenterBackend(
                device=device, cache_dir=cache_dir
            ),
        )
        reference = ZipformerNeuralAligner(evidence, backend_factory=make_zipformer_backend)
        forced = Wav2Vec2ForcedAligner(
            logits_provider_factory=lambda: TransformersWav2Vec2LogitsProvider(
                device=device, cache_dir=cache_dir
            )
        )
        return cls(
            segmenter=segmenter,
            reference_aligner=reference,
            forced_aligner=forced,
            canonical_reference_provider=CanonicalQuranReferenceProvider(),
            require_canonical_references=True,
        )

    @staticmethod
    def _reference_metadata(reference: CanonicalReference) -> dict[str, Any]:
        return {
            "riwaya": reference.riwaya,
            "parts": [
                {
                    "kind": part.kind,
                    "surah_id": part.surah_id,
                    "ayah_number": part.ayah_number,
                    "source": part.source,
                    "source_sha256": part.source_sha256,
                }
                for part in reference.parts
            ],
            "source_file": reference.source_file,
            "source_file_sha256": reference.source_file_sha256,
            "text_sha256": reference.text_sha256,
        }

    def align(
        self,
        audio: Any,
        targets: Sequence[tuple[Sequence[int], Sequence[str]]] | None = None,
        *,
        references: Sequence[CanonicalReferenceRequest] | None = None,
    ) -> HybridAlignmentResult:
        if self._closed:
            raise ModelUnavailableError("alignment engine is closed")
        if (targets is None) == (references is None):
            raise InvalidProviderOutputError(
                "provide exactly one of canonical references or legacy token targets"
            )
        if self.require_canonical_references and references is None:
            raise InvalidProviderOutputError(
                "production alignment requires canonical verse references, not caller token IDs"
            )
        if references is not None and self.canonical_reference_provider is None:
            raise ModelUnavailableError("no canonical Quran reference provider is configured")

        buffer = (
            audio
            if isinstance(audio, AudioBuffer)
            else self.segmenter._loader(audio, self.segmenter.config.sample_rate)
        )
        groups = self.segmenter.segment(buffer)
        selected_references: list[CanonicalReference] | None = None
        legacy_targets: list[tuple[Sequence[int], Sequence[str]]] | None = None
        if references is not None:
            if len(references) != len(groups):
                raise InvalidProviderOutputError(
                    "one canonical verse reference is required per breath group"
                )
            assert self.canonical_reference_provider is not None
            selected_references = [
                self.canonical_reference_provider.get_reference(request) for request in references
            ]
        else:
            assert targets is not None
            if len(targets) != len(groups):
                raise InvalidProviderOutputError(
                    "one reference target is required per breath group"
                )
            legacy_targets = list(targets)

        spans: list[AlignmentSpan] = []
        reference_evidence: list[dict[str, Any]] = []
        fused_groups = 0
        for index, group in enumerate(groups):
            canonical = selected_references[index] if selected_references is not None else None
            group_alignment = None
            if self.reference_aligner is not None:
                emissions = self.reference_aligner.align(buffer, group)
                evidence = {
                    "group_index": index,
                    "frame_duration_sec": ZIPFORMER_FRAME_DURATION_SEC,
                    "canonical_reference": (
                        self._reference_metadata(canonical) if canonical is not None else None
                    ),
                    "emissions": [
                        {
                            "token_id": item.token_id,
                            "start_frame": item.start_frame,
                            "end_frame": item.end_frame,
                            "score": item.score,
                        }
                        for item in emissions
                    ],
                }
                is_public = (
                    canonical is not None
                    and self.reference_aligner.evidence is not None
                    and self.reference_aligner.evidence.repository == PUBLIC_ZIPFORMER_REPOSITORY
                )
                if is_public:
                    group_alignment = align_zipformer_to_reference(
                        emissions, self.reference_aligner.token_table(), canonical
                    )
                    evidence["role"] = "canonical_character_dp_fusion"
                    evidence["alignment"] = alignment_evidence(
                        group_alignment, group, frame_duration_sec=ZIPFORMER_FRAME_DURATION_SEC
                    )
                    fused_groups += 1
                else:
                    evidence["role"] = "unaligned_phoneme_emissions"
                reference_evidence.append(evidence)
            if canonical is not None:
                token_ids, token_texts = self.forced_aligner.encode_text(canonical.text)
            else:
                assert legacy_targets is not None
                token_ids, token_texts = legacy_targets[index]
            next_start = groups[index + 1].start if index + 1 < len(groups) else None
            group_spans = list(
                self.forced_aligner.align_group(
                    buffer, group, token_ids, token_texts, next_group_start=next_start
                )
            )
            if group_alignment is not None:
                group_spans = [
                    AlignmentSpan(
                        span.token,
                        span.start,
                        span.end,
                        span.confidence,
                        "wav2vec2-ctc+zipformer-character-dp",
                    )
                    for span in group_spans
                ]
            spans.extend(group_spans)

        warnings: list[str] = []
        public_fusion_complete = (
            self.reference_aligner is not None
            and selected_references is not None
            and fused_groups == len(groups)
        )
        if self.reference_aligner is not None and not public_fusion_complete:
            warnings.append(
                "Public Zipformer emissions are available, but canonical phoneme-target DP "
                "fusion is not enabled until an authoritative token-target provider is supplied."
            )
        result = HybridAlignmentResult(
            groups,
            tuple(spans),
            metadata={
                "provider": "wav2vec2-ctc",
                "reference_groups": len(reference_evidence),
                "canonical_reference_groups": len(selected_references or ()),
                "canonical_references": (
                    [self._reference_metadata(item) for item in selected_references]
                    if selected_references is not None
                    else []
                ),
                "zipformer_role": (
                    "unaligned_phoneme_emissions"
                    if self.reference_aligner is not None
                    else "not_configured"
                ),
                "zipformer_reference_alignment_completed": public_fusion_complete,
                "zipformer_reference_alignment_status": (
                    (
                        "canonical_character_dp_verified"
                        if public_fusion_complete
                        else "public_model_alignment_pending"
                        if self.reference_aligner.evidence is not None
                        and self.reference_aligner.evidence.repository
                        == PUBLIC_ZIPFORMER_REPOSITORY
                        else "blocked_gated_artifact_access"
                    )
                    if self.reference_aligner is not None
                    else "not_configured"
                ),
                "zipformer_evidence_fused_into_final_spans": (
                    public_fusion_complete if self.reference_aligner is not None else None
                ),
                "zipformer_reference_evidence": reference_evidence,
                "zipformer_and_wav2vec2_vocabularies_are_independent": True,
                "warnings": warnings,
            },
        )
        result.validate(buffer.duration)
        return result

    def align_surah(
        self,
        audio_path: Any,
        surah_id: int,
        riwaya: str = "hafs",
        min_silence_ms: int = 300,
        min_speech_ms: int = 100,
        pad_ms: int = 100,
        repetition_attach: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
        *,
        cancel_event: Any = None,
    ) -> SurahAlignmentResult:
        """Synchronous production entry point; use ``from_pretrained()``.

        Callback receives (completed_breaths, total_breaths), beginning at zero.
        Exceptions from callbacks propagate. Cancellation is cooperative between
        model calls and DP rows and raises concurrent.futures.CancelledError.
        No partial result escapes on errors. A successful partial Surah contains
        only acoustically supported whole words, with omitted words in metadata.
        Calls are serialized per instance; close() is terminal and idempotent.
        PR #125 has no repetition_attach semantics yet: True fails explicitly.
        """
        with self._operation_lock:
            if self._closed:
                raise ModelUnavailableError("alignment engine is closed")
            if not isinstance(repetition_attach, bool):
                raise ValueError("repetition_attach must be bool")
            if repetition_attach:
                raise ValueError("repetition_attach=True has no defined PR #125 engine contract")
            for name, value, minimum in (
                ("min_silence_ms", min_silence_ms, 100),
                ("min_speech_ms", min_speech_ms, 100),
                ("pad_ms", pad_ms, 0),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                    raise ValueError(f"{name} must be an integer >= {minimum}")
            reference = full_surah_reference(surah_id, riwaya)
            if self.reference_aligner is None:
                raise ModelUnavailableError(
                    "full-Surah mapping requires a Zipformer reference aligner"
                )

            def check_cancelled() -> None:
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError("Surah alignment cancelled")
                if self._closed:
                    raise ModelUnavailableError("alignment engine is closed")

            check_cancelled()
            buffer = _load_production_audio(audio_path, self.segmenter.config.sample_rate)
            self.segmenter.configure_cleanup(min_silence_ms, min_speech_ms, pad_ms)
            groups = self.segmenter.segment(buffer)
            total = len(groups)
            if progress_callback is not None:
                progress_callback(0, total)
            check_cancelled()
            emissions = []
            for group in groups:
                check_cancelled()
                emissions.append(self.reference_aligner.align(buffer, group))
            mapping = map_surah(
                reference,
                emissions,
                self.reference_aligner.token_table(),
                check_cancelled=check_cancelled,
            )
            aligned = {}  # canonical word index -> (word result, physical breath index)
            all_spans = []
            for gi, (group, indices) in enumerate(
                zip(groups, mapping.group_word_indices, strict=True)
            ):
                check_cancelled()
                selected = [mapping.words[wi] for wi in indices]
                # Wav2Vec2 uses its own tokenizer, never Zipformer token IDs.
                text = " ".join(word.normalized for word in selected)
                token_ids, token_texts = self.forced_aligner.encode_text(text)
                spans = self.forced_aligner.align_group(
                    buffer,
                    group,
                    token_ids,
                    token_texts,
                    next_group_start=groups[gi + 1].start if gi + 1 < total else None,
                )
                words = aggregate_words(selected, spans)
                for wi, word in zip(indices, words, strict=True):
                    aligned[wi] = (word, gi)
                all_spans.extend(spans)
                if progress_callback is not None:
                    progress_callback(gi + 1, total)
            check_cancelled()
            HybridAlignmentResult(groups, tuple(all_spans)).validate(buffer.duration)
            ayahs = []
            introductions = []
            ordered = sorted(aligned)
            for pi, part in enumerate(reference.parts):
                indices = [wi for wi in ordered if mapping.words[wi].part_index == pi]
                if not indices:
                    continue
                words = tuple(aligned[wi][0] for wi in indices)
                if part.kind != "ayah":
                    introductions.append(
                        IntroductionResult(part.kind, " ".join(w.word for w in words), words)
                    )
                    continue
                last_index = indices[-1]
                next_index = next((wi for wi in ordered if wi > last_index), None)
                last_group = aligned[last_index][1]
                boundary = next_index is None or aligned[next_index][1] != last_group
                # Pause after the final supported word, measured to the next word
                # (or audio end); a breath boundary is a physical group transition.
                next_time = (
                    aligned[next_index][0].start if next_index is not None else buffer.duration
                )
                ayahs.append(
                    AyahResult(
                        part.ayah_number,
                        words[0].start,
                        words[-1].end,
                        " ".join(w.word for w in words),
                        max(0.0, next_time - words[-1].end),
                        boundary,
                        words,
                    )
                )
            if not ayahs:
                raise InvalidProviderOutputError("no canonical ayah words supported")
            return SurahAlignmentResult(
                tuple(ayahs),
                groups,
                tuple(introductions),
                {
                    "canonical_reference": self._reference_metadata(reference),
                    "canonical_parts": [
                        {"kind": p.kind, "ayah_number": p.ayah_number, "text": p.text}
                        for p in reference.parts
                    ],
                    "mapping": mapping.diagnostics,
                    "omitted_canonical_word_indices": [
                        w.index for w in mapping.words if w.index not in aligned
                    ],
                    "partial": any(
                        w.index not in aligned and reference.parts[w.part_index].kind == "ayah"
                        for w in mapping.words
                    ),
                    "confidence_aggregation": "geometric mean of character CTC posterior summaries; uncalibrated",
                    "timing_provenance": "per-breath wav2vec2-ctc; bounded 0.20s post-roll",
                    "zipformer_and_wav2vec2_vocabularies_are_independent": True,
                },
            )

    def close(self) -> None:
        """Terminal shutdown: attempt all releases and cleanup, then raise first error."""
        with self._operation_lock:
            if self._closed:
                return
            self._closed = True
            # Manus drops top-level references; Codex attempts every teardown.
            components = (self.segmenter, self.reference_aligner, self.forced_aligner)
            self.segmenter = None
            self.reference_aligner = None
            self.forced_aligner = None
            self.canonical_reference_provider = None
            errors = []
            for component in components:
                if component is not None:
                    try:
                        component.close()
                    except Exception as exc:
                        errors.append(exc)
            # Drop our own references before collecting, even if close() failed.
            component = None
            components = ()
            try:
                gc.collect()
            except Exception as exc:
                errors.append(exc)
            torch = sys.modules.get("torch")
            cuda = getattr(torch, "cuda", None)
            if cuda is not None:
                try:
                    if cuda.is_available():
                        cuda.empty_cache()
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise errors[0]
