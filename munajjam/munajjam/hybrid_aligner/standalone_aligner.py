"""
Standalone Zipformer Full-Surah Aligner (Architecture A)
Issue #120 / PR #125 / PR #127 Integration

Provides a self-contained, lightweight, zero-download Quranic alignment engine
powered exclusively by the pre-trained Zipformer ONNX acoustic model (263 MB).

Key Features:
- Candidate D1 Unified Global Viterbi search (no boundary-clipping artifacts).
- Automatic Basmalah prefix detection (eliminates 4s systematic onset drift).
- Multi-Gate Safety (physical pace gating + length-calibrated thresholding).
- Conforms 100% to PR #125 / PR #127 SurahAlignmentResult contract.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any

import numpy as np

from munajjam.core.arabic import normalize_arabic
from munajjam.exceptions import InvalidProviderOutputError, ModelUnavailableError
from munajjam.hybrid_aligner.surah import (
    AyahResult,
    IntroductionResult,
    SurahAlignmentResult,
    WordResult,
    full_surah_reference,
)
from munajjam.hybrid_aligner.types import BreathGroup
from munajjam.hybrid_aligner.zipformer_backend import (
    ZIPFORMER_BLANK_ID,
    ZIPFORMER_DECODE_CHUNK_LEN,
    ZIPFORMER_FRAME_DURATION_SEC,
    ZIPFORMER_INPUT_FRAMES,
    ZipformerOnnxBackend,
)

BASMALA_WORDS = ["بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَٰنِ", "ٱلرَّحِيمِ"]
FRAME_MS = int(ZIPFORMER_FRAME_DURATION_SEC * 1000)  # 40 ms


def _load_audio_16k(audio_input: Any) -> np.ndarray:
    """Load audio from file path, AudioBuffer, or raw numpy array and convert to 16kHz mono."""
    if hasattr(audio_input, "samples"):
        samples = np.asarray(audio_input.samples, dtype=np.float32)
        sr = getattr(audio_input, "sample_rate", 16000)
    elif isinstance(audio_input, (str, pathlib.Path)):
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ModelUnavailableError(
                "soundfile is required to load audio from file paths; "
                "install it with: pip install soundfile"
            ) from exc
        samples, sr = sf.read(str(audio_input), dtype="float32")
    else:
        samples = np.asarray(audio_input, dtype=np.float32)
        sr = 16000

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    if sr != 16000:
        try:
            import torch
            import torchaudio
        except ImportError as exc:
            raise ModelUnavailableError(
                "torch and torchaudio are required for audio resampling; "
                "install them with: pip install torch torchaudio"
            ) from exc
        tensor = torch.from_numpy(samples).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, 16000)
        tensor = resampler(tensor)
        samples = tensor.squeeze(0).numpy()

    return samples


class StandaloneZipformerAligner:
    """
    Production-ready, single-model Quran aligner for Issue #120.
    Replaces the tripartite pipeline with verified Zipformer ONNX alone.
    """

    MAX_TRELLIS_CELLS: int = 100_000_000

    def __init__(self, model_dir: str | pathlib.Path | None = None) -> None:
        if model_dir is None:
            # Default to local model directory in repo
            repo_root = pathlib.Path(__file__).resolve().parents[3]
            candidate_dirs = [
                repo_root / "experiments" / "models" / "model_zipformer",
                repo_root / "munajjam" / "models",
                repo_root / "models",
            ]
            for c in candidate_dirs:
                if (c / "zipformer_p_arabic_v3.onnx").is_file():
                    model_dir = c
                    break
            if model_dir is None:
                model_dir = candidate_dirs[0]

        self.model_dir = pathlib.Path(model_dir)
        self.backend = ZipformerOnnxBackend(
            self.model_dir,
            model_name="zipformer_p_arabic_v3.onnx",
        )
        self.tokens = self.backend.token_table
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.tokens)}
        self._closed = False

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | pathlib.Path | None = None,
        **kwargs: Any,
    ) -> StandaloneZipformerAligner:
        """PR #125 factory entrypoint."""
        return cls(model_dir=model_dir)

    def close(self) -> None:
        """Clean up ONNX runtime sessions and resources."""
        self._closed = True
        if hasattr(self, "backend") and self.backend is not None:
            self.backend._session = None

    def extract_posteriors(self, samples: np.ndarray, finalize_flush: bool = True) -> np.ndarray:
        """Extract frame log-probabilities [T, 251] via ONNX Runtime."""
        if finalize_flush:
            tail = np.zeros(int(0.66 * 16000), dtype=np.float32)
            samples = np.concatenate([samples, tail])

        features = self.backend._feature_extractor(samples, 16000)
        total_input_frames = len(features)
        state = self.backend._initial_state()
        processed_lens = self.backend._initial_processed_lens()

        all_log_probs = []
        output_names = [item.name for item in self.backend._session.get_outputs()]
        start_frame = 0

        while start_frame + ZIPFORMER_INPUT_FRAMES <= total_input_frames:
            chunk = features[start_frame : start_frame + ZIPFORMER_INPUT_FRAMES]
            chunk_batch = chunk[np.newaxis, :, :]
            feeds = {
                self.backend._x_input: chunk_batch,
                self.backend._processed_input: processed_lens,
                **state,
            }
            outputs = self.backend._session.run(output_names, feeds)
            by_name = dict(zip(output_names, outputs, strict=True))
            lprobs = np.asarray(by_name[self.backend._logit_output], dtype=np.float32)[0]
            all_log_probs.append(lprobs)

            for name in state:
                state[name] = np.asarray(by_name[self.backend._state_output_names[name]])
            processed_lens = np.asarray(by_name[self.backend._processed_output])
            start_frame += ZIPFORMER_DECODE_CHUNK_LEN

        if all_log_probs:
            return np.concatenate(all_log_probs, axis=0)
        return np.zeros((0, 251), dtype=np.float32)

    def tokenize_words(self, words: list[str]) -> tuple[list[int], list[int]]:
        """Tokenize canonical Arabic words to token IDs with word index mapping."""
        token_ids = []
        word_map = []
        for w_idx, w in enumerate(words):
            norm = normalize_arabic(w).replace(" ", "")
            for c in norm:
                if c in self.token_to_id:
                    token_ids.append(self.token_to_id[c])
                    word_map.append(w_idx)
        # Check that every word produced at least one token
        mapped_indices = set(word_map)
        unsupported = [words[i] for i in range(len(words)) if i not in mapped_indices]
        if unsupported:
            raise InvalidProviderOutputError(
                f"words produced zero tokens (unsupported characters): {unsupported}"
            )
        return token_ids, word_map

    def ctc_align_segment(
        self,
        log_probs: np.ndarray,
        words: list[str],
        blank_bias: float = 2.5,
        cancel_event: Any = None,
    ) -> dict[str, Any]:
        """Candidate D1: Unified global Viterbi CTC alignment."""
        T = log_probs.shape[0]
        tokens, word_map = self.tokenize_words(words)
        U = len(tokens)
        if T == 0 or U == 0:
            return {"score_per_frame": -999.0, "total_ll": -999.0, "word_timings": []}

        S = 2 * U + 1
        if T * S > self.MAX_TRELLIS_CELLS:
            raise InvalidProviderOutputError(
                f"trellis size {T} x {S} = {T * S} exceeds maximum limit of {self.MAX_TRELLIS_CELLS} cells"
            )

        # Blank log-odds adjustment
        adjusted = log_probs.copy()
        if blank_bias != 0.0:
            adjusted[:, ZIPFORMER_BLANK_ID] -= blank_bias
            m = np.max(adjusted, axis=-1, keepdims=True)
            adjusted -= m + np.log(np.sum(np.exp(adjusted - m), axis=-1, keepdims=True))

        trellis = np.full((T, S), -np.inf, dtype=np.float32)
        backpointers = np.zeros((T, S), dtype=np.int8)

        trellis[0, 0] = adjusted[0, ZIPFORMER_BLANK_ID]
        trellis[0, 1] = adjusted[0, tokens[0]]

        # Precompute state token mappings and skip validity mask
        tok_ids = np.empty(S, dtype=np.int64)
        tok_ids[0::2] = ZIPFORMER_BLANK_ID
        tok_ids[1::2] = tokens

        skip_mask = np.zeros(S, dtype=bool)
        for s in range(3, S, 2):
            curr_char = tokens[s // 2]
            prev_char = tokens[(s // 2) - 1]
            if curr_char != prev_char:
                skip_mask[s] = True

        c_prev = np.full(S, -np.inf, dtype=np.float32)
        c_skip = np.full(S, -np.inf, dtype=np.float32)

        for t in range(1, T):
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                raise InvalidProviderOutputError("alignment cancelled")

            prev_row = trellis[t - 1]
            c_stay = prev_row
            c_prev[0] = -np.inf
            c_prev[1:] = prev_row[:-1]

            best_prev = np.maximum(c_stay, c_prev)
            bp = (c_prev > c_stay).astype(np.int8)

            c_skip[0:2] = -np.inf
            c_skip[2:] = prev_row[:-2]
            skip_better = skip_mask & (c_skip > best_prev)
            best_prev = np.where(skip_better, c_skip, best_prev)
            bp = np.where(skip_better, 2, bp).astype(np.int8)

            trellis[t] = best_prev + adjusted[t, tok_ids]
            backpointers[t] = bp

        best_terminal = S - 1 if trellis[T - 1, S - 1] >= trellis[T - 1, S - 2] else S - 2
        total_ll = float(trellis[T - 1, best_terminal])
        score_per_frame = float(total_ll / T)

        # Traceback
        curr_s = best_terminal
        frame_to_token = {}
        for t in range(T - 1, -1, -1):
            if curr_s % 2 == 1:
                frame_to_token[t] = curr_s // 2
            curr_s -= backpointers[t, curr_s]

        num_words = len(words)
        word_frames: dict[int, list[int]] = {w: [] for w in range(num_words)}
        word_emissions: dict[int, list[float]] = {w: [] for w in range(num_words)}

        for t, tok_idx in frame_to_token.items():
            if tok_idx < len(word_map):
                w_idx = word_map[tok_idx]
                word_frames[w_idx].append(t)
                tok_id = tokens[tok_idx]
                word_emissions[w_idx].append(float(np.exp(adjusted[t, tok_id])))

        word_timings = []
        for w in range(num_words):
            frames = word_frames[w]
            emits = word_emissions[w]
            if frames:
                s_ms = min(frames) * FRAME_MS
                e_ms = (max(frames) + 1) * FRAME_MS
                mean_conf = float(np.mean(emits)) if emits else 0.0
                word_timings.append(
                    {
                        "word_idx": w,
                        "word": words[w],
                        "start_ms": s_ms,
                        "end_ms": e_ms,
                        "conf": mean_conf,
                    }
                )
            else:
                word_timings.append(
                    {
                        "word_idx": w,
                        "word": words[w],
                        "start_ms": -1,
                        "end_ms": -1,
                        "conf": 0.0,
                    }
                )

        return {
            "score_per_frame": round(score_per_frame, 3),
            "total_ll": round(total_ll, 2),
            "word_timings": word_timings,
        }

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
        """
        Synchronous production entry point conforming 100% to PR #125 contract.
        """
        if self._closed:
            raise ModelUnavailableError("alignment engine is closed")
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            raise InvalidProviderOutputError("alignment cancelled")
        if repetition_attach:
            raise ValueError("repetition_attach=True has no defined PR #125 engine contract")

        samples = _load_audio_16k(audio_path)
        dur_sec = len(samples) / 16000.0 if len(samples) > 0 else 0.0

        if dur_sec <= 0.0 or len(samples) == 0:
            raise InvalidProviderOutputError("empty or zero-length audio cannot be aligned")

        ref = full_surah_reference(surah_id, riwaya)
        ayah_parts = [p for p in ref.parts if p.kind == "ayah"]
        all_canonical_words = [w for p in ayah_parts for w in p.text.split()]
        num_words = len(all_canonical_words)

        # 1. Pace Gate Check
        pace = dur_sec / max(1, num_words)
        if pace < 0.50 or pace > 3.00:
            raise InvalidProviderOutputError(
                f"audio rejected by pace gate: {pace:.2f} s/word (expected 0.50-3.00 s/word)"
            )

        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            raise InvalidProviderOutputError("alignment cancelled")

        # 2. Extract Zipformer Posteriors
        lprobs = self.extract_posteriors(samples, finalize_flush=True)
        T = lprobs.shape[0]
        if T == 0:
            raise InvalidProviderOutputError("no acoustic frames extracted from audio")

        if progress_callback is not None:
            progress_callback(1, 4)

        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            raise InvalidProviderOutputError("alignment cancelled")

        # 3. Basmalah Detection (Surahs 2-114)
        has_intro_basmala = False
        basmala_offset_ms = 0
        basmala_res_words: list[WordResult] = []

        if surah_id not in (1, 9):
            intro_frames = min(T, int(5.0 / ZIPFORMER_FRAME_DURATION_SEC))
            if intro_frames > 20:
                intro_lprobs = lprobs[:intro_frames]
                basm_res = self.ctc_align_segment(
                    intro_lprobs, BASMALA_WORDS, blank_bias=2.5, cancel_event=cancel_event
                )
                if basm_res["score_per_frame"] >= -1.25:
                    has_intro_basmala = True
                    for wt in basm_res["word_timings"]:
                        if wt["start_ms"] >= 0:
                            basmala_res_words.append(
                                WordResult(
                                    word=wt["word"],
                                    start=round(wt["start_ms"] / 1000.0, 3),
                                    end=round(wt["end_ms"] / 1000.0, 3),
                                    probability=round(wt.get("conf", 0.9), 3),
                                )
                            )
                    last_w = basm_res["word_timings"][-1]
                    if last_w["end_ms"] > 0:
                        basmala_offset_ms = last_w["end_ms"]

        basmala_start_frame = int(basmala_offset_ms / FRAME_MS)
        surah_lprobs = lprobs[basmala_start_frame:]

        if progress_callback is not None:
            progress_callback(2, 4)

        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            raise InvalidProviderOutputError("alignment cancelled")

        # 4. Alignment Execution (Candidate D1 Unified Global Viterbi)
        res_alignment = self.ctc_align_segment(
            surah_lprobs, all_canonical_words, blank_bias=2.5, cancel_event=cancel_event
        )
        score_global = res_alignment["score_per_frame"]

        # 5. Tiered Threshold Gate
        tau = -0.66 if num_words <= 25 else (-1.00 if num_words <= 100 else -1.35)
        if score_global < tau:
            raise InvalidProviderOutputError(
                f"audio rejected: score {score_global:.3f} below threshold {tau:.2f}"
            )

        if progress_callback is not None:
            progress_callback(3, 4)

        # 6. Construct Structured AyahResult and WordResult Hierarchy
        aligned_words = res_alignment["word_timings"]
        ayah_results = []
        w_cursor = 0

        for a_idx, part in enumerate(ayah_parts):
            p_words = part.text.split()
            current_ayah_words = []
            for _ in p_words:
                wt = aligned_words[w_cursor]
                w_cursor += 1
                if wt["start_ms"] < 0 or wt["end_ms"] < 0:
                    raise InvalidProviderOutputError(
                        f"canonical word '{wt['word']}' (index {wt['word_idx']}) has no valid acoustic frames"
                    )
                s_sec = round((wt["start_ms"] + basmala_offset_ms) / 1000.0, 3)
                e_sec = round((wt["end_ms"] + basmala_offset_ms) / 1000.0, 3)
                current_ayah_words.append(
                    WordResult(
                        word=wt["word"],
                        start=s_sec,
                        end=e_sec,
                        probability=round(wt.get("conf", 0.95), 3),
                    )
                )

            a_start = current_ayah_words[0].start if current_ayah_words else 0.0
            a_end = current_ayah_words[-1].end if current_ayah_words else a_start
            ayah_results.append(
                AyahResult(
                    ayah_number=part.ayah_number or (a_idx + 1),
                    start_time=a_start,
                    end_time=a_end,
                    text=part.text,
                    pause_duration=0.0,
                    is_breath_boundary=True,
                    words=tuple(current_ayah_words),
                )
            )

        # Calculate inter-ayah pause durations
        final_ayah_results = []
        for i, a in enumerate(ayah_results):
            next_start = ayah_results[i + 1].start_time if i + 1 < len(ayah_results) else dur_sec
            pause = max(0.0, round(next_start - a.end_time, 3))
            is_boundary = pause >= (min_silence_ms / 1000.0)
            final_ayah_results.append(
                AyahResult(
                    ayah_number=a.ayah_number,
                    start_time=a.start_time,
                    end_time=a.end_time,
                    text=a.text,
                    pause_duration=pause,
                    is_breath_boundary=is_boundary,
                    words=a.words,
                )
            )

        # Generate breath groups from Ayah boundaries
        breath_groups = [
            BreathGroup(start=a.start_time, end=a.end_time, score=1.0) for a in final_ayah_results
        ]

        introductions = []
        if has_intro_basmala and basmala_res_words:
            introductions.append(
                IntroductionResult(
                    kind="basmalah",
                    text=" ".join(BASMALA_WORDS),
                    words=tuple(basmala_res_words),
                )
            )

        if progress_callback is not None:
            progress_callback(4, 4)

        return SurahAlignmentResult(
            data=tuple(final_ayah_results),
            breath_groups=tuple(breath_groups),
            introductions=tuple(introductions),
            metadata={
                "provider": "zipformer-onnx-standalone",
                "architecture": "Architecture A",
                "score_per_frame": score_global,
                "tau_threshold": tau,
                "pace_sec_per_word": round(pace, 3),
                "has_intro_basmala": has_intro_basmala,
                "basmala_offset_ms": basmala_offset_ms,
            },
        )
