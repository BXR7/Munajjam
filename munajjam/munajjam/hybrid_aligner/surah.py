"""Full-reference monotonic mapping and the PR #125 result contract.

Times are seconds. Confidence values summarize acoustic evidence, not calibrated
probabilities. Unsafe/ambiguous mappings raise InvalidProviderOutputError before
any result is published. No canonical words are inferred from duration.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from importlib import resources
from typing import Any

import numpy as np

from munajjam.core.arabic import BASMALA_PATTERN, normalize_arabic
from munajjam.exceptions import InvalidProviderOutputError, QuranDataError
from munajjam.hybrid_aligner.canonical_zipformer import _observed_units
from munajjam.hybrid_aligner.reference import CanonicalReference, CanonicalReferencePart
from munajjam.hybrid_aligner.types import AlignmentSpan, BreathGroup, PhonemeEmission

# The exact phrase already used by upstream's example_alignment.py and tests.
# This is an introductory phrase under project conventions, never a Quran ayah.
ISTIADHAH_TEXT = "أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيمِ"
ISTIADHAH_SOURCE = (
    "https://github.com/Itqan-community/Munajjam/blob/"
    "c8012e12cb3d335728bd1ed61de38143d1b7781a/munajjam/examples/example_alignment.py#L268"
)


def full_surah_reference(surah_id: int, riwaya: str) -> CanonicalReference:
    """Read every canonical slot, including riwaya-specific numbering, or fail."""
    if isinstance(surah_id, bool) or not isinstance(surah_id, int) or not 1 <= surah_id <= 114:
        raise ValueError("surah_id must be an integer in 1..114")
    if riwaya not in ("hafs", "warsh"):
        raise ValueError("riwaya must be hafs or warsh")
    filename = f"quran_{riwaya}.json"
    payload = resources.files("munajjam.data").joinpath(filename).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    source = f"munajjam.data/{filename}"
    data = json.loads(payload)
    texts = data.get(str(surah_id))
    if (
        not isinstance(texts, list)
        or not texts
        or any(not isinstance(text, str) or not normalize_arabic(text) for text in texts)
    ):
        raise QuranDataError(f"complete canonical {riwaya} Surah {surah_id} is unavailable")
    parts = [
        CanonicalReferencePart(
            "istiadhah",
            ISTIADHAH_TEXT,
            ISTIADHAH_SOURCE,
            hashlib.sha256(ISTIADHAH_TEXT.encode()).hexdigest(),
        )
    ]
    if surah_id not in (1, 9):
        # Warsh 1:1 includes Alhamdu: take only its verified Basmalah prefix.
        basmalah = " ".join(data["1"][0].split()[:4])
        if not BASMALA_PATTERN.fullmatch(normalize_arabic(basmalah)):
            raise QuranDataError(f"no verified {riwaya} Basmalah prefix at 1:1")
        parts.append(CanonicalReferencePart("basmalah", basmalah, source + "#1:1", digest))
    parts.extend(
        CanonicalReferencePart(
            "ayah",
            text,
            f"{source}#{surah_id}:{i}",
            digest,
            surah_id,
            i,
        )
        for i, text in enumerate(texts, 1)
    )
    text = " ".join(part.text for part in parts)
    return CanonicalReference(
        riwaya, tuple(parts), source, digest, text, hashlib.sha256(text.encode()).hexdigest()
    )


@dataclass(frozen=True, slots=True)
class ReferenceWord:
    index: int
    part_index: int
    text: str
    normalized: str


def reference_words(reference: CanonicalReference) -> tuple[ReferenceWord, ...]:
    words = []
    for part_index, part in enumerate(reference.parts):
        for text in part.text.split():
            normalized = normalize_arabic(text)
            if normalized:
                words.append(ReferenceWord(len(words), part_index, text, normalized))
    return tuple(words)


@dataclass(frozen=True, slots=True)
class SurahMapping:
    words: tuple[ReferenceWord, ...]
    group_word_indices: tuple[tuple[int, ...], ...]
    diagnostics: dict[str, Any]


class _PackedBacktrace:
    """Three optimal-operation bits per cell instead of one byte per cell."""

    def __init__(self, rows: int, columns: int, max_bytes: int):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_traceback_bytes must be a positive integer")
        stride = (columns + 7) // 8
        self.nbytes = rows * 3 * stride
        if self.nbytes > max_bytes:
            raise InvalidProviderOutputError("global canonical DP traceback memory budget exceeded")
        self.data = np.zeros((rows, 3, stride), dtype=np.uint8)

    def write_row(self, row: int, masks: np.ndarray) -> None:
        for bit in range(3):
            self.data[row, bit] = np.packbits((masks >> bit) & 1)

    def __getitem__(self, index: tuple[int, int]) -> int:
        row, column = index
        shift = 7 - column % 8
        return sum(
            ((int(self.data[row, bit, column // 8]) >> shift) & 1) << bit for bit in range(3)
        )


def _map_surah_variant(
    reference: CanonicalReference,
    emissions: Sequence[Sequence[PhonemeEmission]],
    token_table: Sequence[str],
    *,
    max_cells: int = 1_000_000_000,
    max_traceback_bytes: int = 384 * 1024 * 1024,
    candidate_costs: list[int] | None = None,
    check_cancelled: Callable[[], None] = lambda: None,
) -> SurahMapping:
    """Semi-global character edit DP with free *whole-word* prefix/suffix skips.

    One path consumes all observations in breath order. Internal skipped words,
    split words across breaths, weak word support, and competing canonical
    intervals are rejected. Optional intros may be deleted, but never emitted
    without acoustic support. Partial Surahs are allowed only at word boundaries.
    Memory is O(observed * canonical) bytes for backtrace plus rolling score rows.
    A fixed cell budget prevents unbounded allocation; no greedy fallback exists.
    """
    if isinstance(max_cells, bool) or not isinstance(max_cells, int) or max_cells <= 0:
        raise ValueError("max_cells must be a positive integer")
    words = reference_words(reference)
    canonical = "".join(w.normalized for w in words)
    char_word = np.repeat(np.arange(len(words)), [len(w.normalized) for w in words])
    boundaries = np.cumsum([0] + [len(w.normalized) for w in words])
    observed = []
    origins = []
    for group_index, group_emissions in enumerate(emissions):
        units = _observed_units(group_emissions, token_table)
        observed.extend(units)
        origins.extend([group_index] * len(units))
    m, n = len(observed), len(canonical)
    if not m or not n or (m + 1) * (n + 1) > max_cells:
        raise InvalidProviderOutputError("global canonical DP cell budget exceeded or empty input")
    # Initial free starts only at word boundaries; remaining leading chars cost 1.
    previous = np.arange(n + 1, dtype=np.int32)
    starts = np.zeros(n + 1, dtype=np.int32)
    boundary_set = set(map(int, boundaries))
    for j in range(n + 1):
        if j in boundary_set:
            previous[j] = 0
            starts[j] = j
        elif j:
            previous[j] = previous[j - 1] + 1
            starts[j] = starts[j - 1]
    back = _PackedBacktrace(m + 1, n + 1, max_traceback_bytes)
    canonical_array = np.array(list(canonical))
    columns = np.arange(n, dtype=np.int32)
    for i, unit in enumerate(observed, 1):
        check_cancelled()
        diagonal = previous[:-1] + (canonical_array != unit.unit)
        insertion = previous[1:] + 1
        base = np.minimum(diagonal, insertion)
        # Resolve the left-to-right deletion recurrence with a prefix minimum.
        # This computes the exact full DP row, with no beam/greedy pruning.
        values = np.minimum.accumulate(base - columns) + columns
        values = np.minimum(values, i + columns + 1)
        current = np.empty(n + 1, dtype=np.int32)
        current[0], current[1:] = i, values
        masks = np.empty(n + 1, dtype=np.uint8)
        masks[0] = 2
        masks[1:] = (
            (diagonal == values).astype(np.uint8)
            | ((insertion == values).astype(np.uint8) << 1)
            | ((current[:-1] + 1 == values).astype(np.uint8) << 2)
        )
        back.write_row(i, masks)
        previous = current
    best_cost = min(int(previous[j]) for j in boundaries[1:])
    if candidate_costs is not None:
        candidate_costs.append(best_cost)
    ends = [int(j) for j in boundaries[1:] if previous[j] == best_cost and back[m, j] & 3]
    if len(ends) != 1:
        raise InvalidProviderOutputError("ambiguous global canonical interval")
    j, i = ends[0], m
    pairs = []
    operations = {"match": 0, "substitute": 0, "insert": 0, "delete": 0}
    inserted = []
    while i:
        mask = int(back[i, j])
        op = 0 if mask & 1 else 1 if mask & 2 else 2
        if op == 0:
            same = observed[i - 1].unit == canonical[j - 1]
            pairs.append((i - 1, j - 1, same))
            operations["match" if same else "substitute"] += 1
            i, j = i - 1, j - 1
        elif op == 1:
            operations["insert"] += 1
            inserted.append(i - 1)
            i -= 1
        else:
            operations["delete"] += 1
            j -= 1
    # Traverse every optimal backtrace, not only deterministic tie-breaking.
    # If another optimal path moves a canonical word across a physical breath,
    # the boundary is acoustically ambiguous even when endpoints agree.
    pending = [(m, ends[0])]
    visited: set[int] = set()
    possible_starts: set[int] = set()
    possible_groups: dict[int, set[int]] = {}
    while pending:
        oi, ci = pending.pop()
        if not oi:
            possible_starts.add(int(starts[ci]))
            if len(possible_starts) > 1:
                raise InvalidProviderOutputError("ambiguous global canonical interval")
            continue
        key = oi * (n + 1) + ci
        if key in visited:
            continue
        visited.add(key)
        if len(visited) > 2_000_000:
            raise InvalidProviderOutputError("ambiguous global DP backtrace budget exceeded")
        if len(visited) % 4096 == 0:
            check_cancelled()
        mask = int(back[oi, ci])
        if mask & 1 and ci:
            wi = int(char_word[ci - 1])
            possible_groups.setdefault(wi, set()).add(origins[oi - 1])
            if len(possible_groups[wi]) > 1:
                raise InvalidProviderOutputError("ambiguous word assignment across breaths")
            pending.append((oi - 1, ci - 1))
        if mask & 2:
            pending.append((oi - 1, ci))
        if mask & 4 and ci:
            pending.append((oi, ci - 1))
    pairs.reverse()
    insertion_runs = []
    for oi in sorted(inserted):
        if insertion_runs and oi == insertion_runs[-1][-1] + 1:
            insertion_runs[-1].append(oi)
        else:
            insertion_runs.append([oi])
    for run in insertion_runs:
        extra = "".join(observed[oi].unit for oi in run)
        if any(len(word.normalized) >= 2 and word.normalized in extra for word in words):
            raise InvalidProviderOutputError("repeated or unrelated full-word insertion rejected")
    cost = best_cost / max(m, ends[0] - j, 1)
    if cost > 0.25 or operations["match"] / m < 0.70:
        raise InvalidProviderOutputError(
            f"low-quality global canonical DP: normalized_cost={cost:.4f}, "
            f"observed_match_coverage={operations['match'] / m:.4f}"
        )
    word_groups: dict[int, set[int]] = {}
    support: dict[int, int] = {}
    exact_chars = set()
    group_matches = [0] * len(emissions)
    for oi, ci, same in pairs:
        wi = int(char_word[ci])
        word_groups.setdefault(wi, set()).add(origins[oi])
        support[wi] = support.get(wi, 0) + int(same)
        if same:
            exact_chars.add(ci)
            group_matches[origins[oi]] += 1
    selected = sorted(word_groups)
    for pi, part in enumerate(reference.parts):
        if part.kind != "ayah":
            part_words = {w.index for w in words if w.part_index == pi}
            if part_words.intersection(selected) and not part_words.issubset(selected):
                raise InvalidProviderOutputError("partial introductory phrase is ambiguous")
    if not selected:
        raise InvalidProviderOutputError("no canonical words supported")
    for wi in range(selected[0], selected[-1] + 1):
        part = reference.parts[words[wi].part_index]
        if wi not in word_groups:
            if part.kind != "ayah":
                continue  # optional unrecited introduction
            raise InvalidProviderOutputError(
                "missing internal canonical word; no fabricated timing"
            )
        if len(word_groups[wi]) != 1:
            raise InvalidProviderOutputError("word crosses breaths; ambiguous boundary")
        if int(boundaries[wi]) not in exact_chars or int(boundaries[wi + 1]) - 1 not in exact_chars:
            raise InvalidProviderOutputError("partial or uncertain canonical word boundary")
        if support[wi] / len(words[wi].normalized) < 0.70:
            raise InvalidProviderOutputError("low-confidence or partial canonical word")
    slices = tuple(
        tuple(wi for wi in selected if word_groups[wi] == {gi}) for gi in range(len(emissions))
    )
    if any(not indices for indices in slices):
        raise InvalidProviderOutputError("breath has no supported canonical words")
    # Group-specific quality prevents a good long breath hiding an unrelated short one.
    for gi in range(len(emissions)):
        count = origins.count(gi)
        matches = group_matches[gi]
        if matches / count < 0.70:
            raise InvalidProviderOutputError("low-confidence breath mapping")
    evidence = [
        {
            "group_index": origins[oi],
            "canonical_word_index": int(char_word[ci]),
            "canonical_character_index": ci,
            "match": same,
            "token_id": observed[oi].token_id,
            "start_frame": observed[oi].start_frame,
            "end_frame": observed[oi].end_frame,
            "score": observed[oi].score,
        }
        for oi, ci, same in pairs
    ]
    return SurahMapping(
        words,
        slices,
        {
            "algorithm": "global-monotonic-character-dp-v1",
            "normalized_cost": cost,
            "total_cost": best_cost,
            "traceback_bytes": back.nbytes,
            "observed_match_coverage": operations["match"] / m,
            "operations": operations,
            "group_word_indices": [list(indices) for indices in slices],
            "cells": evidence,
            "policy": "word-boundary partial interval; reject ambiguity/internal omissions/restarts",
        },
    )


def map_surah(
    reference: CanonicalReference,
    emissions: Sequence[Sequence[PhonemeEmission]],
    token_table: Sequence[str],
    *,
    max_cells: int = 1_000_000_000,
    max_traceback_bytes: int = 384 * 1024 * 1024,
    check_cancelled: Callable[[], None] = lambda: None,
) -> SurahMapping:
    """Compare complete optional-introduction variants using global DP.

    Each variant still contains the *entire* canonical Surah. Optional phrases
    are all-or-none branches, never zero-cost per-character deletions that could
    steal matching letters from another phrase. Equal-cost different assignments
    fail closed. Duplicate paths yielding the same original word indices agree.
    """
    original_words = reference_words(reference)
    optional_parts = [i for i, part in enumerate(reference.parts) if part.kind != "ayah"]
    candidates = []
    errors = []
    candidate_costs = []
    for mask in range(1 << len(optional_parts)):
        keep = [
            i
            for i in range(len(reference.parts))
            if i not in optional_parts or mask & (1 << optional_parts.index(i))
        ]
        parts = tuple(reference.parts[i] for i in keep)
        text = " ".join(part.text for part in parts)
        variant = CanonicalReference(
            reference.riwaya,
            parts,
            reference.source_file,
            reference.source_file_sha256,
            text,
            hashlib.sha256(text.encode()).hexdigest(),
        )
        try:
            candidate = _map_surah_variant(
                variant,
                emissions,
                token_table,
                max_cells=max_cells,
                max_traceback_bytes=max_traceback_bytes,
                candidate_costs=candidate_costs,
                check_cancelled=check_cancelled,
            )
        except InvalidProviderOutputError as exc:
            errors.append(exc)
            continue
        # Restore canonical indices/provenance into the complete reference.
        original_part_words = {
            pi: [w.index for w in original_words if w.part_index == pi] for pi in keep
        }
        remap = {}
        offsets = dict.fromkeys(keep, 0)
        for word in candidate.words:
            pi = keep[word.part_index]
            remap[word.index] = original_part_words[pi][offsets[pi]]
            offsets[pi] += 1
        slices = tuple(
            tuple(remap[wi] for wi in indices) for indices in candidate.group_word_indices
        )
        diagnostics = dict(candidate.diagnostics)
        diagnostics["group_word_indices"] = [list(indices) for indices in slices]
        diagnostics["cells"] = [
            dict(cell, canonical_word_index=remap[cell["canonical_word_index"]])
            for cell in diagnostics["cells"]
        ]
        diagnostics["canonical_character_indices_scope"] = "selected optional-introduction variant"
        candidates.append(SurahMapping(original_words, slices, diagnostics))
    if not candidates:
        raise InvalidProviderOutputError(
            "global canonical mapping rejected: " + "; ".join(dict.fromkeys(map(str, errors)))
        )
    best_cost = min(candidate_costs)
    winners = [c for c in candidates if c.diagnostics["total_cost"] == best_cost]
    if not winners:
        raise InvalidProviderOutputError(
            "optimal global canonical path rejected; no higher-cost fallback"
        )
    if len({c.group_word_indices for c in winners}) != 1:
        raise InvalidProviderOutputError("ambiguous introductory phrase/canonical mapping")
    return winners[0]


@dataclass(frozen=True, slots=True)
class WordResult:
    word: str
    start: float
    end: float
    probability: float


@dataclass(frozen=True, slots=True)
class AyahResult:
    ayah_number: int
    start_time: float
    end_time: float
    text: str
    pause_duration: float
    is_breath_boundary: bool
    words: tuple[WordResult, ...]


@dataclass(frozen=True, slots=True)
class IntroductionResult:
    kind: str
    text: str
    words: tuple[WordResult, ...]


@dataclass(frozen=True, slots=True)
class SurahAlignmentResult:
    data: tuple[AyahResult, ...]
    breath_groups: tuple[BreathGroup, ...]
    introductions: tuple[IntroductionResult, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[AyahResult]:
        return iter(self.data)

    def to_dict(self) -> dict[str, Any]:
        """Exact PR #125 public keys; separate evidence remains on the object."""
        return {
            "data": [asdict(ayah) for ayah in self.data],
            "breath_groups": [
                {"start_sec": b.start, "end_sec": b.end, "duration_sec": b.duration}
                for b in self.breath_groups
            ],
        }


def aggregate_words(
    selected: Sequence[ReferenceWord], spans: Sequence[AlignmentSpan]
) -> tuple[WordResult, ...]:
    """Recover canonical words from actual character spans, consuming delimiters.

    Word probability is the geometric mean of character posterior summaries,
    giving each character equal weight. Delimiters are excluded. Unmatched,
    missing, reordered, or unknown symbols fail rather than acquire timestamps.
    """
    result = []
    cursor = 0
    previous = 0.0
    for word in selected:
        characters = []
        consumed = ""
        while cursor < len(spans) and spans[cursor].token in ("|", " "):
            cursor += 1
        while len(consumed) < len(word.normalized) and cursor < len(spans):
            span = spans[cursor]
            symbol = normalize_arabic(span.token)
            if len(symbol) != 1 or span.start < previous or span.confidence is None:
                raise InvalidProviderOutputError("invalid character span for word aggregation")
            if not math.isfinite(span.confidence) or not 0 <= span.confidence <= 1:
                raise InvalidProviderOutputError("invalid character confidence")
            consumed += symbol
            characters.append(span)
            previous = span.end
            cursor += 1
        if consumed != word.normalized:
            raise InvalidProviderOutputError("character spans do not cover canonical word")
        probabilities = [span.confidence for span in characters]
        probability = (
            0.0
            if 0.0 in probabilities
            else math.exp(sum(math.log(p) for p in probabilities) / len(probabilities))
        )
        result.append(WordResult(word.text, characters[0].start, characters[-1].end, probability))
    if any(span.token not in ("|", " ") for span in spans[cursor:]):
        raise InvalidProviderOutputError("unexpected trailing character spans")
    return tuple(result)
