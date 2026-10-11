"""Canonical character-level DP fusion for public Zipformer emissions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from munajjam.core.arabic import normalize_arabic
from munajjam.exceptions import InvalidProviderOutputError
from munajjam.hybrid_aligner.reference import CanonicalReference
from munajjam.hybrid_aligner.types import BreathGroup, PhonemeEmission


@dataclass(frozen=True, slots=True)
class ZipformerObservedUnit:
    emission_index: int
    token_id: int
    raw_symbol: str
    normalized_symbol: str
    start_frame: int
    end_frame: int
    score: float | None
    unit_index: int
    unit: str


@dataclass(frozen=True, slots=True)
class CanonicalUnit:
    index: int
    unit: str
    source: str
    source_sha256: str
    part_kind: str
    surah_id: int | None
    ayah_number: int | None


@dataclass(frozen=True, slots=True)
class ZipformerDPCell:
    operation: str
    observed: ZipformerObservedUnit | None
    canonical: CanonicalUnit | None
    cost: float

    @property
    def token_id(self) -> int | None:
        return self.observed.token_id if self.observed else None

    @property
    def start_frame(self) -> int | None:
        return self.observed.start_frame if self.observed else None

    @property
    def end_frame(self) -> int | None:
        return self.observed.end_frame if self.observed else None


@dataclass(frozen=True, slots=True)
class ZipformerDPAlignment:
    cells: tuple[ZipformerDPCell, ...]
    total_cost: float
    normalized_cost: float
    match_coverage: float
    observed_units: int
    canonical_units: int
    quality_threshold: float
    coverage_threshold: float

    @property
    def accepted(self) -> bool:
        return (
            self.normalized_cost <= self.quality_threshold
            and self.match_coverage >= self.coverage_threshold
        )


def _canonical_units(reference: CanonicalReference) -> tuple[CanonicalUnit, ...]:
    units: list[CanonicalUnit] = []
    for part in reference.parts:
        normalized = normalize_arabic(part.text).replace(" ", "")
        for char in normalized:
            units.append(
                CanonicalUnit(
                    index=len(units),
                    unit=char,
                    source=part.source,
                    source_sha256=part.source_sha256,
                    part_kind=part.kind,
                    surah_id=part.surah_id,
                    ayah_number=part.ayah_number,
                )
            )
    if not units:
        raise InvalidProviderOutputError("canonical reference has no alignable characters")
    return tuple(units)


def _observed_units(
    emissions: Sequence[PhonemeEmission], token_table: Sequence[str]
) -> tuple[ZipformerObservedUnit, ...]:
    units: list[ZipformerObservedUnit] = []
    for emission_index, emission in enumerate(emissions):
        if emission.token_id < 0 or emission.token_id >= len(token_table):
            raise InvalidProviderOutputError("Zipformer token ID is outside the token table")
        raw = token_table[emission.token_id]
        if raw == "<blank>":
            raise InvalidProviderOutputError("blank token cannot be fused as evidence")
        normalized = normalize_arabic(raw).replace(" ", "")
        for unit_index, unit in enumerate(normalized):
            units.append(
                ZipformerObservedUnit(
                    emission_index=emission_index,
                    token_id=emission.token_id,
                    raw_symbol=raw,
                    normalized_symbol=normalized,
                    start_frame=emission.start_frame,
                    end_frame=emission.end_frame,
                    score=float(emission.score) if emission.score is not None else None,
                    unit_index=unit_index,
                    unit=unit,
                )
            )
    if not units:
        raise InvalidProviderOutputError("Zipformer emitted no alignable characters")
    return tuple(units)


def align_zipformer_to_reference(
    emissions: Sequence[PhonemeEmission],
    token_table: Sequence[str],
    reference: CanonicalReference,
    *,
    quality_threshold: float = 0.45,
    coverage_threshold: float = 0.70,
    max_dp_cells: int = 30_000_000,
) -> ZipformerDPAlignment:
    """Align normalized public-token characters to canonical Quran characters.

    The DP has the four explicit operations required by the production contract:
    MATCH, SUBSTITUTE, INSERT, and DELETE. Token chunks are expanded only for
    comparison; every cell retains the original token ID, symbol, frames, score,
    and canonical provenance. Quality retains the explicit-reference API's
    normalized edit cost (all edits divided by max observed/canonical length)
    and canonical exact-match coverage. Gates are heuristics, not calibrated
    probabilities. Rolling rows and bounded byte backtrace come from Manus;
    missing scores remain unknown as in Codex. Automatic Surah mapping uses
    its separate semi-global DP and ambiguity policy.
    """
    if not 0 <= quality_threshold <= 1 or not 0 <= coverage_threshold <= 1:
        raise ValueError("DP quality thresholds must be between zero and one")
    if isinstance(max_dp_cells, bool) or not isinstance(max_dp_cells, int) or max_dp_cells <= 0:
        raise ValueError("max_dp_cells must be a positive integer")
    observed = _observed_units(emissions, token_table)
    canonical = _canonical_units(reference)
    rows, cols = len(observed), len(canonical)
    if (rows + 1) * (cols + 1) > max_dp_cells:
        raise InvalidProviderOutputError(
            "Zipformer canonical DP exceeds its configured cell budget"
        )
    # Store only two score rows and one byte per traceback cell. The former
    # implementation copied the entire path tuple into every DP cell, which
    # made full-Surah alignment quadratic in both time and path-memory usage.
    back = np.zeros((rows + 1, cols + 1), dtype=np.uint8)
    previous_cost = np.arange(cols + 1, dtype=np.float32)
    previous_matches = np.zeros(cols + 1, dtype=np.int32)
    back[0, 1:] = 2  # DELETE from the canonical reference
    for i in range(1, rows + 1):
        current_cost = np.empty(cols + 1, dtype=np.float32)
        current_matches = np.empty(cols + 1, dtype=np.int32)
        current_cost[0] = i
        current_matches[0] = 0
        back[i, 0] = 1  # INSERT an observed character
        obs = observed[i - 1]
        for j in range(1, cols + 1):
            can = canonical[j - 1]
            same = obs.unit == can.unit
            diagonal_cost = float(previous_cost[j - 1]) + (0.0 if same else 1.0)
            diagonal_matches = int(previous_matches[j - 1]) + int(same)
            above_cost = float(previous_cost[j]) + 1.0
            above_matches = int(previous_matches[j])
            left_cost = float(current_cost[j - 1]) + 1.0
            left_matches = int(current_matches[j - 1])
            candidates = (
                (diagonal_cost, -diagonal_matches, 0),
                (above_cost, -above_matches, 1),
                (left_cost, -left_matches, 2),
            )
            cost, neg_matches, operation = min(candidates)
            current_cost[j] = cost
            current_matches[j] = -neg_matches
            back[i, j] = operation
        previous_cost, previous_matches = current_cost, current_matches

    final_cost = float(previous_cost[cols])
    final_matches = int(previous_matches[cols])
    i, j = rows, cols
    cells_reversed: list[ZipformerDPCell] = []
    while i or j:
        operation = int(back[i, j])
        if i and j and operation == 0:
            obs, can = observed[i - 1], canonical[j - 1]
            same = obs.unit == can.unit
            cells_reversed.append(
                ZipformerDPCell("MATCH" if same else "SUBSTITUTE", obs, can, 0.0 if same else 1.0)
            )
            i -= 1
            j -= 1
        elif i and (j == 0 or operation == 1):
            cells_reversed.append(ZipformerDPCell("INSERT", observed[i - 1], None, 1.0))
            i -= 1
        elif j:
            cells_reversed.append(ZipformerDPCell("DELETE", None, canonical[j - 1], 1.0))
            j -= 1
        else:
            raise InvalidProviderOutputError("invalid global Zipformer DP traceback")
    alignment = ZipformerDPAlignment(
        cells=tuple(reversed(cells_reversed)),
        total_cost=final_cost,
        normalized_cost=final_cost / float(max(rows, cols, 1)),
        match_coverage=final_matches / float(cols),
        observed_units=rows,
        canonical_units=cols,
        quality_threshold=quality_threshold,
        coverage_threshold=coverage_threshold,
    )
    if not alignment.accepted:
        raise InvalidProviderOutputError(
            "Zipformer canonical alignment quality below threshold: "
            f"normalized_cost={alignment.normalized_cost:.4f}, "
            f"match_coverage={alignment.match_coverage:.4f}"
        )
    return alignment


def alignment_evidence(
    alignment: ZipformerDPAlignment,
    group: BreathGroup,
    *,
    frame_duration_sec: float,
) -> dict[str, object]:
    """Serialize accepted DP evidence without losing token/timing provenance."""
    cells: list[dict[str, object]] = []
    for cell in alignment.cells:
        observed, canonical = cell.observed, cell.canonical
        item: dict[str, object] = {
            "operation": cell.operation,
            "cost": cell.cost,
            "canonical_unit": canonical.unit if canonical else None,
            "canonical_index": canonical.index if canonical else None,
            "canonical_provenance": (
                {
                    "source": canonical.source,
                    "source_sha256": canonical.source_sha256,
                    "part_kind": canonical.part_kind,
                    "surah_id": canonical.surah_id,
                    "ayah_number": canonical.ayah_number,
                }
                if canonical
                else None
            ),
        }
        if observed is not None:
            item.update(
                {
                    "token_id": observed.token_id,
                    "raw_symbol": observed.raw_symbol,
                    "normalized_symbol": observed.normalized_symbol,
                    "unit": observed.unit,
                    "start_frame": observed.start_frame,
                    "end_frame": observed.end_frame,
                    "start_time": group.start + observed.start_frame * frame_duration_sec,
                    "end_time": group.start + observed.end_frame * frame_duration_sec,
                    "score": observed.score,
                }
            )
        else:
            item.update(
                {
                    "token_id": None,
                    "raw_symbol": None,
                    "normalized_symbol": None,
                    "unit": None,
                    "start_frame": None,
                    "end_frame": None,
                    "start_time": None,
                    "end_time": None,
                    "score": None,
                }
            )
        cells.append(item)
    return {
        "algorithm": "zipformer-character-dp-v1",
        "operations": ["MATCH", "SUBSTITUTE", "INSERT", "DELETE"],
        "total_cost": alignment.total_cost,
        "normalized_cost": alignment.normalized_cost,
        "match_coverage": alignment.match_coverage,
        "quality_threshold": alignment.quality_threshold,
        "coverage_threshold": alignment.coverage_threshold,
        "observed_units": alignment.observed_units,
        "canonical_units": alignment.canonical_units,
        "cells": cells,
    }
