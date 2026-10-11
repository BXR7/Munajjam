"""Stable Issue #120 engine import used by PR #125 (no eager ML imports)."""

from munajjam.hybrid_aligner.hybrid_pipeline import HybridQuranAligner
from munajjam.hybrid_aligner.surah import AyahResult, SurahAlignmentResult, WordResult

__all__ = ["HybridQuranAligner", "AyahResult", "SurahAlignmentResult", "WordResult"]
