"""
Integration Test: PR #125 server_adapter with StandaloneZipformerAligner (Architecture A)

Validates that PR #125's run_hybrid_alignment successfully drives the standalone
Zipformer alignment engine without modifying server.py and without downloading external models.
"""

import asyncio
import os
import pathlib
import pytest

from munajjam.hybrid_aligner.server_adapter import run_hybrid_alignment
from munajjam.hybrid_aligner.standalone_aligner import StandaloneZipformerAligner
from munajjam.core.hybrid_aligner import HybridQuranAligner

CORPUS_DIR = pathlib.Path(r"E:\github\Munajjam-research\research\issue120-final-three-stage-ablation\corpus")
AUDIO_108 = CORPUS_DIR / "108.mp3"


@pytest.fixture(autouse=True)
def patch_engine(monkeypatch):
    """Ensure HybridQuranAligner.from_pretrained instantiates StandaloneZipformerAligner."""
    monkeypatch.setattr(HybridQuranAligner, "from_pretrained", lambda **kwargs: StandaloneZipformerAligner.from_pretrained())


def test_standalone_aligner_direct():
    """Direct test of StandaloneZipformerAligner on Surah 108."""
    engine = StandaloneZipformerAligner.from_pretrained()
    try:
        res = engine.align_surah(AUDIO_108, surah_id=108, riwaya="hafs")
        assert len(res.data) == 3, f"Surah 108 has 3 ayahs, got {len(res.data)}"
        d = res.to_dict()
        assert "data" in d
        assert "breath_groups" in d
        assert len(d["data"]) == 3
        total_words = sum(len(a["words"]) for a in d["data"])
        assert total_words == 10, f"Expected 10 words, got {total_words}"
    finally:
        engine.close()


def test_pr125_run_hybrid_alignment_async():
    """Verify asynchronous execution via PR #125's run_hybrid_alignment runner."""
    progress_updates = []

    def progress(completed, total):
        progress_updates.append((completed, total))

    params = {
        "method": "hybrid",
        "surah_id": 108,
        "riwaya": "hafs",
        "min_silence_ms": 300,
        "min_speech_ms": 100,
        "pad_ms": 100,
        "repetition_attach": False,
    }

    result = asyncio.run(
        run_hybrid_alignment(
            job_id="test-job-108",
            audio_path=str(AUDIO_108),
            params=params,
            progress_callback=progress,
        )
    )

    assert result is not None
    assert len(result.data) == 3
    assert len(progress_updates) >= 4
    assert progress_updates[-1] == (4, 4)

    # Contract validation matching server.py expectations
    res_dict = result.to_dict()
    assert "data" in res_dict
    assert "breath_groups" in res_dict

    # Check that timestamps are non-negative and monotonic
    for ayah in res_dict["data"]:
        assert ayah["start_time"] >= 0
        assert ayah["end_time"] >= ayah["start_time"]
        for w in ayah["words"]:
            assert w["start"] >= 0
            assert w["end"] >= w["start"]
            assert len(w["word"]) > 0
