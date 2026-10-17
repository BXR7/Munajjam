"""
Integration Test: PR #125 server_adapter with StandaloneZipformerAligner (Architecture A)

Validates that PR #125's run_hybrid_alignment successfully drives the standalone
Zipformer alignment engine without modifying server.py and without downloading external models.

Requires environment variables:
    MUNAJJAM_CORPUS_DIR – path to the evaluation corpus directory
    MUNAJJAM_AUDIO_108 – (optional) override for the 108.mp3 test audio file
"""

import asyncio
import os
import pathlib
import pytest

from munajjam.hybrid_aligner.server_adapter import run_hybrid_alignment
from munajjam.hybrid_aligner.standalone_aligner import StandaloneZipformerAligner
from munajjam.core.hybrid_aligner import HybridQuranAligner

CORPUS_DIR = pathlib.Path(os.environ.get("MUNAJJAM_CORPUS_DIR", ""))
AUDIO_108 = pathlib.Path(
    os.environ.get("MUNAJJAM_AUDIO_108", str(CORPUS_DIR / "108.mp3") if str(CORPUS_DIR) else "")
)
MODEL_DIR = pathlib.Path(os.environ.get("MUNAJJAM_MODEL_DIR", ""))

_RUN_REAL_MODEL = os.environ.get("RUN_REAL_MODEL") == "1"
_HAS_REAL_PATHS = CORPUS_DIR.is_dir() and AUDIO_108.is_file()

_SKIP_REASON = (
    "Real-model integration tests skipped. To enable, set RUN_REAL_MODEL=1 "
    "and provide MUNAJJAM_CORPUS_DIR with valid audio."
)

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not (_RUN_REAL_MODEL and _HAS_REAL_PATHS), reason=_SKIP_REASON),
]


@pytest.fixture(autouse=True)
def patch_engine(monkeypatch):
    """Ensure HybridQuranAligner.from_pretrained instantiates StandaloneZipformerAligner."""
    model_dir = str(MODEL_DIR) if MODEL_DIR.is_dir() else None
    monkeypatch.setattr(
        HybridQuranAligner,
        "from_pretrained",
        lambda **kwargs: StandaloneZipformerAligner.from_pretrained(model_dir=model_dir),
    )


def test_standalone_aligner_direct():
    """Direct test of StandaloneZipformerAligner on Surah 108."""
    model_dir = str(MODEL_DIR) if MODEL_DIR.is_dir() else None
    engine = StandaloneZipformerAligner.from_pretrained(model_dir=model_dir)
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
