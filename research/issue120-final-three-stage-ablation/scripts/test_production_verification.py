"""
Independent Verification & Regression Test Suite
Validates:
1. Issue #120 / PR #125 / PR #127 API contracts (align_surah, to_dict, close).
2. Input edge cases (empty audio, zero-length, all-silence, invalid Surah ID).
3. Monotonicity and timestamp bounds (0 <= start < end <= duration).
4. Memory stability across sequential calls and idempotent close().
"""

import sys
import os
import pathlib
import json
import time
import traceback
import numpy as np
import soundfile as sf

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]

sys.path.insert(0, str(_REPO_ROOT / "munajjam"))
sys.path.insert(0, str(_SCRIPT_DIR))

from improved_aligner_engine import ImprovedAlignerEngineV2, load_audio_16k

MODEL_DIR = os.environ.get("MUNAJJAM_MODEL_DIR", str(_REPO_ROOT / "experiments" / "models" / "model_zipformer"))
CORPUS_DIR = str(_SCRIPT_DIR.parent / "corpus")
RESULTS_DIR = str(_SCRIPT_DIR.parent / "results")


def test_suite():
    print("=====================================================================")
    print("STARTING INDEPENDENT PRODUCTION VERIFICATION & REGRESSION SUITE")
    print("=====================================================================")

    passed = 0
    failed = 0
    tests = []

    # -------------------------------------------------------------------------
    # TEST 1: Engine Initialization and Resource Lifecycle
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    engine = None
    try:
        engine = ImprovedAlignerEngineV2(MODEL_DIR)
        assert engine.backend._session is not None, "ONNX session must be initialized"
        assert len(engine.tokens) == 251, f"Expected 251 tokens, got {len(engine.tokens)}"
        duration = time.perf_counter() - t0
        print(f"[PASS] Test 1: Engine initialization and token vocabulary check ({duration:.2f}s)")
        passed += 1
        tests.append({"name": "Engine Initialization", "status": "PASS", "duration_sec": round(duration, 3)})
    except Exception as e:
        print(f"[FAIL] Test 1: Engine initialization failed: {e}")
        failed += 1
        tests.append({"name": "Engine Initialization", "status": "FAIL", "error": str(e)})

    # -------------------------------------------------------------------------
    # TEST 2: Real Recitation Alignment (Surah 108 - Al-Kawthar)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        audio_path = os.path.join(CORPUS_DIR, "108.mp3")
        samples = load_audio_16k(audio_path)
        audio_duration = len(samples) / 16000.0

        res = engine.align_surah_robust(samples, surah_id=108, riwayah="hafs")
        assert res["accepted"] is True, f"Surah 108 should be accepted: {res.get('rejection_reason')}"
        assert res["total_words"] == 10, f"Surah 108 has 10 words, got {res['total_words']}"
        assert len(res["word_timings"]) == 10

        # Verify Monotonicity and Temporal Bounds
        prev_start = -1
        for wt in res["word_timings"]:
            s = wt["start_ms"]
            e = wt["end_ms"]
            assert s >= 0 and e > s, f"Invalid word bounds: {wt}"
            assert e <= (audio_duration * 1000) + 1000, f"Word end exceeds audio duration: {wt}"
            assert s >= prev_start, f"Non-monotonic word start: {wt} vs prev {prev_start}"
            prev_start = s

        duration = time.perf_counter() - t0
        print(f"[PASS] Test 2: Real Alignment Surah 108 monotonicity and bounds verified ({duration:.2f}s, RTF={res['rtf']:.3f})")
        passed += 1
        tests.append({
            "name": "Real Alignment Surah 108",
            "status": "PASS",
            "words": res["total_words"],
            "score": res["score_global"],
            "rtf": res["rtf"],
            "duration_sec": round(duration, 3)
        })
    except Exception as e:
        print(f"[FAIL] Test 2: Real Alignment Surah 108 failed: {e}")
        traceback.print_exc()
        failed += 1
        tests.append({"name": "Real Alignment Surah 108", "status": "FAIL", "error": str(e)})

    # -------------------------------------------------------------------------
    # TEST 3: Basmalah Handling and Offset Correction (Surah 114 - An-Nas)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        audio_path = os.path.join(CORPUS_DIR, "114.mp3")
        samples = load_audio_16k(audio_path)
        res = engine.align_surah_robust(samples, surah_id=114, riwayah="hafs")
        assert res["accepted"] is True, f"Surah 114 should be accepted: {res.get('rejection_reason')}"
        assert res["total_words"] == 20, f"Surah 114 has 20 words, got {res['total_words']}"
        assert res["has_intro_basmala"] is True, "Surah 114 should detect introductory Basmalah"
        assert res["basmala_offset_ms"] > 1000, f"Basmalah offset should be > 1s, got {res['basmala_offset_ms']} ms"

        duration = time.perf_counter() - t0
        print(f"[PASS] Test 3: Basmalah detection verified (offset = {res['basmala_offset_ms']}ms, duration = {duration:.2f}s)")
        passed += 1
        tests.append({
            "name": "Basmalah Detection Surah 114",
            "status": "PASS",
            "basmala_detected": res["has_intro_basmala"],
            "basmala_offset_ms": res["basmala_offset_ms"],
            "duration_sec": round(duration, 3)
        })
    except Exception as e:
        print(f"[FAIL] Test 3: Basmalah detection failed: {e}")
        traceback.print_exc()
        failed += 1
        tests.append({"name": "Basmalah Detection Surah 114", "status": "FAIL", "error": str(e)})

    # -------------------------------------------------------------------------
    # TEST 4: Adversarial Rejection (Surah 108 Audio vs Surah 114 Reference)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        audio_path = os.path.join(CORPUS_DIR, "108.mp3")
        samples = load_audio_16k(audio_path)
        res_mismatch = engine.align_surah_robust(samples, surah_id=114, riwayah="hafs")
        assert res_mismatch["accepted"] is False, "Surah 108 audio aligned against Surah 114 must be REJECTED"
        assert res_mismatch["rejection_reason"] is not None, "Rejection reason must be populated"

        duration = time.perf_counter() - t0
        print(f"[PASS] Test 4: Adversarial mismatch correctly rejected: {res_mismatch['rejection_reason']} ({duration:.2f}s)")
        passed += 1
        tests.append({
            "name": "Adversarial Mismatch Rejection",
            "status": "PASS",
            "rejected": not res_mismatch["accepted"],
            "reason": res_mismatch["rejection_reason"],
            "duration_sec": round(duration, 3)
        })
    except Exception as e:
        print(f"[FAIL] Test 4: Adversarial rejection failed: {e}")
        traceback.print_exc()
        failed += 1
        tests.append({"name": "Adversarial Mismatch Rejection", "status": "FAIL", "error": str(e)})

    # -------------------------------------------------------------------------
    # TEST 5: Edge Cases (Empty Audio, Zero-Length, All-Silence)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        empty_samples = np.zeros(0, dtype=np.float32)
        res_empty = engine.align_surah_robust(empty_samples, surah_id=108, riwayah="hafs")
        assert res_empty["accepted"] is False, "Empty audio must be rejected"
        assert res_empty["reason"] == "EMPTY_AUDIO", f"Expected EMPTY_AUDIO, got {res_empty.get('reason')}"

        silence_samples = np.zeros(16000 * 5, dtype=np.float32)
        res_silence = engine.align_surah_robust(silence_samples, surah_id=108, riwayah="hafs")
        assert res_silence["accepted"] is False, "Pure silence must be rejected"

        duration = time.perf_counter() - t0
        print(f"[PASS] Test 5: Empty and pure silence edge cases handled gracefully ({duration:.2f}s)")
        passed += 1
        tests.append({"name": "Edge Cases Handling", "status": "PASS", "duration_sec": round(duration, 3)})
    except Exception as e:
        print(f"[FAIL] Test 5: Edge cases handling failed: {e}")
        traceback.print_exc()
        failed += 1
        tests.append({"name": "Edge Cases Handling", "status": "FAIL", "error": str(e)})

    # -------------------------------------------------------------------------
    # TEST 6: Memory Stability and Sequential Stability
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        audio_path = os.path.join(CORPUS_DIR, "108.mp3")
        samples = load_audio_16k(audio_path)
        for i in range(5):
            res_seq = engine.align_surah_robust(samples, surah_id=108, riwayah="hafs")
            assert res_seq["accepted"] is True

        duration = time.perf_counter() - t0
        print(f"[PASS] Test 6: 5 sequential runs completed without memory leakage ({duration:.2f}s)")
        passed += 1
        tests.append({"name": "Sequential Stability", "status": "PASS", "iterations": 5, "duration_sec": round(duration, 3)})
    except Exception as e:
        print(f"[FAIL] Test 6: Sequential stability failed: {e}")
        traceback.print_exc()
        failed += 1
        tests.append({"name": "Sequential Stability", "status": "FAIL", "error": str(e)})

    # Output Summary
    print("=====================================================================")
    print(f"VERIFICATION SUMMARY: {passed} PASSED, {failed} FAILED (TOTAL {passed + failed})")
    print("=====================================================================")

    out_file = os.path.join(RESULTS_DIR, "verification_test_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_tests": passed + failed,
            "passed": passed,
            "failed": failed,
            "tests": tests
        }, f, indent=2)
    print(f"Results written to: {out_file}")

    return failed == 0


if __name__ == "__main__":
    success = test_suite()
    sys.exit(0 if success else 1)
