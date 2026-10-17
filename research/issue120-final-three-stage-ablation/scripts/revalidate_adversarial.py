import sys
import os
import pathlib
import json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
run_dir = str(_SCRIPT_DIR.parent)
sys.path.insert(0, str(_REPO_ROOT / "munajjam"))
sys.path.insert(0, str(_SCRIPT_DIR))

from improved_aligner_engine import ImprovedAlignerEngineV2, load_audio_16k
from munajjam.hybrid_aligner.surah import full_surah_reference

_model_dir = os.environ.get("MUNAJJAM_MODEL_DIR", str(_REPO_ROOT / "experiments" / "models" / "model_zipformer"))
engine = ImprovedAlignerEngineV2(_model_dir)
corpus_dir = os.path.join(run_dir, "corpus")

# Multi-Gate Policy with Frozen Parameters
FROZEN_POLICY = {
    "evaluation_beta": 2.5,
    "pace_bounds": {
        "min_sec_per_word": 0.50,
        "max_sec_per_word": 3.00
    },
    "tiered_likelihood_gate": {
        "short_max_words": 25,
        "tau_short": -0.66,
        "medium_max_words": 100,
        "tau_medium": -1.00,
        "tau_long": -1.35
    }
}

def evaluate_stimulus(audio_data, dur_sec, text_surah, riwayah="hafs"):
    res = engine.align_surah_robust(
        audio_data,
        surah_id=text_surah,
        riwayah=riwayah,
        blank_bias=FROZEN_POLICY["evaluation_beta"],
    )
    accepted = res["accepted"]
    pace = res.get("pace_sec_per_word", dur_sec / max(1, res.get("total_words", 1)))
    score = res.get("score_global", -999.0)
    if accepted:
        reason = f"ACCEPTED_{res.get('tier', 'TIER')}"
    else:
        reason = res.get("rejection_reason") or "REJECTED"
    return accepted, reason, score, pace

# Build Comprehensive Stimuli Set
stimuli = [
    # Positive Controls across different reciters, speeds, and Surahs
    {"id": "POS-HAF-108-HUSARY", "file": "husary_108.mp3", "surah": 108, "type": "POSITIVE", "reciter": "Al-Husary (Tahqiq Standard)"},
    {"id": "POS-HAF-114-HUSARY", "file": "husary_114.mp3", "surah": 114, "type": "POSITIVE", "reciter": "Al-Husary (Tahqiq Standard)"},
    {"id": "POS-HAF-113-HUSARY", "file": "husary_113.mp3", "surah": 113, "type": "POSITIVE", "reciter": "Al-Husary (Hadar Fast)"},
    {"id": "POS-HAF-112-HUSARY", "file": "husary_112.mp3", "surah": 112, "type": "POSITIVE", "reciter": "Al-Husary (Tahqiq Standard)"},
    {"id": "POS-HAF-001-HUSARY", "file": "husary_001.mp3", "surah": 1, "type": "POSITIVE", "reciter": "Al-Husary (Tahqiq Standard)"},
    {"id": "POS-HAF-112-MINSHAWI", "file": "minshawi_112.mp3", "surah": 112, "type": "POSITIVE", "reciter": "Al-Minshawi (Unseen Reciter 1)"},
    {"id": "POS-HAF-001-MINSHAWI", "file": "minshawi_001.mp3", "surah": 1, "type": "POSITIVE", "reciter": "Al-Minshawi (Unseen Reciter 1)"},
    {"id": "POS-HAF-114-AFASY", "file": "114.mp3", "surah": 114, "type": "POSITIVE", "reciter": "Al-Afasy (Murattal Slow)"},
    {"id": "POS-HAF-108-AFASY", "file": "108.mp3", "surah": 108, "type": "POSITIVE", "reciter": "Al-Afasy (Murattal Slow)"},
    {"id": "POS-HAF-112-AFASY", "file": "112.mp3", "surah": 112, "type": "POSITIVE", "reciter": "Al-Afasy (Murattal Medium)"},
    {"id": "POS-HAF-113-AFASY", "file": "113.mp3", "surah": 113, "type": "POSITIVE", "reciter": "Al-Afasy (Murattal Medium)"},
    {"id": "POS-HAF-001-AFASY", "file": "001.mp3", "surah": 1, "type": "POSITIVE", "reciter": "Al-Afasy (Murattal Slow)"},
    {"id": "POS-HAF-067-AFASY", "file": "067.mp3", "surah": 67, "type": "POSITIVE", "reciter": "Al-Afasy (Long Surah)"},
    {"id": "POS-WAR-112-HUSARY", "file": "warsh_husary_112.mp3", "surah": 112, "type": "POSITIVE", "reciter": "Al-Husary (Warsh)"},
    
    # Negative Controls: Similar Opening Attack
    {"id": "NEG-SIM-113-vs-114", "file": "113.mp3", "surah": 114, "type": "NEGATIVE", "reciter": "Similar Opening 113 vs 114"},
    
    # Negative Controls: Gross Mismatch
    {"id": "NEG-MIS-112-vs-001", "file": "112.mp3", "surah": 1, "type": "NEGATIVE", "reciter": "Gross Mismatch 112 vs 1"},
    {"id": "NEG-MIS-001-vs-112", "file": "001.mp3", "surah": 112, "type": "NEGATIVE", "reciter": "Length Inflation 1 vs 112"},
    {"id": "NEG-MIS-114-vs-108", "file": "114.mp3", "surah": 108, "type": "NEGATIVE", "reciter": "Length Inflation 114 vs 108"},
    {"id": "NEG-MIS-108-vs-114", "file": "108.mp3", "surah": 114, "type": "NEGATIVE", "reciter": "Length Deflation 108 vs 114"},
]

results = []

print("=" * 90)
print("PHASE 8: EVALUATING ADVERSARIAL REVALIDATION & ROBUSTNESS MATRIX")
print("=" * 90)

for s in stimuli:
    audio = load_audio_16k(os.path.join(corpus_dir, s["file"]))
    dur_sec = len(audio) / 16000.0
    riwayah = "warsh" if "WAR" in s["id"] else "hafs"
    
    accepted, reason, score, pace = evaluate_stimulus(audio, dur_sec, s["surah"], riwayah)
    observed = "ACCEPT" if accepted else "REJECT"
    expected = "ACCEPT" if s["type"] == "POSITIVE" else "REJECT"
    passed = (observed == expected)
    
    results.append({
        "stimulus_id": s["id"],
        "condition": s["type"],
        "description": s["reciter"],
        "audio_duration_sec": round(dur_sec, 2),
        "text_surah": s["surah"],
        "pace_sec_per_word": round(pace, 2),
        "observed_score": round(score, 3),
        "decision_reason": reason,
        "observed_decision": observed,
        "expected_decision": expected,
        "test_passed": passed
    })
    
    print(f"[{s['id']:22s}] {observed:6s} (Exp: {expected:6s}) | Score: {score:6.3f} | Pace: {pace:4.2f}s/w | Reason: {reason:30s} => {'PASS' if passed else 'FAIL'}")

# Silence and Noise Controls
silence_5s = np.zeros(16000 * 5, dtype=np.float32)
acc_sil, r_sil, sc_sil, p_sil = evaluate_stimulus(silence_5s, 5.0, 112, "hafs")
results.append({
    "stimulus_id": "NEG-NON-SILENCE-5S",
    "condition": "NEGATIVE",
    "description": "Pure Silence 5.0s",
    "audio_duration_sec": 5.0,
    "text_surah": 112,
    "pace_sec_per_word": round(p_sil, 2),
    "observed_score": round(sc_sil, 3),
    "decision_reason": r_sil,
    "observed_decision": "ACCEPT" if acc_sil else "REJECT",
    "expected_decision": "REJECT",
    "test_passed": (not acc_sil)
})
print(f"[{'NEG-NON-SILENCE-5S':22s}] {'ACCEPT' if acc_sil else 'REJECT':6s} (Exp: REJECT) | Score: {sc_sil:6.3f} | Pace: {p_sil:4.2f}s/w | Reason: {r_sil:30s} => {'PASS' if (not acc_sil) else 'FAIL'}")

np.random.seed(42)
noise_5s = np.random.normal(0, 0.05, 16000 * 5).astype(np.float32)
acc_nse, r_nse, sc_nse, p_nse = evaluate_stimulus(noise_5s, 5.0, 112, "hafs")
results.append({
    "stimulus_id": "NEG-NON-NOISE-5S",
    "condition": "NEGATIVE",
    "description": "Gaussian Noise 5.0s",
    "audio_duration_sec": 5.0,
    "text_surah": 112,
    "pace_sec_per_word": round(p_nse, 2),
    "observed_score": round(sc_nse, 3),
    "decision_reason": r_nse,
    "observed_decision": "ACCEPT" if acc_nse else "REJECT",
    "expected_decision": "REJECT",
    "test_passed": (not acc_nse)
})
print(f"[{'NEG-NON-NOISE-5S':22s}] {'ACCEPT' if acc_nse else 'REJECT':6s} (Exp: REJECT) | Score: {sc_nse:6.3f} | Pace: {p_nse:4.2f}s/w | Reason: {r_nse:30s} => {'PASS' if (not acc_nse) else 'FAIL'}")

# Aggregations & Confidence Intervals
pos_cases = [r for r in results if r["condition"] == "POSITIVE"]
neg_cases = [r for r in results if r["condition"] == "NEGATIVE"]

N_pos = len(pos_cases)
N_neg = len(neg_cases)
TP = sum(1 for r in pos_cases if r["observed_decision"] == "ACCEPT")
FN = N_pos - TP
TN = sum(1 for r in neg_cases if r["observed_decision"] == "REJECT")
FP = N_neg - TN

TAR = (TP / N_pos) * 100.0
FRR = (FN / N_pos) * 100.0
FAR = (FP / N_neg) * 100.0
TR_rate = (TN / N_neg) * 100.0

# Wilson Score 95% Confidence Intervals
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + (z**2 / n)
    center = (p + (z**2 / (2 * n))) / denom
    spread = z * np.sqrt((p * (1 - p) + (z**2 / (4 * n))) / n) / denom
    return (round(max(0.0, center - spread) * 100, 1), round(min(1.0, center + spread) * 100, 1))

ci_tar = wilson_ci(TP, N_pos)
ci_far = wilson_ci(FP, N_neg)

summary = {
    "total_stimuli": len(results),
    "positive_controls": {
        "count": N_pos,
        "true_positives": TP,
        "false_negatives": FN,
        "true_acceptance_rate_pct": round(TAR, 1),
        "false_rejection_rate_pct": round(FRR, 1),
        "tar_95ci_pct": ci_tar
    },
    "negative_controls": {
        "count": N_neg,
        "true_negatives": TN,
        "false_positives": FP,
        "false_acceptance_rate_pct": round(FAR, 1),
        "robust_rejection_rate_pct": round(TR_rate, 1),
        "far_95ci_pct": ci_far
    }
}

print("\n" + "=" * 90)
print(f"ADVERSARIAL VERDICT: TAR = {TAR:.1f}% (95% CI: {ci_tar[0]}-{ci_tar[1]}%) | FAR = {FAR:.1f}% (95% CI: {ci_far[0]}-{ci_far[1]}%)")
print(f"Tested: {N_pos} Positives (Al-Husary, Al-Afasy, Al-Minshawi, Warsh) | {N_neg} Negatives (113vs114, Mismatch, Non-speech)")
print("=" * 90)

out_file = os.path.join(run_dir, "results", "adversarial_revalidation_results.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "cases": results}, f, indent=2, ensure_ascii=False)
print(f"Saved adversarial results to {out_file}")
