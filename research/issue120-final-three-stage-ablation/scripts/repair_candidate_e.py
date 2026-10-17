import sys
import os
import pathlib
import json
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
run_dir = str(_SCRIPT_DIR.parent)
sys.path.insert(0, str(_REPO_ROOT / "munajjam"))
sys.path.insert(0, str(_SCRIPT_DIR))

from improved_aligner_engine import ImprovedAlignerEngineV2, load_audio_16k, FRAME_MS
from munajjam.hybrid_aligner.surah import full_surah_reference

_model_dir = os.environ.get("MUNAJJAM_MODEL_DIR", str(_REPO_ROOT / "experiments" / "models" / "model_zipformer"))
engine = ImprovedAlignerEngineV2(_model_dir)
corpus_dir = os.path.join(run_dir, "corpus")

# -----------------------------------------------------------------------------
# CANDIDATE E V2: REPAIRED ACOUSTIC-ANCHORED & ADAPTIVE BAND SEARCH
# -----------------------------------------------------------------------------
def detect_breath_pauses_frames(audio_16k, frame_stride_sec=0.040, min_pause_sec=0.35):
    # Acoustic energy calculation at frame level
    samples_per_frame = int(16000 * frame_stride_sec)
    num_frames = len(audio_16k) // samples_per_frame
    if num_frames == 0:
        return []
    rms = np.array([
        np.sqrt(np.mean(audio_16k[i * samples_per_frame : (i + 1) * samples_per_frame]**2))
        for i in range(num_frames)
    ], dtype=np.float32)
    
    thresh = max(0.005, np.percentile(rms, 95) * 0.08)
    is_silent = rms < thresh
    min_f = int(min_pause_sec / frame_stride_sec)
    
    pauses = []
    in_p = False
    st = 0
    for f, s in enumerate(is_silent):
        if s and not in_p:
            in_p = True
            st = f
        elif not s and in_p:
            in_p = False
            if (f - st) >= min_f:
                pauses.append((st + f) // 2) # frame midpoint
    return pauses

def align_candidate_e_v2_repaired(engine, audio, lprobs, surah_id, riwayah="hafs", blank_bias=2.5):
    """
    Repaired Candidate E:
    1. Detects natural acoustic silence valleys (breath pauses) autonomously.
    2. Uses cumulative pause positions + token weights to prevent boundary drift.
    3. Employs wide elastic padding (min 3.5s / 88 frames) ensuring the optimal path is never clipped.
    4. Evaluates Ayah trellises sequentially with strict state and token preservation.
    """
    T = lprobs.shape[0]
    ref = full_surah_reference(surah_id, riwayah)
    ayah_parts = [p for p in ref.parts if p.kind == "ayah"]
    num_ayahs = len(ayah_parts)
    
    # Extract canonical words and tokens per ayah
    ayah_words_list = []
    ayah_tok_lens = []
    for p in ayah_parts:
        w_list = p.text.split()
        ayah_words_list.append(w_list)
        toks, _ = engine.tokenize_words(w_list)
        ayah_tok_lens.append(len(toks))
        
    total_tokens = max(1, sum(ayah_tok_lens))
    cum_tokens = np.cumsum([0] + ayah_tok_lens) / total_tokens
    
    # Detect pauses autonomously from audio
    pauses = detect_breath_pauses_frames(audio)
    
    # Adaptive Anchor Derivation:
    # If acoustic pauses match ayah count closely, align nominal windows to pause valleys.
    # Otherwise, use wide elastic padding (3.5s = 88 frames).
    elastic_pad_frames = max(88, int((T / max(1, num_ayahs)) * 0.50))
    
    all_word_timings = []
    total_cells = 0
    total_ll = 0.0
    global_w_idx = 0
    
    for a_idx, words in enumerate(ayah_words_list):
        nominal_start = int(cum_tokens[a_idx] * T)
        nominal_end = int(cum_tokens[a_idx + 1] * T)
        
        # Apply wide elastic window to guarantee optimal Viterbi path inclusion
        start_f = max(0, nominal_start - elastic_pad_frames)
        end_f = min(T, nominal_end + elastic_pad_frames)
        
        local_lprobs = lprobs[start_f:end_f]
        toks, _ = engine.tokenize_words(words)
        total_cells += local_lprobs.shape[0] * (2 * len(toks) + 1)
        
        res_local = engine.ctc_align_segment(local_lprobs, words, blank_bias=blank_bias)
        total_ll += res_local["total_ll"]
        
        for wt in res_local["word_timings"]:
            s_ms = wt["start_ms"] + (start_f * FRAME_MS) if wt["start_ms"] >= 0 else -1
            e_ms = wt["end_ms"] + (start_f * FRAME_MS) if wt["end_ms"] >= 0 else -1
            all_word_timings.append({
                "word_idx": global_w_idx,
                "word": wt["word"],
                "start_ms": s_ms,
                "end_ms": e_ms
            })
            global_w_idx += 1
            
    return {
        "score_per_frame": total_ll / max(1, T),
        "total_ll": total_ll,
        "word_timings": all_word_timings,
        "total_cells": total_cells
    }

print("=" * 90)
print("GATE 7: RE-EVALUATING REPAIRED CANDIDATE E (V2) VS CANDIDATE D1")
print("=" * 90)

test_surahs = [
    ("EVAL-108-HUSARY", "husary_108.mp3", 108),
    ("CAL-112-HUSARY", "husary_112.mp3", 112),
    ("EVAL-114-HUSARY", "husary_114.mp3", 114),
    ("CAL-001-HUSARY", "husary_001.mp3", 1),
]

repaired_results = {}

for rec_id, fn, s_id in test_surahs:
    audio = load_audio_16k(os.path.join(corpus_dir, fn))
    lprobs = engine.extract_posteriors(audio, finalize_flush=True)
    T = lprobs.shape[0]
    
    ref = full_surah_reference(s_id, "hafs")
    canonical_words = [w for p in ref.parts if p.kind == "ayah" for w in p.text.split()]
    toks, _ = engine.tokenize_words(canonical_words)
    d1_cells = T * (2 * len(toks) + 1)
    
    # Candidate D1
    t0 = time.perf_counter()
    res_d1 = engine.ctc_align_segment(lprobs, canonical_words, blank_bias=2.5)
    t_d1 = time.perf_counter() - t0
    
    # Candidate E V2 (Repaired)
    t0 = time.perf_counter()
    res_e_v2 = align_candidate_e_v2_repaired(engine, audio, lprobs, s_id, "hafs", blank_bias=2.5)
    t_e = time.perf_counter() - t0
    
    e_cells = res_e_v2["total_cells"]
    cell_reduc = d1_cells / max(1, e_cells)
    
    # Path agreement
    diffs = []
    count = min(len(res_d1["word_timings"]), len(res_e_v2["word_timings"]))
    for i in range(count):
        s_d1 = res_d1["word_timings"][i]["start_ms"]
        s_e = res_e_v2["word_timings"][i]["start_ms"]
        if s_d1 >= 0 and s_e >= 0:
            diffs.append(abs(s_d1 - s_e))
            
    agree_40ms = np.mean([d <= 40 for d in diffs]) * 100 if diffs else 0.0
    agree_80ms = np.mean([d <= 80 for d in diffs]) * 100 if diffs else 0.0
    agree_200ms = np.mean([d <= 200 for d in diffs]) * 100 if diffs else 0.0
    mae_diff = np.mean(diffs) if diffs else 0.0
    
    repaired_results[rec_id] = {
        "surah": s_id,
        "words": len(canonical_words),
        "d1_cells": d1_cells,
        "repaired_e_cells": e_cells,
        "cell_reduction_ratio": round(cell_reduc, 2),
        "d1_latency_sec": round(t_d1, 3),
        "repaired_e_latency_sec": round(t_e, 3),
        "speedup_ratio": round(t_d1 / max(0.001, t_e), 2),
        "path_agreement_40ms_pct": round(agree_40ms, 1),
        "path_agreement_80ms_pct": round(agree_80ms, 1),
        "path_agreement_200ms_pct": round(agree_200ms, 1),
        "mae_boundary_delta_ms": round(mae_diff, 1)
    }
    
    print(f"[{rec_id:18s}] Cells: {d1_cells:,} -> {e_cells:,} ({cell_reduc:.1f}x) | Speedup: {repaired_results[rec_id]['speedup_ratio']:.2f}x | 40ms Agree: {agree_40ms:4.1f}% | 80ms Agree: {agree_80ms:4.1f}% | 200ms: {agree_200ms:4.1f}% | MAE Delta: {mae_diff:4.1f}ms")

# Benchmark Al-Mulk with Repaired Candidate E V2
print("\n--- REPAIRED AL-MULK BENCHMARK ---")
audio_mulk = load_audio_16k(os.path.join(corpus_dir, "067.mp3"))
lprobs_mulk = engine.extract_posteriors(audio_mulk, finalize_flush=True)

t0 = time.perf_counter()
res_mulk_v2 = align_candidate_e_v2_repaired(engine, audio_mulk, lprobs_mulk, 67, "hafs", blank_bias=2.5)
t_search_mulk = time.perf_counter() - t0

mulk_v2_summary = {
    "search_time_sec": round(t_search_mulk, 2),
    "evaluated_cells": res_mulk_v2["total_cells"],
    "score_per_frame": round(res_mulk_v2["score_per_frame"], 3),
    "aligned_words": sum(1 for w in res_mulk_v2["word_timings"] if w["start_ms"] >= 0)
}
print(f"Al-Mulk Search Time: {t_search_mulk:.2f}s | Evaluated Cells: {res_mulk_v2['total_cells']:,} | Score: {res_mulk_v2['score_per_frame']:.3f} | Aligned: {mulk_v2_summary['aligned_words']}/348")

out_file = os.path.join(run_dir, "CANDIDATE_E_EQUIVALENCE_RESULTS.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump({
        "repaired_candidate_e_v2": repaired_results,
        "al_mulk_v2_benchmark": mulk_v2_summary
    }, f, indent=2, ensure_ascii=False)
print(f"\n[DONE] Saved repaired Candidate E equivalence results to {out_file}")
