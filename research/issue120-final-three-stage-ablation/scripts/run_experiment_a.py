import sys
import os
import json
import time
import tracemalloc
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
run_dir = r"E:\github\Munajjam-research\research\issue120-final-three-stage-ablation"
sys.path.insert(0, r"E:\github\Munajjam-research\munajjam")
sys.path.insert(0, os.path.join(run_dir, "scripts"))

from improved_aligner_engine import ImprovedAlignerEngineV2, load_audio_16k, FRAME_MS
from munajjam.hybrid_aligner.surah import full_surah_reference

engine = ImprovedAlignerEngineV2(r"E:\github\Munajjam-research\experiments\models\model_zipformer")
corpus_dir = os.path.join(run_dir, "corpus")
gt_file = os.path.join(run_dir, "manifests", "ground_truth_segments.json")

with open(gt_file, "r", encoding="utf-8") as f:
    gt_db = json.load(f)

# Candidate E implementation with token proportions
def align_candidate_e(engine, lprobs, surah_id, riwayah="hafs", blank_bias=2.5, overlap_ratio=0.25):
    T = lprobs.shape[0]
    ref = full_surah_reference(surah_id, riwayah)
    ayah_parts = [p for p in ref.parts if p.kind == "ayah"]
    num_ayahs = len(ayah_parts)
    
    ayah_token_counts = []
    ayah_words_list = []
    for p in ayah_parts:
        w_list = p.text.split()
        ayah_words_list.append(w_list)
        toks, _ = engine.tokenize_words(w_list)
        ayah_token_counts.append(len(toks))
        
    total_tokens = max(1, sum(ayah_token_counts))
    cum_token_fractions = np.cumsum([0] + ayah_token_counts) / total_tokens
    
    all_word_timings = []
    total_cells = 0
    total_ll = 0.0
    global_w_idx = 0
    
    avg_frames_per_ayah = T / max(1, num_ayahs)
    pad_frames = int(avg_frames_per_ayah * overlap_ratio)
    
    for a_idx, words in enumerate(ayah_words_list):
        nominal_start = int(cum_token_fractions[a_idx] * T)
        nominal_end = int(cum_token_fractions[a_idx + 1] * T)
        
        start_f = max(0, nominal_start - pad_frames)
        end_f = min(T, nominal_end + pad_frames)
        
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

eval_recordings = [
    {"id": "EVAL-108-HUSARY", "file": "husary_108.mp3", "surah": 108, "reciter": "Al-Husary"},
    {"id": "EVAL-114-HUSARY", "file": "husary_114.mp3", "surah": 114, "reciter": "Al-Husary"},
    {"id": "EVAL-113-HUSARY", "file": "husary_113.mp3", "surah": 113, "reciter": "Al-Husary"},
    {"id": "CAL-112-HUSARY", "file": "husary_112.mp3", "surah": 112, "reciter": "Al-Husary"},
    {"id": "CAL-001-HUSARY", "file": "husary_001.mp3", "surah": 1, "reciter": "Al-Husary"},
    {"id": "EVAL-067-AFASY", "file": "067.mp3", "surah": 67, "reciter": "Al-Afasy"},
]

results_a = {}

print("=" * 90)
print("EXPERIMENT A: ZIPFORMER ONLY BENCHMARK (CANDIDATE D1 VS CANDIDATE E)")
print("=" * 90)

for rec in eval_recordings:
    rec_id = rec["id"]
    s_id = rec["surah"]
    audio_path = os.path.join(corpus_dir, rec["file"])
    audio = load_audio_16k(audio_path)
    dur_sec = len(audio) / 16000.0
    
    ref = full_surah_reference(s_id, "hafs")
    canonical_words = [w for p in ref.parts if p.kind == "ayah" for w in p.text.split()]
    num_words = len(canonical_words)
    
    # 1. Forward Pass
    t0 = time.perf_counter()
    lprobs = engine.extract_posteriors(audio, finalize_flush=True)
    t_infer = time.perf_counter() - t0
    
    # 2. Candidate D1 (Unconstrained)
    t0 = time.perf_counter()
    res_d1 = engine.ctc_align_segment(lprobs, canonical_words, blank_bias=2.5)
    t_d1 = time.perf_counter() - t0
    
    # 3. Candidate E (Bounded)
    t0 = time.perf_counter()
    res_e = align_candidate_e(engine, lprobs, s_id, "hafs", blank_bias=2.5)
    t_e = time.perf_counter() - t0
    
    # Evaluate against GT
    gt_ayahs = gt_db.get(rec_id, {}).get("ayah_segments", [])
    gt_words = [seg for a in gt_ayahs for seg in a.get("segments", []) if isinstance(seg, (list, tuple)) and len(seg) >= 3]
    
    def calc_metrics(timings):
        count = min(len(gt_words), len(timings))
        start_errs = []
        end_errs = []
        for i in range(count):
            p_s = timings[i]["start_ms"]
            p_e = timings[i]["end_ms"]
            r_s = gt_words[i][1]
            r_e = gt_words[i][2]
            if p_s >= 0:
                start_errs.append(abs(p_s - r_s))
                end_errs.append(abs(p_e - r_e))
        return {
            "mae_start": round(float(np.mean(start_errs)), 1) if start_errs else None,
            "median_start": round(float(np.median(start_errs)), 1) if start_errs else None,
            "p90_start": round(float(np.percentile(start_errs, 90)), 1) if start_errs else None,
            "p95_start": round(float(np.percentile(start_errs, 95)), 1) if start_errs else None,
            "mae_end": round(float(np.mean(end_errs)), 1) if end_errs else None,
            "acc_50ms": round(float(np.mean([e <= 50 for e in start_errs]) * 100), 1) if start_errs else 0.0,
            "acc_100ms": round(float(np.mean([e <= 100 for e in start_errs]) * 100), 1) if start_errs else 0.0,
            "acc_200ms": round(float(np.mean([e <= 200 for e in start_errs]) * 100), 1) if start_errs else 0.0,
            "acc_500ms": round(float(np.mean([e <= 500 for e in start_errs]) * 100), 1) if start_errs else 0.0,
            "missing_words": sum(1 for w in timings if w["start_ms"] < 0)
        }
        
    m_d1 = calc_metrics(res_d1["word_timings"])
    m_e = calc_metrics(res_e["word_timings"])
    
    results_a[rec_id] = {
        "surah": s_id,
        "reciter": rec["reciter"],
        "audio_duration_sec": round(dur_sec, 2),
        "total_canonical_words": num_words,
        "neural_inference_sec": round(t_infer, 3),
        "candidate_d1": {
            "search_time_sec": round(t_d1, 3),
            "total_latency_sec": round(t_infer + t_d1, 3),
            "total_rtf": round((t_infer + t_d1) / dur_sec, 4),
            "score_per_frame": res_d1["score_per_frame"],
            **m_d1
        },
        "candidate_e": {
            "search_time_sec": round(t_e, 3),
            "total_latency_sec": round(t_infer + t_e, 3),
            "total_rtf": round((t_infer + t_e) / dur_sec, 4),
            "score_per_frame": round(res_e["score_per_frame"], 3),
            "total_cells": res_e["total_cells"],
            **m_e
        }
    }
    
    print(f"[{rec_id:18s}] Words: {num_words:3d} | D1 MAE: {str(m_d1['mae_start']):7s} (RTF: {results_a[rec_id]['candidate_d1']['total_rtf']:.3f}x) | CandE MAE: {str(m_e['mae_start']):7s} (RTF: {results_a[rec_id]['candidate_e']['total_rtf']:.3f}x)")

out_json = os.path.join(run_dir, "results", "experiment_a_results.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results_a, f, indent=2, ensure_ascii=False)
print(f"\n[DONE] Saved Experiment A results to {out_json}")
