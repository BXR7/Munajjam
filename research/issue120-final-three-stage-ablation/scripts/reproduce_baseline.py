import sys
import os
import json
import time
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

test_recordings = [
    {"id": "EVAL-108-HUSARY", "file": "husary_108.mp3", "surah": 108, "reciter": "Mahmoud Khalil Al-Husary", "partition": "evaluation_holdout"},
    {"id": "EVAL-114-HUSARY", "file": "husary_114.mp3", "surah": 114, "reciter": "Mahmoud Khalil Al-Husary", "partition": "evaluation_holdout"},
    {"id": "EVAL-113-HUSARY", "file": "husary_113.mp3", "surah": 113, "reciter": "Mahmoud Khalil Al-Husary", "partition": "evaluation_holdout"},
    {"id": "CAL-112-HUSARY", "file": "husary_112.mp3", "surah": 112, "reciter": "Mahmoud Khalil Al-Husary", "partition": "calibration"},
    {"id": "CAL-001-HUSARY", "file": "husary_001.mp3", "surah": 1, "reciter": "Mahmoud Khalil Al-Husary", "partition": "calibration"},
    {"id": "EVAL-067-AFASY", "file": "067.mp3", "surah": 67, "reciter": "Mishari Rashid al-Afasy", "partition": "evaluation_holdout"},
    {"id": "WARSH-112-HUSARY", "file": "warsh_husary_112.mp3", "surah": 112, "reciter": "Mahmoud Khalil Al-Husary", "partition": "warsh_holdout"},
]

baseline_results = {}

for item in test_recordings:
    rec_id = item["id"]
    s_id = item["surah"]
    riwayah = "warsh" if "WARSH" in rec_id else "hafs"
    audio_path = os.path.join(corpus_dir, item["file"])
    
    t0 = time.perf_counter()
    audio = load_audio_16k(audio_path)
    dur_sec = len(audio) / 16000.0
    t_load = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    lprobs = engine.extract_posteriors(audio, finalize_flush=True)
    t_infer = time.perf_counter() - t0
    
    ref = full_surah_reference(s_id, riwayah)
    canonical_words = [w for p in ref.parts if p.kind == "ayah" for w in p.text.split()]
    
    t0 = time.perf_counter()
    res = engine.ctc_align_segment(lprobs, canonical_words, blank_bias=2.5)
    t_search = time.perf_counter() - t0
    
    # Ground truth evaluation
    gt_ayahs = gt_db.get(rec_id, {}).get("ayah_segments", [])
    gt_words = [seg for a in gt_ayahs for seg in a.get("segments", []) if isinstance(seg, (list, tuple)) and len(seg) >= 3]
    
    count = min(len(gt_words), len(res["word_timings"]))
    start_errs = []
    end_errs = []
    word_records = []
    
    for i in range(count):
        pw = res["word_timings"][i]
        gw = gt_words[i]
        p_s = pw["start_ms"]
        p_e = pw["end_ms"]
        r_s = gw[1]
        r_e = gw[2]
        
        diff_s = abs(p_s - r_s) if p_s >= 0 else None
        diff_e = abs(p_e - r_e) if p_e >= 0 else None
        
        if diff_s is not None:
            start_errs.append(diff_s)
            end_errs.append(diff_e)
            
        word_records.append({
            "word_idx": i + 1,
            "word": pw["word"],
            "pred_start": p_s,
            "pred_end": p_e,
            "ref_start": r_s,
            "ref_end": r_e,
            "abs_start_err": diff_s,
            "abs_end_err": diff_e
        })
        
    mae_start = float(np.mean(start_errs)) if start_errs else None
    med_start = float(np.median(start_errs)) if start_errs else None
    mae_end = float(np.mean(end_errs)) if end_errs else None
    med_end = float(np.median(end_errs)) if end_errs else None
    p90_start = float(np.percentile(start_errs, 90)) if start_errs else None
    p95_start = float(np.percentile(start_errs, 95)) if start_errs else None
    
    acc50 = float(np.mean([e <= 50 for e in start_errs]) * 100) if start_errs else 0.0
    acc100 = float(np.mean([e <= 100 for e in start_errs]) * 100) if start_errs else 0.0
    acc200 = float(np.mean([e <= 200 for e in start_errs]) * 100) if start_errs else 0.0
    acc500 = float(np.mean([e <= 500 for e in start_errs]) * 100) if start_errs else 0.0
    
    baseline_results[rec_id] = {
        "surah": s_id,
        "riwayah": riwayah,
        "reciter": item["reciter"],
        "partition": item["partition"],
        "audio_duration_sec": round(dur_sec, 2),
        "total_canonical_words": len(canonical_words),
        "aligned_words": sum(1 for w in res["word_timings"] if w["start_ms"] >= 0),
        "score_per_frame": res["score_per_frame"],
        "latency": {
            "load_sec": round(t_load, 3),
            "neural_inference_sec": round(t_infer, 3),
            "ctc_search_sec": round(t_search, 3),
            "total_sec": round(t_load + t_infer + t_search, 3),
            "rtf": round((t_load + t_infer + t_search) / dur_sec, 4)
        },
        "accuracy_start_ms": {
            "mae": round(mae_start, 1) if mae_start is not None else "UNESTABLISHED",
            "median": round(med_start, 1) if med_start is not None else "UNESTABLISHED",
            "p90": round(p90_start, 1) if p90_start is not None else "UNESTABLISHED",
            "p95": round(p95_start, 1) if p95_start is not None else "UNESTABLISHED",
            "acc_50ms_pct": round(acc50, 1) if start_errs else "UNESTABLISHED",
            "acc_100ms_pct": round(acc100, 1) if start_errs else "UNESTABLISHED",
            "acc_200ms_pct": round(acc200, 1) if start_errs else "UNESTABLISHED",
            "acc_500ms_pct": round(acc500, 1) if start_errs else "UNESTABLISHED"
        },
        "accuracy_end_ms": {
            "mae": round(mae_end, 1) if mae_end is not None else "UNESTABLISHED",
            "median": round(med_end, 1) if med_end is not None else "UNESTABLISHED"
        },
        "raw_words": word_records
    }
    
    print(f"[{rec_id:18s}] S{s_id:03d} | Words: {len(canonical_words):3d} | Score: {res['score_per_frame']:6.3f} | Start MAE: {str(baseline_results[rec_id]['accuracy_start_ms']['mae']):8s} | RTF: {baseline_results[rec_id]['latency']['rtf']:.3f}x")

out_file = os.path.join(run_dir, "BASELINE_RAW_RESULTS.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(baseline_results, f, indent=2, ensure_ascii=False)
print(f"\n[DONE] Saved baseline raw results to {out_file}")
