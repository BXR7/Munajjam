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

# Non-Neural Acoustic VAD / Energy-based Breath Segmentation
def detect_breath_pauses_vad(audio_16k, frame_ms=20, energy_thresh_ratio=0.08, min_pause_ms=400):
    samples_per_frame = int(16000 * (frame_ms / 1000.0))
    num_frames = len(audio_16k) // samples_per_frame
    
    # Compute RMS energy per frame
    rms = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        chunk = audio_16k[i * samples_per_frame : (i + 1) * samples_per_frame]
        rms[i] = np.sqrt(np.mean(chunk**2))
        
    p95_energy = np.percentile(rms, 95)
    silence_thresh = max(0.005, p95_energy * energy_thresh_ratio)
    is_silent = rms < silence_thresh
    
    # Find contiguous silence blocks >= min_pause_ms
    min_silent_frames = int(min_pause_ms / frame_ms)
    pauses = []
    in_pause = False
    start_f = 0
    
    for f_idx, silent in enumerate(is_silent):
        if silent and not in_pause:
            in_pause = True
            start_f = f_idx
        elif not silent and in_pause:
            in_pause = False
            if (f_idx - start_f) >= min_silent_frames:
                # Store pause midpoint in seconds
                pause_mid_sec = ((start_f + f_idx) / 2.0) * (frame_ms / 1000.0)
                pauses.append({
                    "start_sec": start_f * (frame_ms / 1000.0),
                    "end_sec": f_idx * (frame_ms / 1000.0),
                    "mid_sec": pause_mid_sec
                })
                
    return pauses

print("=" * 90)
print("EXPERIMENT B: ACOUSTIC VAD BREATH SEGMENTATION VS ARCHITECTURE A")
print("(Neural Recitation Segmenter v2 status: UNAVAILABLE on disk - missing 2.2GB external weights)")
print("=" * 90)

vad_benchmark_recordings = [
    {"id": "EVAL-114-HUSARY", "file": "husary_114.mp3", "surah": 114, "num_ayahs": 6},
    {"id": "CAL-001-HUSARY", "file": "husary_001.mp3", "surah": 1, "num_ayahs": 7},
    {"id": "EVAL-067-AFASY", "file": "067.mp3", "surah": 67, "num_ayahs": 30},
]

vad_results = {}

for item in vad_benchmark_recordings:
    rec_id = item["id"]
    s_id = item["surah"]
    audio = load_audio_16k(os.path.join(corpus_dir, item["file"]))
    dur_sec = len(audio) / 16000.0
    
    pauses = detect_breath_pauses_vad(audio)
    print(f"[{rec_id:18s}] Duration: {dur_sec:6.1f}s | Detected {len(pauses)} acoustic breath pauses")
    for idx, p in enumerate(pauses[:5]):
        print(f"   Pause {idx+1}: [{p['start_sec']:.2f}s - {p['end_sec']:.2f}s] (dur: {p['end_sec']-p['start_sec']:.2f}s)")
        
    vad_results[rec_id] = {
        "audio_duration_sec": round(dur_sec, 2),
        "total_detected_pauses": len(pauses),
        "target_ayah_count": item["num_ayahs"],
        "pause_distribution_ratio": round(len(pauses) / item["num_ayahs"], 2),
        "pauses_summary": pauses[:10]
    }

out_json = os.path.join(run_dir, "results", "vad_breath_segmentation_results.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(vad_results, f, indent=2, ensure_ascii=False)
print(f"\n[DONE] Saved VAD segmentation results to {out_json}")
