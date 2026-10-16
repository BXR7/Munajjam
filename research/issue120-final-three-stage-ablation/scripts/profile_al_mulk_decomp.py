import sys
import os
import time
import json
import hashlib
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
run_dir = r"E:\github\Munajjam-research\research\issue120-final-three-stage-ablation"
sys.path.insert(0, r"E:\github\Munajjam-research\munajjam")
sys.path.insert(0, os.path.join(run_dir, "scripts"))

from improved_aligner_engine import ImprovedAlignerEngineV2, load_audio_16k, FRAME_MS
from munajjam.hybrid_aligner.surah import full_surah_reference

model_dir = r"E:\github\Munajjam-research\experiments\models\model_zipformer"
onnx_path = os.path.join(model_dir, "zipformer_p_arabic_v3.onnx")

# 1. SHA-256 of Zipformer Model
h = hashlib.sha256()
with open(onnx_path, "rb") as f:
    while chunk := f.read(1024 * 1024):
        h.update(chunk)
sha256_model = h.hexdigest()
size_bytes = os.path.getsize(onnx_path)

print(f"Zipformer ONNX: {size_bytes:,} bytes | SHA-256: {sha256_model}")

engine = ImprovedAlignerEngineV2(model_dir)

# 2. Timing Decomposition on Surah 67 (Al-Mulk, 326.1s)
audio_path = os.path.join(run_dir, "corpus", "067.mp3")

# Phase A: Audio Preprocessing
t0 = time.perf_counter()
audio = load_audio_16k(audio_path)
t_audio = time.perf_counter() - t0
dur_sec = len(audio) / 16000.0

# Phase B: Neural Acoustic Inference (ONNX Session Run)
t0 = time.perf_counter()
lprobs = engine.extract_posteriors(audio, finalize_flush=True)
t_neural = time.perf_counter() - t0
T = lprobs.shape[0]

# Phase C: Alignment Search Time (Candidate E bounded)
t0 = time.perf_counter()
ref = full_surah_reference(67, "hafs")
ayah_parts = [p for p in ref.parts if p.kind == "ayah"]
num_ayahs = len(ayah_parts)
frames_per_ayah = T / num_ayahs

total_cells = 0
all_words = []
total_ll = 0.0

for a_idx, p in enumerate(ayah_parts):
    words = p.text.split()
    sf = int(max(0, (a_idx - 0.25) * frames_per_ayah))
    ef = int(min(T, (a_idx + 1.25) * frames_per_ayah))
    sub_lprobs = lprobs[sf:ef]
    toks, _ = engine.tokenize_words(words)
    total_cells += sub_lprobs.shape[0] * (2 * len(toks) + 1)
    res = engine.ctc_align_segment(sub_lprobs, words, blank_bias=2.5)
    total_ll += res["total_ll"]
    all_words.extend(res["word_timings"])

t_search = time.perf_counter() - t0
t_total = t_audio + t_neural + t_search

print("\n--- TIMING DECOMPOSITION (AL-MULK 326.1s) ---")
print(f"Audio Preprocessing Time   : {t_audio:6.2f} s")
print(f"Neural Inference Time (CPU): {t_neural:6.2f} s (RTF: {t_neural/dur_sec:.3f}x)")
print(f"Candidate E Search Time    : {t_search:6.2f} s")
print(f"Total Pipeline Latency     : {t_total:6.2f} s (Overall RTF: {t_total/dur_sec:.3f}x)")
print(f"Evaluated Cells            : {total_cells:,}")
print(f"Global Score per Frame     : {total_ll/T:.3f}")

# Save baseline raw decomposition
baseline_decomp = {
    "model_metadata": {
        "model_file": "zipformer_p_arabic_v3.onnx",
        "size_bytes": size_bytes,
        "sha256": sha256_model,
        "vocab_size": 251,
        "blank_id": 0,
        "frame_stride_ms": 40.0
    },
    "al_mulk_benchmark_profile": {
        "audio_duration_sec": round(dur_sec, 2),
        "total_frames": T,
        "audio_preprocessing_sec": round(t_audio, 3),
        "neural_inference_sec": round(t_neural, 3),
        "alignment_search_sec": round(t_search, 3),
        "total_pipeline_latency_sec": round(t_total, 3),
        "neural_rtf": round(t_neural / dur_sec, 4),
        "overall_pipeline_rtf": round(t_total / dur_sec, 4),
        "evaluated_cells": total_cells,
        "score_per_frame": round(total_ll / T, 3)
    }
}

out_json = os.path.join(run_dir, "results", "baseline_al_mulk_decomp.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(baseline_decomp, f, indent=2, ensure_ascii=False)
