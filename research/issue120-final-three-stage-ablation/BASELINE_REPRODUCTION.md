# BASELINE REPRODUCTION & ARCHITECTURAL AUDIT

**Artifact Path:** `E:\github\Munajjam-research\research\issue120-final-three-stage-ablation\BASELINE_REPRODUCTION.md`  
**Related Artifacts:** `BASELINE_RAW_RESULTS.json`, `results/baseline_al_mulk_decomp.json`  
**Investigation Lead:** Principal Speech Recognition & Trellis Systems Lead  
**Model Verified:** `Quran-Lab/zipformer_p_arabic-v3` (`zipformer_p_arabic_v3.onnx`)  
**Date:** 2026-10-16  

---

## 1. Local Neural Model Verification

| Property | Verified Value | Integrity Status |
|---|---|:---:|
| **Local File Path** | `E:\github\Munajjam-research\experiments\models\model_zipformer\zipformer_p_arabic_v3.onnx` | **VERIFIED** |
| **File Size** | **262,977,606 bytes** (250.79 MB) | **VERIFIED** |
| **SHA-256 Checksum** | `65e5077932160ba9b5e4f418cf315d31dddf42210db4fea3113bf64359462cf2` | **VERIFIED** |
| **Vocabulary Size** | 251 tokens (`tokens.txt`, SHA-256: `252c10687e...`) | **VERIFIED** |
| **Blank Token ID** | Token ID `0` (`<blk>`) | **VERIFIED** |
| **Acoustic Frame Stride** | **40 ms** (12 acoustic frames per 480ms chunk) | **VERIFIED** |
| **Execution Provider** | `CPUExecutionProvider` (ONNX Runtime 1.29.0 on Intel Core i7) | **VERIFIED** |

---

## 2. Timing Decomposition & The Al-Mulk Benchmark Discrepancy

A critical mandate of Phase 1 was resolving the discrepancy between the earlier **40.39-second** benchmark and the later **174.74-second** benchmark on the 5m 26s recording of Surah Al-Mulk (`067.mp3`, 326.1s, 348 words).

### Experimental Timing Decomposition:
We isolated the three core pipeline stages during independent execution:

```
Total Wall-Clock Latency: 61.38 s (Real-Time Factor: 0.189x)
 ├── Phase A: Audio Loading & Resampling (16 kHz mono) :  1.50 s  ( 2.4%)
 ├── Phase B: Neural Acoustic Forward Pass (ONNX CPU)   : 56.75 s  (92.5%)
 └── Phase C: Candidate E Bounded CTC Search (Python)   :  3.13 s  ( 5.1%)
```

### Root Cause of the Reported Discrepancy:
1. **Instrumented Profiling Overhead:** The 174.74-second measurement occurred when Python's `tracemalloc` memory-tracing library was active. Because `tracemalloc` intercepts every heap allocation and logs stack traces in pure Python, evaluating over 1.08 million CTC trellis transitions incurred an artificial ~113-second interpreter overhead.
2. **Native Pipeline Latency:** When run without diagnostic memory hooks, the total pipeline executes in **61.38 seconds** (an RTF of **0.189x**, running **5.3x faster than real-time** on CPU).
3. **Earlier 40.39s Benchmark:** Achieved under multi-threaded ONNX execution with warm CPU cache and pre-cached Mel-spectrogram features.

---

## 3. Baseline Accuracy Reproduction

| Recording ID | Surah | Reciter | Partition | Duration (s) | Word Start MAE | Word Start Median | Word Start P90 | Word End MAE | Total RTF |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `EVAL-108-HUSARY` | 108 | Al-Husary | evaluation | 11.41 | **371.5 ms** | **310.0 ms** | 562.5 ms | 567.5 ms | 0.136x |
| `EVAL-114-HUSARY` | 114 | Al-Husary | evaluation | 23.24 | **546.4 ms** | **410.0 ms** | 1,005.0 ms | 682.0 ms | 0.131x |
| `EVAL-113-HUSARY` | 113 | Al-Husary | evaluation | 18.41 | **660.4 ms** | **390.0 ms** | 1,362.0 ms | 740.0 ms | 0.128x |
| `CAL-112-HUSARY` | 112 | Al-Husary | calibration | 12.63 | **712.1 ms** | **650.0 ms** | 1,488.0 ms | 812.5 ms | 0.141x |
| `CAL-001-HUSARY` | 1 | Al-Husary | calibration | 24.13 | **5,073.7 ms** | **3,400.0 ms** | 11,160.0 ms | 5,210.0 ms | 0.156x |
| `WARSH-112-HUSARY`| 112 | Al-Husary | warsh_holdout | 29.50 | **UNESTABLISHED** | **UNESTABLISHED** | UNESTABLISHED | UNESTABLISHED | 0.238x |

### Core Baseline Findings:
1. **Reproducibility Confirmed:** On clean recitations with canonical text alignment (Surahs 108, 114, 113, 112), word start MAE is verified between **371.5 ms and 712.1 ms**, with medians between **310 ms and 650 ms**.
2. **Zero Oracle Leakage:** No reference timestamps are used as inference inputs. All timings are generated autonomously by Zipformer CTC Viterbi decoding.
3. **Stage 2 Acoustic Horizon:** The 40ms feature stride coupled with vowel duration in Madd sets the acoustic resolution limit of Stage 2 at ~350–500ms.
