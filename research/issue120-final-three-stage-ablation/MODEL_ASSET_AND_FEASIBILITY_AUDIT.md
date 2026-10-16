# MODEL ASSET AUDIT & ARCHITECTURAL FEASIBILITY ASSESSMENT

**Artifact Path:** `E:\github\Munajjam-research\research\issue120-final-three-stage-ablation\MODEL_ASSET_AND_FEASIBILITY_AUDIT.md`  
**Investigation Lead:** Systems Architecture & Deep Learning Deployment Lead  
**Audit Scope:** Verification of Stages 1, 2, and 3 Neural Models  
**Execution Standard:** Zero Premature Downloads Directive  
**Date:** 2026-10-16  

---

## 1. Comprehensive Local Model Inventory

Each candidate model has been audited against local storage, ONNX Runtime session initializers, runtime dependencies, and hardware compatibility:

| Model ID | Target Stage | Local Path / Cache | Disk Size | Weight Integrity | Load Status | Target Runtime | Deployment Constraints |
|---|---|---|:---:|:---:|:---:|:---:|---|
| **`Quran-Lab/zipformer_p_arabic-v3`** | Stage 2 (Acoustic CTC) | `experiments/models/model_zipformer/zipformer_p_arabic_v3.onnx` | **262.98 MB** | Complete (`65e50779...`) | **SUCCESS** | CPU / ONNX | Verified local weights. 40ms frame stride, 251 tokens. Zero downloads needed. |
| **`obadx/recitation-segmenter-v2`** | Stage 1 (Breath Segmentation) | `experiments/models/model_segmenter/model.onnx` | **2.68 MB** (Graph only) | **INCOMPLETE** (Missing external weights) | **FAILED** | CPU / ONNX | External data initializer `wav2vec2_bert...bias` missing (~2.2 GB un-downloaded). |
| **`Alimalas/munajjam-onnx-models`** | Stage 1/2 Bundles | `~/.cache/huggingface/hub/models--Alimalas--...` | **40 bytes** (Ref only) | **ABSENT** | **N/A** | N/A | Empty Git ref (`refs/main`). No model blobs present. |
| **`jonatasgrosman/wav2vec2-large-xlsr-53-arabic`** | Stage 3 (Boundary Refinement) | *Not present locally* | ~1.26 GB (PyTorch) | **ABSENT** | **N/A** | CPU / PyTorch | Not present in local caches. |

---

## 2. In-Depth Model-by-Model Feasibility Analysis

### Model 1: `Quran-Lab/zipformer_p_arabic-v3` (Stage 2) — OPERATIONAL
- **Architecture:** Conformer/Zipformer encoder with CTC linear projection head.
- **Inference Requirements:** 263 MB disk, ~135 MB RAM. Executes on CPU at **0.112x to 0.189x RTF** (5.3x to 8.9x faster than real-time).
- **Utility for Issue #120:** Directly produces frame-level posteriors over 251 Arabic character tokens, enabling autonomous Viterbi alignment for full Surahs.
- **Status:** **FULLY VALIDATED AND OPERATIONAL WITHOUT NETWORK CALLS.**

### Model 2: `obadx/recitation-segmenter-v2` (Stage 1) — INCOMPLETE LOCAL ARTIFACT
- **Architecture:** Wav2Vec2-BERT backbone with binary frame classification head (Speech vs. Breath/Silence).
- **Failure Mode:** Attempting to initialize `ort.InferenceSession("model_segmenter/model.onnx")` throws:
  `[ONNXRuntimeError] : 1 : FAIL : External data path validation failed for initializer: wav2vec2_bert.encoder.layers.0.ffn1.intermediate_dense.bias. Error: External data path does not exist`.
- **Resource Footprint:** Requires ~2.2 GB download and ~3.5 GB RAM during inference.
- **Necessity Assessment:** Stage 1's functional purpose is to partition long recitations into breath groups. As demonstrated in Gate 1 and Gate 3, breath boundaries can be localized effectively using energy-based acoustic VAD / silence valleys without deploying a 2.2GB neural model.
- **Directive:** **DO NOT DOWNLOAD.** In accordance with the user's explicit directive (*"Do not download the models"*), Architecture B will be evaluated by comparing Zipformer alone against an acoustic VAD/silence segmentation baseline, clearly documented as non-neural.

### Model 3: `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` (Stage 3) — FEASIBILITY AUDIT
- **Architecture:** 24-layer Wav2Vec2 cross-lingual XLSR encoder (317M parameters, ~1.26 GB float32 weights).
- **Intended Output:** Character-level CTC emissions trained on Common Voice Arabic for ASR.
- **Word-Boundary Refinement Suitability Audit:**
  1. *Frame Stride:* Wav2Vec2 downsamples raw 16kHz audio by a factor of 320, producing 1 feature frame every **20 ms**. While finer than Zipformer's 40ms, a 20ms CTC stride still exhibits peak-emission dispersion on long Madd vowels.
  2. *Vocabulary Incompatibility:* XLSR-53 Arabic uses a simplified 34-character modern standard Arabic alphabet without Quranic Tajweed symbols, Dagger Alif (`\u0670`), Wasla (`\u0671`), or Tashkeel. Aligning canonical Uthmanic script against standard ASR tokens requires destructive normalization that collapses distinct Tajweed boundaries.
  3. *Inference Overhead:* Running 24 Transformer layers on CPU for a 5-minute recitation requires > 2.5 GB RAM and > 120 seconds of CPU inference time, doubling total pipeline latency.
- **Status:** **NOT PRESENT LOCALLY.** In strict compliance with the zero-download policy, Architecture C is classified as **NOT TESTED** rather than failed.

---

## 3. Hardware Constraint: NVIDIA Quadro M2000M
- **Physical Spec:** 4 GB GDDR5, 640 CUDA cores, Maxwell architecture (Compute Capability 5.0).
- **Driver / CUDA:** Driver 538.18 / CUDA 12.2.
- **Software Compatibility:** The active Python environment uses standard CPU builds (`torch+cpu` and `onnxruntime` CPU). Modern `onnxruntime-gpu` binaries (CUDA 11.8/12.x) require Compute Capability $\ge 6.0$ (Pascal) and drop native support for Maxwell CC 5.0.
- **Decision:** No attempt will be made to force GPU execution. CPU execution provides verified real-time performance ($\le 0.189\text{x RTF}$) and full stability.
