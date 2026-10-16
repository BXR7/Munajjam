# FROZEN EVALUATION PROTOCOL & GROUND TRUTH PROVENANCE

**Artifact Path:** `E:\github\Munajjam-research\research\issue120-final-three-stage-ablation\FROZEN_EVALUATION_PROTOCOL.md`  
**Related Artifact:** `CORPUS_AND_GROUND_TRUTH_MANIFEST.json`  
**Investigation Lead:** Independent Verification & Scientific Audit Lead  
**Corpus Version:** `2.0-ablation-frozen` (16 Real Audio Recordings)  
**Date:** 2026-10-16  

---

## 1. Zero-Input Evaluation Protocol & Leakage Controls

To ensure scientific validity and comply with Master Mission guidelines:
1. **Zero-Input Evaluation Standard:** External reference timestamps (e.g. from Quran.com) are **strictly isolated** and used solely for post-hoc scoring. They are **never** passed into `ImprovedAlignerEngineV2`, CTC Viterbi decoding, or search-window boundaries.
2. **Strict Partitioning Discipline:**
   - **Development Partition (`DEV`):** 2 recordings (`DEV-112-AFASY`, `DEV-001-AFASY`). Used solely for exploratory code debugging and inspecting edge cases.
   - **Calibration Partition (`CAL`):** 3 recordings (`CAL-112-HUSARY`, `CAL-001-HUSARY`, `CAL-108-AFASY`). Used to calibrate blank de-weighting ($\beta = 2.5$) and pace bounds ($[0.50, 3.00]\text{ s/word}$).
   - **Untouched Evaluation Holdout (`EVAL`):** 8 recordings spanning short, medium, and long Surahs (including 5m 26s Al-Mulk) across Al-Husary, Al-Afasy, and Al-Minshawi. Hyperparameters and thresholds were frozen prior to running this partition.
   - **Warsh Partition (`WARSH`):** 3 recordings (`WARSH-112-HUSARY`, `WARSH-112-YASSEN`, `WARSH-001-YASSEN`). Evaluated strictly independently from Hafs to prevent phonetic cross-contamination.

---

## 2. Ground Truth Provenance & Independence Audit

| Recitation Partition | Total Files | Reference Source | Independent Provenance Status | Metric Interpretation Standard |
|---|:---:|---|---|---|
| **Hafs (`DEV`, `CAL`, `EVAL`)** | 13 | Quran.com Reviewed Word Segments REST API | **PROXY INDEPENDENT** | Evaluated against community-reviewed aligner boundaries. Treat boundaries as proxy ground truth. |
| **Warsh (`WARSH`)** | 3 | None available from public human-reviewed APIs | **UNESTABLISHED** | Correctness metrics strictly labeled **UNESTABLISHED** in accordance with scientific integrity. |

---

## 3. Audited Metric Suite

Every evaluation run computes:
- **Word Identity & Occurrence Integrity:** Verifying that 100% of canonical words are aligned without duplicates or silent omissions.
- **Boundary Precision:** Mean Absolute Error (MAE), Median Error, P90, and P95 for both **Word Start** and **Word End**.
- **Tolerance Brackets:** Percentage of word boundaries localizing within $\le 50\text{ ms}$, $\le 100\text{ ms}$, $\le 200\text{ ms}$, and $\le 500\text{ ms}$.
- **Computational Overhead:** Latency decomposition (audio prep, forward pass, Viterbi search), Real-Time Factor (RTF), and Peak Resident Memory (RSS).
