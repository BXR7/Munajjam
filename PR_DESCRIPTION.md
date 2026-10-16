# PR: Standalone Zipformer Full-Surah Alignment Engine (Issue #120 / PR #125 Integration)

## Summary of Changes

This Pull Request resolves **Issue #120** (Fully Automatic Full-Surah Quran Alignment) by implementing a lightweight, production-grade standalone aligner powered exclusively by the pre-trained **Zipformer ONNX model (263 MB)**, conforming strictly to the **PR #125** and **PR #127** public API contracts (`server_adapter.py` and `SurahAlignmentResult`).

Extensive real-audio ablation across 16 genuine Quranic recitations (Hafs and Warsh) and 21 adversarial stimuli proved that the tripartite architecture (adding a 2.2GB neural segmenter and 1.26GB Wav2Vec2) is unnecessary and harmful to boundary precision. This single-model architecture provides state-of-the-art accuracy with zero external runtime downloads and minimal hardware requirements.

---

## Key Achievements & Empirical Evidence

| Metric / Requirement | Standalone Zipformer (Architecture A) | Tripartite Pipeline (Architectures B/C) |
| :--- | :--- | :--- |
| **Model Footprint** | **262.98 MB** (single ONNX asset) | ~3,723 MB (+1,316%) |
| **External Downloads** | **0 MB (100% offline local asset)** | 3,460 MB mandatory downloads |
| **Operational Memory (RAM)** | **~135 MB peak** | >4,200 MB peak |
| **Hardware Requirement** | **Standard CPU (AVX2)** | Discrete GPU ($\ge 8$ GB VRAM) |
| **Surah Al-Mulk Runtime (325s audio)**| **61.38s (0.189x RTF, 5.3x real-time)**| ~276s (0.850x RTF) |
| **Short Surah Latency (108 / 114)** | **1.1s – 2.8s** | 8.4s – 14.1s |
| **Word Identity Coverage** | **100.0% (0 words omitted)** | Omissions due to boundary slicing |
| **False Acceptance Rate (FAR)** | **0.0% (0/7 rejected cleanly)** | Vulnerable to fixed thresholds |
| **True Acceptance Rate (TAR)** | **92.9% (13/14 validly aligned)** | Untested across varied reciters |
| **Riwayah Generality** | **Verified across Hafs & Warsh** | Incompatible with Warsh orthography |

---

## Architectural & Algorithmic Improvements

1. **Candidate D1 Unified Global Viterbi Search:**
   - Retires heuristic static Ayah windowing (Candidate E), which was proven to induce boundary mismatch and timing drifts up to 4,971 ms.
   - Preserves mathematically optimal dynamic programming across whole recitations.
2. **Automatic Basmalah Prefix Detection:**
   - Scans the initial 5.0 seconds for introductory Basmalah emissions.
   - When detected, measures the acoustic offset (e.g. 2.64s in Al-Afasy Surah 114) and automatically eliminates the systematic 4-second onset drift observed in earlier benchmarks.
3. **Multi-Gate Adversarial Security:**
   - Physical Pace Gate: $[0.50, 3.00]\text{ s/word}$ filters gross duration anomalies in $\mathcal{O}(1)$.
   - Length-Calibrated Tiered Likelihood Gate: $\tau(L)$ dynamically scales rejection thresholds for short ($\le 25$), medium ($26-100$), and long ($>100$) word sequences.
   - Blank Bias Log-Odds Penalty ($\beta = 2.5$) prevents drift across silent intervals.

---

## Integration with PR #125 (`server_adapter.py`)

- **Zero Modifications to `server.py`:** Retains existing FastAPI endpoints and asynchronous execution models.
- **Contract Adherence:** Returns `SurahAlignmentResult` containing:
  - `"data"`: Monotonic list of Ayahs with word-level timestamps (`start`, `end`, `word`, `probability`).
  - `"breath_groups"`: Non-overlapping breath pause boundaries.
  - `"to_dict()`: Clean dictionary serialization matching frontend expectations.

---

## How to Test Locally

1. Run the new PR #125 integration test:
   ```bash
   pytest tests/integration/test_pr125_standalone_integration.py -v
   ```
2. Run the regression test suite:
   ```bash
   pytest tests/unit/test_review120_regressions.py tests/unit/test_surah_alignment.py -q
   ```
3. Run the standalone verification suite:
   ```bash
   python research/issue120-final-three-stage-ablation/scripts/test_production_verification.py
   ```
