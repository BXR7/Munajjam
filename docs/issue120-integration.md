# Issue #120 engine integration

## Baselines

- PR base: upstream `main`, `c8012e12cb3d335728bd1ed61de38143d1b7781a`
  (fetched 2026-10-11; the maintainer's `c8012e1` is still current).
- Reviewed Manus A: `3d08e9f3e8067df80fe7d989671a10d24363361e`.
- Reviewed Codex B: `fa194ad023fc7580b91c3494689224113db40864`.
- Implementation source: `BXR7/HybridQuranAligner-`,
  `cdcb83188d232feaf3931155733c509e7e20eaa1`.
- Consumer: PR #125, `cab55a7c869d7afb16ce8731523f3467a60b1acc`.
- Existing local source branch remains at
  `92a875a003f00dfef94ffccb70183fbad9b66307`, unchanged.

The integration branch starts at upstream main and transplants only the engine,
its regression tests, Quran slot-preservation fix and dependency additions. The
full-Surah mapper, aggregation, core import and server adapter are new here.
Upstream's downloaders, server, requirements and unrelated files are preserved;
source investigation reports and the Kaggle notebook are not included.

## Public API and policy

```python
from munajjam.core.hybrid_aligner import HybridQuranAligner

engine = HybridQuranAligner.from_pretrained()
try:
    result = engine.align_surah("recitation.wav", 114, riwaya="hafs")
    response = result.to_dict()  # JSON-compatible data and breath_groups
finally:
    engine.close()
```

Install `munajjam[zipformer,segmenter,wav2vec2]` for the model runtimes. Imports,
construction and CPU-only shutdown remain safe without them. Model revisions
are pinned, and production Zipformer resolution checks token and model SHA-256.
`align()` retains the earlier explicit-reference API for low-level callers.

The real segmenter receives the millisecond cleanup controls. All Zipformer
observations are mapped together against the complete bundled Surah using exact
semi-global character edit DP; only whole-word prefix/suffix omissions are free.
Optional introductory phrases are complete alternative reference branches.
Accepted words remain globally ordered across breaths. Each inferred word slice
is encoded by Wav2Vec2's independent tokenizer and aligned inside its own breath,
with at most 0.20 seconds of post-roll, clipped at the next breath/audio end.
The 251-token and 51-token vocabularies are never translated into one another.

Policy is deliberately conservative: normalized edit cost <= 0.25, observed,
per-word and per-breath exact-match coverage >= 0.70, and exact word-edge support.
Competing intervals, competing optimal breath assignments, internal missing
words, word splits across breaths, full-word insertions/restarts and partial
introductory phrases raise errors. DP allocation is capped at one billion cells;
traceback storage is bit-packed and independently capped at 384 MiB;
ambiguous backtrace exploration is capped at two million cells. No greedy fallback
exists. These are heuristic acceptance gates, not calibrated probabilities.
A successful partial Surah returns only supported whole words and records
omitted canonical indices and full source text/provenance in metadata. Failures
publish no partial result, consistent with PR #125's job error handling.

Bundled Hafs/Warsh numbering is authoritative for this project. Warsh Al-Fatiha
ayah 1 includes both Basmalah and Alhamdu. Basmalah outside Al-Fatiha is an
introduction (except Surah 9, where no introductory Basmalah is offered).
Isti'adhah follows upstream's sourced example convention and never becomes an
ayah. Incomplete references fail closed: upstream Warsh 2:254 is empty, so a
complete Warsh Surah 2 request is rejected without borrowing Hafs text.

Ayah `text` and each `word` preserve supported canonical spelling. Timings are
seconds. Word `probability` is an **uncalibrated** geometric mean of character
confidence summaries; each character uses the mean of its aligned frame
posteriors, not accumulated CTC path scores. Delimiters are excluded. A final
supported ayah word marks a breath boundary when the next word is in another
physical group or there is no next word. `pause_duration` measures to the next
supported word (or audio end), including silence after a terminal word. An ayah
may span multiple breaths; multiple ayahs may share a breath.

Callbacks receive `(completed_breaths, total_breaths)` from zero through completion;
callback errors propagate. `cancel_event` is checked between model calls and DP
rows. Full-Surah calls on an instance are serialized. `close()` is terminal and idempotent,
releases model/factory references, collects garbage and clears an already-loaded
Torch CUDA cache when available. It does not import Torch to perform cleanup.

## PR #125 connection

The verified server still contains a runner that always raises
`NotImplementedError` and an extractor that always returns `[]`. An engine alone
cannot make those stubs execute or expose breath groups. This PR leaves
`server.py` unchanged and supplies the two engine-owned replacements:

```python
from munajjam.hybrid_aligner.server_adapter import (
    run_hybrid_alignment as _run_hybrid_alignment,
    extract_breath_boundaries as _extract_breath_boundaries,
)
```

PR #125's integration owner should replace its two stub definitions with these
imports. The async runner preserves the exact `(job_id, audio_path, params,
progress_callback)` signature and closes models on success, failure or cooperative
cancellation. Native inference cannot be interrupted mid-call. `chunk_duration`
is a server placeholder, not a physical breath control. PR #125 defines no
repetition attachment semantics: `repetition_attach=True` explicitly errors;
False rejects unsafe restarts rather than silently attaching them.

The result is iterable over ayahs. Existing server formatters consume it without
renaming fields. Public breath JSON keys are the **actual** PR #125 keys
`start_sec`, `end_sec`, `duration_sec`; engine breath objects expose `start`, `end`,
`duration`. Introductory results and detailed provenance stay on the result object
rather than being mislabeled as canonical ayahs in the public response.

## Requirements to tests

| Maintainer requirement | Implementation | Verification |
| --- | --- | --- |
| Clean upstream PR | Upstream worktree; scoped transplant; no source rewrite | Base ancestry, source SHA preservation, staged paths, `git diff --check` |
| Automatic full-Surah mapping | `hybrid_pipeline.py`, `surah.py`, real segmenter/Zipformer/Wav2Vec2 adapters | `test_surah_alignment.py`: zero/one/multiple breaths, unequal ayah boundaries, partial/missing text, restarts, ambiguity, monotonicity, invalid IDs, Hafs/Warsh, optional intros, no invented reference, DP budget/quality rejection |
| Word/ayah aggregation | `surah.py`; CTC confidence/state fix in `wav2vec2_aligner.py` | Canonical word spelling, geometric confidence, missing/reordered spans, repeated CTC targets, timestamps, pauses and breath semantics |
| Server API and cleanup | `core/hybrid_aligner.py`, `server_adapter.py`, `HybridQuranAligner.align_surah/close` | Actual PR #125 formatters and success/error job lifecycle, JSON serialization, callback/cancellation, repeated close, CUDA cleanup and blocked-ML-import subprocess |
| Existing Issue #120 behavior | Transplanted adapter/reference/DP regression suites | `test_hybrid_aligner.py`, `test_hybrid_reference.py`, `test_canonical_zipformer.py`, `test_sherpa_zipformer_backend.py`, `test_zipformer_backend.py` |
| Real public runtime | Pinned Sherpa Zipformer | `test_zipformer_real_model.py`, actual Alafasy audio; no synthetic model emissions |

## Verification and limits (2026-10-11)

Full suite including real public Zipformer and the exact PR #125 consumer:
**532 passed, 5 skipped**. Four skips require upstream's missing WhisperX audio
fixture; one requires `kaldi_native_fbank` for the alternate feature backend.
Configured Ruff and format checks pass for all 23 changed Python files; diff checks
pass. Whole-repository Ruff reports 33 pre-existing diagnostics in 22 unchanged
upstream files; whole-repository formatting reports 23 pre-existing files. These
unrelated baseline files are preserved. Tests run on Python 3.14.2; asynchronous tests require execution outside
this environment's sandbox because its local socket wakeup is blocked.

Real Zipformer: Sherpa 1.13.8, public pinned ONNX, Alafasy 1:1 audio downloaded
from `https://everyayah.com/data/Alafasy_128kbps/001001.mp3`. Token SHA-256 is
`252c10687e442aa9291973065fae19fa39bcd681c4f5612ec496a647e20b43a1`; ONNX SHA-256 is
`65e5077932160ba9b5e4f418cf315d31dddf42210db4fea3113bf64359462cf2`.
The CTC greedy runtime returns no token score array; missing evidence remains
`None`, never fabricated certainty. The real inference integration passes.

The short Basmalah sample is **rejected as ambiguous by the new automatic
full-Surah mapper**. This is not evidence of successful full-Surah alignment.
Complete tripartite GPU E2E remains unverified: this environment has CPU-only
Torch and no recitation-segmenter package. Deterministic orchestration tests use
injected adapters and are not real-model evidence. Production threshold accuracy,
full tripartite timing and GPU memory behavior require further real-recitation
validation. Maintainer decisions remain on repetition attachment semantics,
replacement of PR #125's two stubs and incomplete Warsh reference data.

## Comparative review

Both candidates share the original engine ancestry. Components were selected by
semantic behavior and reproducible evidence, not the number of tests.

| Feature | Manus A | Codex B | Evidence | Selection and reason |
| --- | --- | --- | --- | --- |
| Global canonical mapping | Ordered DP followed by dense-anchor/midpoint slicing; deletion-excluding quality | Semi-global whole-word mapping with competing-optimum checks | Removing برب from Surah 114 still returns بِرَبِّ in A; final regression rejects it | B: no unsupported intervening words |
| Short real Basmalah | Accepted scattered matches across Al-Fatiha, coverage 0.122302 | Rejected ambiguous optimum | Actual ONNX output normalized to بسمللااهررحماانررح; minimum edit cost 9, equal endpoints after الرحمن and الرحيم | B: valid fail-closed result; thresholds unchanged |
| Optional introductions | Caller-provided Istiadhah; prefix-only Basmalah stripping | Complete optional reference branches, independently aligned intro spans | Hafs/Warsh intro and canonical Fatiha tests | B: automatic and preserves canonical distinction |
| Riwaya provenance | Hafs ayah count used for Warsh | Riwaya-specific bundled slots; missing slots rejected | Warsh Fatiha convention and empty Warsh 2:254 tests | B: no invented reference or borrowed Hafs numbering |
| Low-level explicit-reference DP | Rolling score rows and bounded byte traceback | Path tuple copied per cell | Memory-budget and deletion-cost regressions | A storage algorithm, B original cost semantics and nullable scores |
| Full-Surah DP storage | Smaller low-level budget; permissive slice mapper | Dense byte per cell up to one billion cells | Packed traceback and byte-budget regressions | B algorithm with new three-bit traceback and independent byte limit |
| Words/ayahs and CTC confidence | Arithmetic character aggregation; weaker slice checks | Geometric character aggregation; span correspondence checks | Repeated CTC target, canonical spelling, monotonic timing, missing span tests | B validation; neither aggregation is calibrated |
| Unknown Sherpa scores | Missing score array becomes certainty | Missing score array remains None | Real model returns empty ys_probs; nullable-score tests | B: preserve unknown evidence |
| Lifecycle and cancellation | Drops component references; early close error skips remaining cleanup; no async adapter | Attempts all cleanup; async adapter but repeated cancellation escapes wait | Close-failure, CUDA-failure and repeated-cancel regressions | Combined: detach references, attempt every close, wait through repeated cancellation, preserve primary errors |
| PR #125 consumer | No engine-owned runner adapter | Actual formatter and bound-runner job lifecycle tests | Exact PR #125 source tests with two runtime adapter bindings | B: compatible; untouched server stubs remain unfinished |

Unmodified A in the same real-model environment: **473 passed, 5 skipped,
3 failed**. Its three failures inspect shared-process ML imports; prior upstream
tests already imported Torch, so these failures do not establish eager import.
Unmodified B: **517 passed, 5 skipped**. Final uses fresh-interpreter import
isolation and adds semantic regressions; counts alone did not decide selection.

The real Basmalah result does not demonstrate a canonical boundary defect: even
choosing an endpoint leaves edit cost above the unchanged 0.25 acceptance gate.
A's low reported cost excludes canonical deletions and rewards dispersed matches.
Acoustic/reference mismatch remains a real deployment limitation requiring more
recitations, not permission to lower gates or equate the two vocabularies.

Reproduce the full suite with the pinned artifacts and actual consumer available:

```sh
RUN_REAL_MODEL=1 NUMBA_CACHE_DIR=/tmp/review120-numba \
ZIPFORMER_ARTIFACT_DIR=/tmp/issue120-models \
QURAN_AUDIO_PATH=/tmp/issue120-real-audio.mp3 \
PR125_SERVER_PATH=/tmp/munajjam-pr125/server.py \
PYTHONPATH=munajjam pytest -q -rs
```

Additional fixes are covered in `test_review120_regressions.py`. The reference
implementation and tests are deterministic; recorded-emission replay is labeled
separately from the actual public inference integration. No models, audio,
notebooks or generated evidence are committed.
