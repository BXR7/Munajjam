# Issue #120 — Requirements-to-Tests Matrix

| Requirement | Implementation | Evidence / status |
|---|---|---|
| Full-Surah input (`audio_path`, `surah_id`, `riwaya`) | `HybridQuranAligner.align_surah()`; bundled provenance-bearing reference | `tests/unit/test_align_surah.py` |
| Breath detection and unequal breath/ayah boundaries | Segmenter output; global ordered Zipformer DP; inferred canonical word windows | One/multiple-group tests, including an ayah spanning breaths |
| Partial, missing, low-quality, ambiguous matches | Dense anchor windows, explicit coverage/error policy, warnings or fail-closed errors; bounded DP memory | Partial/no-speech/too-short tests, repeated-phrase monotonic DP test, DP budget test |
| Basmalah and Istiadhah | Source-verified Basmalah auto-detection/optional inclusion; Istiadhah requires `VerifiedSpecialPhrase` | Intro Basmalah and Istiadhah high-level tests; reference-provider provenance tests |
| Hafs and Warsh references | Riwaya-specific source data/digests; unavailable placeholders fail closed | Warsh alignment/provenance and missing-placeholder tests |
| Independent model vocabularies | Separate pinned 251-token Zipformer and 51-token Wav2Vec2 tokenizers; no ID bridge | Backend/tokenizer tests and result metadata |
| Word/ayah/breath response schema | `AyahAlignment`, `WordAlignment`, `BreathGroupResult`; seconds, pauses, boundary flag | Schema, monotonicity, bounds, pause and arithmetic-mean CTC score tests |
| `probability` semantics | Arithmetic mean of per-character local acoustic-frame posterior scores; explicitly uncalibrated | CTC local-posterior regression and per-word aggregation assertions |
| Progress, cleanup, lazy CPU import | Synchronous callback contract; idempotent `close()`, cached-CUDA cleanup only if Torch is already loaded; context manager | Callback, repeated-close, CUDA fake, import-isolation tests |
| `repetition_attach` | Rejected because PR #125 only supplies the flag; no verified attachment semantics found | Explicit fail-closed test; behavior awaits maintainer contract |
| PR #125 response serializer | Engine result iteration, exact data/word fields, JSON encoding; verified in detached PR #125 worktree | Actual `_build_success_response` returned serialized ayah data; its `_extract_breath_boundaries()` remains a deliberate `[]` stub, while `_format_breath_groups()` accepts engine boundaries. Endpoint breath-group extraction awaits PR #125 integration; no `server.py` change is included. |
| Public real-model evidence | Opt-in Zipformer smoke test provided | Skipped: no real recording fixture was available. No real full tripartite inference claimed. |
| Existing upstream regressions | Full applicable test tree, excluding the legacy Torch-importing real-data module | 472 passed, 3 skipped; see runtime validation note. |
