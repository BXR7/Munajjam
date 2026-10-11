# Issue #120 — Requirements-to-Tests Matrix

| Requirement | Implementation | Evidence / status |
|---|---|---|
| Issue #120 acceptance 1: integrate all three models and load on demand | Production factory wires the segmenter, Zipformer reference aligner, and Wav2Vec2 CTC aligner; heavyweight imports remain lazy | Import-isolation and factory/backend unit tests; no real full-stack run claimed |
| Issue #120 acceptance 2: constrain trellis alignment to breath groups | Per-breath Wav2Vec2 CTC uses each detected group and clips post-roll at the next group/audio boundary | `test_post_roll_clipped_at_next_breath`, full-Surah multi-breath tests |
| Issue #120 acceptance 3: retain trailing phonetic decay | Configurable post-roll is preserved within safe next-breath/audio bounds | `test_post_roll_retention`, `test_post_roll_bounded_by_audio_end` |
| Issue #120 acceptance 4: comprehensive passing unit suite | Regression coverage for API, reference, mapping, aggregation, imports, cleanup, and backends | 472 passed, 3 skipped; real-audio/GPU evidence unavailable in this sandbox |
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
