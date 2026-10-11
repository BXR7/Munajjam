# Issue #120 — Public Runtime Validation

## Implemented path

`HybridQuranAligner.align_surah()` segments audio, maps Zipformer emissions against the
complete bundled Hafs/Warsh Surah with ordered character DP, then aligns only anchored
word ranges locally with Wav2Vec2 CTC. Results retain source digests, canonical match
coverage, token/frame evidence, and partial-result warnings. Introductory Basmalah is
source-verified and excluded from ayah labels by default outside Al-Fatiha. Word
probability is the arithmetic mean of local character-frame scores and is **uncalibrated**.

The public Zipformer artifact is pinned to
[`Alimalas/munajjam-onnx-models`](https://huggingface.co/Alimalas/munajjam-onnx-models/tree/5dbab4db48a88f5a2a76ead282b2bc3d4b958ee0),
revision `5dbab4db48a88f5a2a76ead282b2bc3d4b958ee0`. The token table is checked against
SHA-256 `252c10687e442aa9291973065fae19fa39bcd681c4f5612ec496a647e20b43a1`. Wav2Vec2
and Zipformer token IDs belong to independent vocabularies; no ID bridge is used.

## Validation and limits

Deterministic tests cover complete and partial Surahs, multiple/one/no breath groups,
low-quality and too-short matches, Basmalah, JSON-ready response fields, lifecycle
cleanup, and CTC confidence aggregation. They use injected fakes and do not download
models. The opt-in public Zipformer test needs `RUN_REAL_MODEL=1`, a recording at
`QURAN_AUDIO_PATH`, and the model runtime (or `ZIPFORMER_ARTIFACT_DIR`). It skipped in
this run because no real audio fixture was available.

The legacy `tests/integration/test_real_data.py` imports Torch during collection. Torch
was not installed in this sandbox, so a single unfiltered `pytest -q` could not collect
that optional Whisper integration module. No full segmenter → Zipformer → Wav2Vec2
real-audio inference is claimed here. The applicable suite, excluding only that legacy
module, finished with **472 passed, 3 skipped**. Changed Python files passed Ruff and
`ruff format --check`; `git diff --check` passed. The opt-in Zipformer test was also run
with `RUN_REAL_MODEL=1` and skipped explicitly because no real Quran recording was
available at `QURAN_AUDIO_PATH`.

## Sources

- [Pinned public Zipformer model](https://huggingface.co/Alimalas/munajjam-onnx-models/tree/5dbab4db48a88f5a2a76ead282b2bc3d4b958ee0)
- [Pinned Arabic Wav2Vec2 model](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic/tree/af46c2d8531b8dcbb5e23b952f739b372c2e5d2d)
- [Recitation segmenter](https://huggingface.co/obadx/recitation-segmenter-v2)
