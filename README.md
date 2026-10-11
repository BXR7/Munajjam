# Munajjam

**A Python library and API Server to synchronize Quran ayat with audio recitations.**

Munajjam uses AI-powered speech recognition to automatically generate precise timestamps for each ayah in a Quran audio recording.

## API Server & Docker (Recommended)

You can run Munajjam as a standalone API server with asynchronous processing and GPU support.

### Running with Docker

The easiest way to run the API server is using Docker Compose:

```bash
git clone https://github.com/Itqan-community/munajjam.git
cd munajjam

# To run with GPU support (default)
docker compose up --build

# To run with CPU only
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
```

### API Endpoints

Once the server is running (default: http://localhost:8000), you can use the following endpoints:

- **`POST /align/{surah_number}`**: Upload an audio file for a specific surah. Returns a `job_id`.
  - Form Data: `file` (audio file), `riwaya` (e.g., "hafs")
- **`GET /align/status/{job_id}`**: Check the status of the alignment job and get the results when ready.
- **`GET /health`**: Health check.

## Library Installation

If you want to use Munajjam as a Python library:

Clone the repository:

```bash
git clone https://github.com/Itqan-community/munajjam.git
cd munajjam/munajjam
```

Install the package:

```bash
pip install .
```

For faster transcription with [faster-whisper](https://github.com/SYSTRAN/faster-whisper):

```bash
pip install ".[faster-whisper]"
```

For development (editable install):

```bash
pip install -e ".[dev]"
```

## Quick Start (Library)

### 1. Download a sample recitation

Download a sample audio file (Surah Al-Fatiha):

```bash
curl -L -o 001.mp3 "https://pub-9ee413c8af4041c6bd5223d08f5d0f0f.r2.dev/media/uploads/assets/11/recitations/001.mp3"
```

> **Note:** Audio files should be named by surah number (e.g., `001.mp3`, `002.mp3`).
> Browse more recitations at [cms.itqan.dev](https://cms.itqan.dev)

### 2. Run the alignment

```python
from munajjam.core import align
from munajjam.data import load_surah_ayahs
from munajjam.transcription import WhisperTranscriber

# Transcribe audio
with WhisperTranscriber() as transcriber:
    segments = transcriber.transcribe("001.mp3")

# Align to ayahs (uses auto strategy by default; override with "greedy", "dp", or "hybrid")
ayahs = load_surah_ayahs(1)
results = align("001.mp3", segments, ayahs)

# Get timestamps
for result in results:
    print(
        f"Ayah {result.ayah.ayah_number}: "
        f"{result.start_time:.2f}s - {result.end_time:.2f}s"
    )
```

### 3. Output

```
Ayah 1: 5.62s - 9.57s
Ayah 2: 10.51s - 14.72s
Ayah 3: 15.45s - 18.53s
Ayah 4: 19.21s - 22.54s
Ayah 5: 23.27s - 28.19s
Ayah 6: 29.00s - 33.07s
Ayah 7: 33.98s - 46.44s
```

## Features

- **API Server** - Async FastAPI server for handling concurrent alignment jobs
- **Whisper Transcription** - Uses faster-whisper as default backend with Quran-tuned models
- **Four Alignment Strategies** - Auto, Hybrid, DP, and Greedy
- **Arabic Text Normalization** - Handles diacritics, hamzas, and character variations
- **Automatic Drift Correction** - Multi-pass zone realignment for long recordings
- **Quality Metrics** - Confidence scores for each aligned ayah
- **Phonetic Similarity** - Arabic ASR confusion-aware similarity scoring
- **Word-level Precision** - Uses per-word timestamps (when available) to improve drift recovery

## Alignment Strategies

The default `auto` strategy works best for most cases. You can override it:

```python
from munajjam.core import Aligner

# Auto (recommended) - picks the best strategy, full pipeline by default
aligner = Aligner("001.mp3")

# Hybrid - DP with greedy fallback (legacy)
aligner = Aligner(
    "001.mp3",  # Audio file path (required)
    strategy="auto",  # "greedy", "dp", "hybrid", or "auto" (default)
    quality_threshold=0.85,  # Similarity threshold for high-quality alignment
    fix_drift=True,  # Run zone realignment for long surahs
    fix_overlaps=True,  # Fix overlapping ayah timings
    min_gap=0.3,  # Minimum gap between consecutive ayahs (seconds)
    energy_snap=True,  # Snap boundaries to energy minima (default True)
)

results = aligner.align(segments, ayahs)
```

## Examples

See the [examples](./examples) directory for more usage patterns:

- `01_basic_usage.py` - Simple transcription and alignment
- `02_comparing_strategies.py` - Compare alignment strategies
- `03_advanced_configuration.py` - Custom settings and options
- `04_batch_processing.py` - Process multiple files

## Requirements

- Python 3.10+
- PyTorch 2.0+
- FFmpeg (for audio processing)
- Docker & Docker Compose (Optional, for running the API server)

## Community

- [Website](https://munajjam.itqan.dev)
- [ITQAN Community](https://community.itqan.dev)

## Acknowledgments

- [Tarteel AI](https://tarteel.ai) for the Quran-specialized Whisper model

## License

MIT License - see [LICENSE](./LICENSE) for details.

## Opt-in Tripartite Hybrid Aligner (Issue #120)

Munajjam includes an **experimental, opt-in** full-Surah alignment engine combining
breath-group segmentation, global Zipformer-to-canonical Quran mapping, and local
Wav2Vec2 CTC forced alignment.

This engine is **not** used by the default `auto` strategy or the server. Call
`align_surah()` to select a Surah and riwaya; verse ranges are mapped automatically from
the full canonical Surah, not supplied per breath group:

```python
from munajjam.core.hybrid_aligner import HybridQuranAligner, load_audio_file

with HybridQuranAligner.from_pretrained(device="cpu") as aligner:
    result = aligner.align_surah(load_audio_file("001.mp3"), surah_id=1, riwaya="hafs")
    for ayah in result:
        print(ayah.ayah_number, ayah.start_time, ayah.end_time, ayah.text)
```

The bundled provider supplies canonical Hafs/Warsh text with source/content SHA-256
provenance. Zipformer characters are globally aligned against that reference; only
breath groups with canonical anchors receive ayah/word assignments. Weak matches fail
closed, while clean partial coverage is returned with a `partial` status and warning.
Wav2Vec2 targets come from its pinned tokenizer; Zipformer and Wav2Vec2 IDs are never
bridged.

For non-Fatiha Surahs, a fully recognized introductory Basmalah is matched to the
bundled source and excluded from ayah labels by default; `include_basmalah=True` adds it
to the reference, without duplicating Fatiha 1:1. Isti'adhah requires an explicit,
source-verified `VerifiedSpecialPhrase`. Empty Warsh placeholders remain unavailable
and fail closed. `repetition_attach=True` is rejected because the upstream contract does
not define its attachment semantics.

### Architecture

```
audio
  ↓
breath segmentation  (pinned obadx/recitation-segmenter-v2 model; group count varies)
  ↓
full-Surah canonical Quran reference (provenance-bearing)
  ↓
Zipformer phoneme emissions (public Alimalas ONNX artifact via sherpa-onnx)
  ↓
global ordered character DP → constrained per-breath Wav2Vec2 CTC
  ↓
validated final alignment
```

The production Zipformer route uses the public, immutable-revision Hugging Face repository
[`Alimalas/munajjam-onnx-models`](https://huggingface.co/Alimalas/munajjam-onnx-models),
specifically `model_zipformer/zipformer_p_arabic_v3.onnx` and `tokens.txt`, through
`sherpa-onnx.OnlineRecognizer.from_zipformer2_ctc`. Its 251-token output is decoded in
the Zipformer vocabulary and timestamped at the 40 ms emission grid. This is independent
of the 51-token Wav2Vec2 vocabulary; no numeric vocabulary bridge is inferred. The
character-DP stage retains token, frame, score, and canonical-source provenance.
The Wav2Vec2 output probability is the arithmetic mean of local CTC character scores and
is explicitly **uncalibrated**.

The public model is pinned to revision `5dbab4db48a88f5a2a76ead282b2bc3d4b958ee0` and uses
sherpa-onnx's exporter-defined streaming state and feature pipeline.

### Optional model dependencies

The hybrid aligner does **not** import any heavy ML framework at import time. Heavy
libraries are lazily imported only when a matching provider/backend is explicitly
constructed:

- **torch** — optional, required only by neural model backends.
- **transformers** — optional, required only by Transformer-based CTC backends.
- **whisperx** — optional, used by the legacy server transcription path.
- **sherpa-onnx** — optional, for sherpa-onnx CTC backends.
- **faster-whisper** — optional, for faster-whisper transcription backends.

The base package already declares shared Transformer and sherpa-onnx dependencies; the
optional `segmenter` extra adds the pinned Torch/torchaudio and segmenter stack. Install
the optional extras when enabling the tripartite aligner:

```bash
cd munajjam
pip install '.[segmenter,zipformer]'
```

### Model access requirements

`from_pretrained()` binds the production aligner to the pinned public artifact. Custom
`ZipformerNeuralAligner` construction requires explicit evidence and fails **closed**
when evidence is missing or invalid:

- **Repository:** production defaults to `Alimalas/munajjam-onnx-models`.
- **Revision:** must be an immutable 40-character git SHA.
- **Approval:** must be explicitly set to `True`.
- **`tokens.txt`:** the pinned file is parsed by its explicit `<piece> <id>`
  entries, not by line position; it must define all 251 IDs and `<blank>` at ID 250.
  The pinned vocabulary SHA-256 is
  `252c10687e442aa9291973065fae19fa39bcd681c4f5612ec496a647e20b43a1`.
- **Feature extractor:** sherpa-onnx uses 16 kHz mono audio and 80-dimensional fbank features.
- **License:** the Zipformer model is subject to its upstream license; verify
  compliance before use.

### Limitations

- `ZipformerNeuralAligner.from_pretrained()` loads the pinned model into the
  Hugging Face cache; an explicit `backend_factory` can instead be injected for tests
  or a separately managed artifact directory.
- Real-model smoke tests are **opt-in** and use the public Hugging Face artifact without
  credentials. The complete tripartite E2E additionally requires the segmenter, Wav2Vec2
  runtime, and real audio.
- Wav2Vec2 defaults to CUDA when available and otherwise uses CPU; an explicit device
  can be supplied.
- Post-roll retention is bounded by the next breath group boundary and the audio end.
- No timestamps are ever synthesized without validated acoustic evidence from the CTC
  trellis.

### Real-model smoke test instructions

```bash
RUN_REAL_MODEL=1 PYTHONPATH=./munajjam pytest -m real_model \
  tests/integration/test_zipformer_real_model.py
```

The opt-in test requires a real audio file (`QURAN_AUDIO_PATH`); the public model is
downloaded at its pinned revision unless `ZIPFORMER_ARTIFACT_DIR` is supplied. Missing
audio causes a pytest skip, not a pass.

See [`docs/issue_120_requirements_matrix.md`](docs/issue_120_requirements_matrix.md) for
the maintainer-requirements-to-tests matrix and
[`docs/issue_120_public_runtime_validation.md`](docs/issue_120_public_runtime_validation.md)
for the exact sandbox validation limits.
