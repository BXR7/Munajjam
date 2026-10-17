import sys
import os
import pathlib
import json
import hashlib
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
run_dir = str(_SCRIPT_DIR.parent)
corpus_dir = os.path.join(run_dir, "corpus")
gt_file = os.path.join(run_dir, "manifests", "ground_truth_segments.json")

with open(gt_file, "r", encoding="utf-8") as f:
    gt_db = json.load(f)

raw_records = [
    # Development Partition
    {"id": "DEV-112-AFASY", "file": "112.mp3", "surah": 112, "reciter": "Mishari Rashid al-Afasy", "riwayah": "hafs", "partition": "development", "pace_style": "Murattal Medium"},
    {"id": "DEV-001-AFASY", "file": "001.mp3", "surah": 1, "reciter": "Mishari Rashid al-Afasy", "riwayah": "hafs", "partition": "development", "pace_style": "Murattal Slow"},
    # Calibration Partition
    {"id": "CAL-112-HUSARY", "file": "husary_112.mp3", "surah": 112, "reciter": "Mahmoud Khalil Al-Husary", "riwayah": "hafs", "partition": "calibration", "pace_style": "Tahqiq Standard"},
    {"id": "CAL-001-HUSARY", "file": "husary_001.mp3", "surah": 1, "reciter": "Mahmoud Khalil Al-Husary", "riwayah": "hafs", "partition": "calibration", "pace_style": "Tahqiq Standard"},
    {"id": "CAL-108-AFASY", "file": "108.mp3", "surah": 108, "reciter": "Mishari Rashid al-Afasy", "riwayah": "hafs", "partition": "calibration", "pace_style": "Murattal Slow"},
    # Untouched Evaluation Holdout (Hafs)
    {"id": "EVAL-108-HUSARY", "file": "husary_108.mp3", "surah": 108, "reciter": "Mahmoud Khalil Al-Husary", "riwayah": "hafs", "partition": "evaluation_holdout", "pace_style": "Tahqiq Standard"},
    {"id": "EVAL-113-HUSARY", "file": "husary_113.mp3", "surah": 113, "reciter": "Mahmoud Khalil Al-Husary", "riwayah": "hafs", "partition": "evaluation_holdout", "pace_style": "Hadar Fast"},
    {"id": "EVAL-114-HUSARY", "file": "husary_114.mp3", "surah": 114, "reciter": "Mahmoud Khalil Al-Husary", "riwayah": "hafs", "partition": "evaluation_holdout", "pace_style": "Tahqiq Standard"},
    {"id": "EVAL-113-AFASY", "file": "113.mp3", "surah": 113, "reciter": "Mishari Rashid al-Afasy", "riwayah": "hafs", "partition": "evaluation_holdout", "pace_style": "Murattal Medium"},
    {"id": "EVAL-114-AFASY", "file": "114.mp3", "surah": 114, "reciter": "Mishari Rashid al-Afasy", "riwayah": "hafs", "partition": "evaluation_holdout", "pace_style": "Murattal Slow"},
    {"id": "EVAL-112-MINSHAWI", "file": "minshawi_112.mp3", "surah": 112, "reciter": "Mohamed Siddiq al-Minshawi", "riwayah": "hafs", "partition": "evaluation_holdout", "pace_style": "Tahqiq Slow"},
    {"id": "EVAL-001-MINSHAWI", "file": "minshawi_001.mp3", "surah": 1, "reciter": "Mohamed Siddiq al-Minshawi", "riwayah": "hafs", "partition": "evaluation_holdout", "pace_style": "Tahqiq Slow"},
    {"id": "EVAL-067-AFASY", "file": "067.mp3", "surah": 67, "reciter": "Mishari Rashid al-Afasy", "riwayah": "hafs", "partition": "evaluation_holdout", "pace_style": "Murattal Full Surah"},
    # Warsh Partition (Evaluated Strictly Separately)
    {"id": "WARSH-112-HUSARY", "file": "warsh_husary_112.mp3", "surah": 112, "reciter": "Mahmoud Khalil Al-Husary", "riwayah": "warsh", "partition": "warsh_holdout", "pace_style": "Tahqiq Warsh"},
    {"id": "WARSH-112-YASSEN", "file": "warsh_yassen_112.mp3", "surah": 112, "reciter": "Yassen Al-Jazaery", "riwayah": "warsh", "partition": "warsh_holdout", "pace_style": "Murattal Warsh"},
    {"id": "WARSH-001-YASSEN", "file": "warsh_yassen_001.mp3", "surah": 1, "reciter": "Yassen Al-Jazaery", "riwayah": "warsh", "partition": "warsh_holdout", "pace_style": "Murattal Warsh"},
]

manifest = {
    "corpus_version": "2.0-ablation-frozen",
    "total_audio_files": len(raw_records),
    "creation_timestamp": "2026-10-16T16:55:00Z",
    "evaluation_standard": "Zero-Input Independent Benchmarking",
    "recordings": []
}

for rec in raw_records:
    p = os.path.join(corpus_dir, rec["file"])
    info = sf.info(p)
    size_b = os.path.getsize(p)
    
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    sha256 = h.hexdigest()
    
    has_gt = rec["id"] in gt_db
    gt_info = gt_db.get(rec["id"], {})
    gt_ayahs = gt_info.get("ayah_segments", [])
    num_gt_words = sum(len(a.get("segments", [])) for a in gt_ayahs)
    
    entry = {
        **rec,
        "path": p,
        "bytes": size_b,
        "sha256": sha256,
        "duration_seconds": round(info.duration, 3),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "has_independent_ground_truth": has_gt,
        "ground_truth_provenance": "Quran.com Official Reviewed API (Proxy)" if has_gt else "UNESTABLISHED (No independent API)",
        "ground_truth_ayah_count": len(gt_ayahs),
        "ground_truth_word_count": num_gt_words
    }
    manifest["recordings"].append(entry)

out_manifest = os.path.join(run_dir, "CORPUS_AND_GROUND_TRUTH_MANIFEST.json")
with open(out_manifest, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"Generated frozen corpus manifest with {len(manifest['recordings'])} entries at {out_manifest}")
