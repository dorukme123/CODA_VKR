"""
Usage:
    python -m src.data.pyannote_vad --manifest results/preprocessed/dusha/crowd_train/manifest.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="pyannote VAD on audio files")
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--hf-token", type=str, default=None,
                   help="HuggingFace token for pyannote models")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Process only first N samples (for testing)")
    return p.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)

    print("Loading pyannote VAD pipeline...")
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    model = Model.from_pretrained(
        "pyannote/segmentation-3.0",
        use_auth_token=args.hf_token,
    )
    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
        "min_duration_on": 0.1,    # min speech segment duration
        "min_duration_off": 0.1,   # min silence duration
    }
    pipeline.instantiate(HYPER_PARAMETERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))

    if args.max_samples:
        entries = entries[:args.max_samples]
    print(f"Processing {len(entries)} entries")

    # Run VAD on each audio file
    total_speech = 0.0
    total_duration = 0.0
    processed = 0

    for entry in tqdm(entries, desc="pyannote VAD"):
        audio_path = entry.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            continue

        try:
            waveform, sr = torchaudio.load(audio_path)
            duration = waveform.shape[1] / sr

            # Run VAD
            vad_result = pipeline({"waveform": waveform, "sample_rate": sr})

            # Extract speech segments
            speech_segments = []
            speech_duration = 0.0
            for segment in vad_result.get_timeline():
                speech_segments.append({
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                })
                speech_duration += segment.end - segment.start

            speech_ratio = speech_duration / max(duration, 0.001)

            entry["vad_speech_duration"] = round(speech_duration, 3)
            entry["vad_total_duration"] = round(duration, 3)
            entry["vad_speech_ratio"] = round(speech_ratio, 4)
            entry["vad_num_segments"] = len(speech_segments)
            entry["vad_segments"] = speech_segments

            total_speech += speech_duration
            total_duration += duration
            processed += 1

        except Exception as e:
            entry["vad_error"] = str(e)

    # Write updated manifest
    all_entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        all_entries = [json.loads(l) for l in f]

    # Merge VAD data back (match by ID)
    vad_by_id = {e["id"]: e for e in entries}
    for entry in all_entries:
        eid = entry["id"]
        if eid in vad_by_id:
            for key in ["vad_speech_duration", "vad_total_duration",
                        "vad_speech_ratio", "vad_num_segments", "vad_segments", "vad_error"]:
                if key in vad_by_id[eid]:
                    entry[key] = vad_by_id[eid][key]

    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Summary
    avg_ratio = total_speech / max(total_duration, 0.001)
    print(f"\npyannote VAD complete:")
    print(f"  Processed: {processed}/{len(entries)}")
    print(f"  Total audio: {total_duration / 3600:.1f}h")
    print(f"  Total speech: {total_speech / 3600:.1f}h")
    print(f"  Avg speech ratio: {avg_ratio:.2%}")
    print(f"  Manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
