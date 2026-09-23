"""
clean_audio.py  --  Cut out background/silence, keep only the actual sound.

WHAT THIS DOES

Many of your Freesound downloads probably look like this:

    [quiet room] --- CHEW --- [quiet gap] --- CHEW --- [quiet gap] --- CHEW --- [quiet]

This script finds where the actual sound events are and exports each one as its
own clean clip, throwing away the silence and background noise between them.
It never touches your original files, only writes new copies.

This is an automatic first pass, not the same as FOAMS's approach in their
documentation, where they listened to every file by ear in Audacity and hand-
marked onset/offset within 50ms. This script uses volume alone to guess where
sound is, so it will occasionally get it wrong on quiet sounds or noisy
recordings. Skim the output folder afterward and delete anything that looks
wrong. For most files this saves you from clicking through everything by hand,
which is the whole point.

THE ONE SETTING THAT MATTERS: --top-db

Confusing at first, so read this once.

  LOWER numbers  = stricter  = only very quiet audio counts as silence
                   = splits MORE aggressively, finds more separate events
  HIGHER numbers = looser    = more audio counts as "not silence"
                   = splits LESS, tends to merge everything into one blob

If your noisy background is getting kept as part of the sound: LOWER the number.
If one continuous sound is getting chopped into fragments: RAISE the number.

Default is 20, tested against realistic background noise levels. Try 15 if a
file's background is louder than usual and events are still merging together;
try 30 if a single continuous sound is getting fragmented into pieces.

RUN
    .venv\\Scripts\\python clean_audio.py --in training_data --out training_data_clean
"""

import argparse
import csv
import os
import shutil
import sys

import numpy as np

import run_test as RT   # find_ffmpeg, audio conversion helpers

AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4",
            ".mov", ".m4v", ".aiff", ".aif")


def find_class_folders(root):
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                 if os.path.isdir(os.path.join(root, d)))


def find_audio(folder):
    return [f for f in sorted(os.listdir(folder))
            if f.lower().endswith(AUDIO_EXT)]


def merge_close_intervals(intervals, sr, min_gap_sec):
    """If two detected sounds are very close together, treat them as one
    event instead of two. Otherwise a single wet chewing sound with a tiny
    natural pause in it could get sliced into unnaturally short fragments."""
    if len(intervals) == 0:
        return intervals
    min_gap = int(min_gap_sec * sr)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] <= min_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged


def clean_one_file(ffmpeg, src_path, out_dir, stem, top_db, pad_sec,
                   min_gap_sec, min_len_sec):
    import librosa
    import soundfile as sf

    tmp_wav = os.path.join(out_dir, f"_tmp_{stem}.wav")
    _convert(ffmpeg, src_path, tmp_wav)

    y, sr = librosa.load(tmp_wav, sr=44100, mono=True)
    os.remove(tmp_wav)

    raw_intervals = librosa.effects.split(y, top_db=top_db)
    intervals = merge_close_intervals(raw_intervals, sr, min_gap_sec)

    pad = int(pad_sec * sr)
    written = []
    for i, (start, end) in enumerate(intervals):
        start = max(0, start - pad)
        end = min(len(y), end + pad)
        if (end - start) / sr < min_len_sec:
            continue
        clip = y[start:end]
        out_path = os.path.join(out_dir, f"{stem}_evt{i:02d}.wav")
        sf.write(out_path, clip, sr)
        written.append({
            "file": os.path.basename(out_path),
            "source": os.path.basename(src_path),
            "start_sec": round(start / sr, 3),
            "end_sec": round(end / sr, 3),
            "duration_sec": round((end - start) / sr, 3),
        })
    return written, len(y) / sr


def _convert(ffmpeg, src, dst):
    import subprocess
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", src, "-vn",
                    "-ac", "1", "-ar", "44100", dst], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True,
                    help="folder with one subfolder per class")
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--top-db", type=float, default=20,
                    help="lower = stricter/more splitting, higher = looser/less splitting")
    ap.add_argument("--pad", type=float, default=0.08,
                    help="seconds of padding kept around each detected event, "
                         "so the attack/decay of the sound isn't clipped")
    ap.add_argument("--min-gap", type=float, default=0.15,
                    help="events closer together than this (seconds) get merged into one")
    ap.add_argument("--min-len", type=float, default=0.2,
                    help="discard anything shorter than this (seconds), usually a click or pop")
    args = ap.parse_args()

    ffmpeg = RT.find_ffmpeg()
    classes = find_class_folders(args.src)
    if not classes:
        sys.exit(f"No class subfolders found in {args.src}")

    manifest = []
    print(f"{'class':<18}{'source files':>13}{'events found':>14}{'avg per file':>14}")
    print("-" * 60)

    for cls in classes:
        src_dir = os.path.join(args.src, cls)
        out_dir = os.path.join(args.out, cls)
        os.makedirs(out_dir, exist_ok=True)

        files = find_audio(src_dir)
        total_events = 0
        for f in files:
            stem = os.path.splitext(f)[0]
            written, duration = clean_one_file(
                ffmpeg, os.path.join(src_dir, f), out_dir, stem,
                args.top_db, args.pad, args.min_gap, args.min_len)
            for row in written:
                row["class"] = cls
            manifest.extend(written)
            total_events += len(written)
            flag = "  <- check this one" if len(written) == 0 or len(written) > 8 else ""
            print(f"   {f:<40} {duration:5.1f}s -> {len(written)} event(s){flag}")

        avg = total_events / len(files) if files else 0
        print(f"{cls:<18}{len(files):>13}{total_events:>14}{avg:>14.1f}")
        print()

    man_path = os.path.join(args.out, "clean_manifest.csv")
    if manifest:
        with open(man_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["class", "source", "file",
                                              "start_sec", "end_sec", "duration_sec"])
            w.writeheader()
            w.writerows(manifest)

    print(f"Manifest: {man_path}")
    print()
    print("NEXT: open the output folder and skim it. Delete any clip that looks wrong,")
    print("especially ones flagged '<- check this one' above (0 events found, or")
    print("suspiciously many small fragments from one file).")
    print()
    print("If a whole class is getting 0 events or way too many, rerun with a")
    print("different --top-db for just that class rather than adjusting everything:")
    print(f"    .venv\\Scripts\\python clean_audio.py --in {args.src} --out {args.out}"
         f" --top-db 20")


if __name__ == "__main__":
    main()
