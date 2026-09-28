"""
segment_ovs_test.py  --  Segment the OVS test videos into a NEW folder.

SAFETY: this only ever READS from your OVS Videos folder. It never writes to
it, never deletes anything in it, never modifies it. Your original videos are
completely untouched no matter what happens when you run this.

WHAT THIS DOES

Your OVS test videos aren't sorted into class folders the way training_data
is, they're just loose files named Vid17_O.mov, Vid18_O.mov, etc. This script
figures out which trigger class each video belongs to automatically (using
the same mapping run_test.py already uses), then runs the exact same
segmentation logic as clean_audio.py on each one, writing the result into a
new output folder with proper class subfolders, ready to use the same way
you already use training_data_clean.

This does NOT rerun your model or compute any accuracy numbers by itself.
It only produces the segmented clips (and optionally Audacity labels) so you
can look at them, or later point compare_segmented_test.py or a training run
at them.

RUN
    .venv\\Scripts\\python segment_ovs_test.py --out ovs_test_segmented
    .venv\\Scripts\\python segment_ovs_test.py --out ovs_test_segmented --audacity-labels
"""

import argparse
import os
import sys

import run_test as RT
import clean_audio as CA

AUDIO_EXT = (".mov", ".mp4", ".m4v", ".avi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="new folder to write segmented test clips into")
    ap.add_argument("--top-db", type=float, default=20)
    ap.add_argument("--pad", type=float, default=0.08)
    ap.add_argument("--min-gap", type=float, default=0.15)
    ap.add_argument("--min-len", type=float, default=0.2)
    ap.add_argument("--audacity-labels", action="store_true")
    args = ap.parse_args()

    ffmpeg = RT.find_ffmpeg()
    ovs = RT.find_ovs_folder()   # read-only from here on
    print(f"Reading from (read-only): {ovs}")
    print(f"Writing new segmented clips to: {args.out}\n")

    videos = [f for f in sorted(os.listdir(ovs)) if f.lower().endswith(AUDIO_EXT)]
    if not videos:
        sys.exit(f"No video files found in {ovs}")

    print(f"{'video':<14}{'class':<16}{'length':>9}{'events found':>14}")
    print("-" * 55)

    total = 0
    for f in videos:
        key = RT.video_key(f)
        sound = RT.VIDEO_TO_SOUND.get(key)
        if sound is None:
            continue   # not one of the 5 trigger classes, skip

        out_dir = os.path.join(args.out, sound)
        os.makedirs(out_dir, exist_ok=True)
        label_dir = (os.path.join(args.out, "_audacity_labels", sound)
                    if args.audacity_labels else None)

        stem = os.path.splitext(f)[0]
        written, duration = CA.clean_one_file(
            ffmpeg, os.path.join(ovs, f), out_dir, stem,
            args.top_db, args.pad, args.min_gap, args.min_len, label_dir)
        total += len(written)
        print(f"{key:<14}{sound:<16}{duration:>8.1f}s{len(written):>14}")

    print()
    print(f"{total} segmented clips written to {args.out}\\<class>\\")
    print()
    print("Your original OVS Videos folder was only read, never modified.")
    if args.audacity_labels:
        print(f"Audacity labels: {args.out}\\_audacity_labels\\<class>\\")
        print("Import against the ORIGINAL video's audio, same as before.")


if __name__ == "__main__":
    main()
