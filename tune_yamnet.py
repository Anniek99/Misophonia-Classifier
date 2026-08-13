"""
tune_yamnet.py  --  Find out what YAMNet actually calls your sounds.

WHY THIS EXISTS

YAMNet knows 521 sound names. To use it for your 5 triggers, someone has to
decide which of its 521 names count as which of your sounds. That mapping was
guessed, and two of the guesses ('Slurp', 'Click') are not real AudioSet names
at all, so gulping and pen_clicking could never score above zero.

This script stops guessing. It plays every clip through YAMNet, records what
YAMNet says, and builds the mapping from what actually fires.

RUN:
    .venv\\Scripts\\python tune_yamnet.py

It prints a LABEL_MAP you can paste into run_test.py.

WORTH KNOWING FOR YOUR WRITE-UP
An empirically derived mapping is a better methods section than a hand-picked
one. You can say you built it from measured activations rather than intuition.
"""

import os
import re
import sys
import shutil
import tempfile

import numpy as np
import pandas as pd

import run_test as RT   # reuse the file finding and clip cutting

TOP_N = 12          # labels to show per sound
SUGGEST_N = 4       # labels to put in the suggested map


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    model = os.path.join(here, "yamnet.tflite")
    cmap = os.path.join(here, "yamnet_class_map.csv")

    if not (os.path.exists(model) and os.path.exists(cmap)):
        sys.exit("Need yamnet.tflite and yamnet_class_map.csv here.\n"
                 "Run:  .venv\\Scripts\\python get_yamnet.py")

    ffmpeg = RT.find_ffmpeg()
    ovs = RT.find_ovs_folder()

    # ------------------------------------------------------------ collect
    tmp = tempfile.mkdtemp(prefix="tune_")
    rows = []

    for f in sorted(os.listdir(ovs)):
        if not f.lower().endswith((".mov", ".mp4", ".m4v", ".avi")):
            continue
        key = RT.video_key(f)
        sound = RT.VIDEO_TO_SOUND.get(key)
        if not sound:
            continue
        clips, _ = RT.extract_clips(ffmpeg, os.path.join(ovs, f), tmp, key)
        for c in clips:
            rows.append({"clip": c, "sound": sound})

    extra_root = os.path.join(here, RT.EXTRA_AUDIO_FOLDER)
    if os.path.isdir(extra_root):
        AUDIO = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4", ".mov")
        for sub in sorted(os.listdir(extra_root)):
            d = os.path.join(extra_root, sub)
            if not os.path.isdir(d):
                continue
            for i, f in enumerate(sorted(os.listdir(d))):
                if not f.lower().endswith(AUDIO):
                    continue
                stem = f"x{sub}{i}"
                try:
                    clips, _ = RT.extract_clips(ffmpeg, os.path.join(d, f), tmp, stem)
                except Exception:
                    continue
                for c in clips:
                    rows.append({"clip": c, "sound": sub})

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("No audio found.")

    print(f"{len(df)} clips across {df['sound'].nunique()} sounds")
    print(df["sound"].value_counts().to_string())
    print()

    # ------------------------------------------------------------ run YAMNet
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    import librosa
    it = Interpreter(model_path=model)
    it.allocate_tensors()
    yin, yout = it.get_input_details()[0], it.get_output_details()[0]
    need = int(yin["shape"][-1])
    rank = len(yin["shape"])
    names = pd.read_csv(cmap)["display_name"].tolist()

    print("Running YAMNet on every clip. This takes a minute...")
    scores = []
    for p in df["clip"]:
        wav, _ = librosa.load(p, sr=16000, mono=True)
        wav = wav.astype(np.float32)
        if len(wav) < need:
            wav = np.pad(wav, (0, need - len(wav)))
        chunk = wav[:need]
        it.set_tensor(yin["index"], chunk if rank == 1 else chunk.reshape(1, -1))
        it.invoke()
        o = it.get_tensor(yout["index"])
        scores.append(o[0] if o.ndim == 2 else o)
    S = np.array(scores)
    print("done\n")

    sounds = sorted(df["sound"].unique())
    mean_by_sound = {s: S[(df["sound"] == s).values].mean(axis=0) for s in sounds}

    # ------------------------------------------------------ what fires
    RT.hr()
    print("WHAT YAMNET ACTUALLY HEARS")
    RT.hr()
    print("Average score for each AudioSet label, per sound.\n")

    for s in sounds:
        m = mean_by_sound[s]
        n_clips = int((df["sound"] == s).sum())
        print(f"--- {s}  ({n_clips} clips) ---")
        for i in np.argsort(m)[::-1][:TOP_N]:
            print(f"      {m[i]:.4f}  {names[i]}")
        print()

    # ------------------------------------------------ discriminative labels
    RT.hr()
    print("SUGGESTED LABEL MAP")
    RT.hr()
    print("A label is only useful if it fires for ONE sound and not the others.")
    print("'Speech' scores high on everything, so it tells you nothing about")
    print("chewing. These are picked by how much each label separates one sound")
    print("from the rest, not by raw score.\n")

    lines = []
    for s in sounds:
        if s == "other":
            continue
        mine = mean_by_sound[s]
        others = np.max([mean_by_sound[o] for o in sounds if o != s], axis=0)
        gap = mine - others
        picks = [i for i in np.argsort(gap)[::-1][:SUGGEST_N] if gap[i] > 0]
        if not picks:
            lines.append(f'    "{s}": [],   # nothing separates this one')
            continue
        quoted = ", ".join(f'"{names[i]}"' for i in picks)
        lines.append(f'    "{s}": [{quoted}],')
        print(f"{s}:")
        for i in picks:
            print(f"      {names[i]:<34} score {mine[i]:.4f}  gap {gap[i]:+.4f}")
        print()

    print("Paste this over YAMNET_MAP in run_test.py:\n")
    print("YAMNET_MAP = {")
    for ln in lines:
        print(ln)
    print("}")
    print()

    # ------------------------------------------------------------ threshold
    RT.hr()
    print("SUGGESTED THRESHOLD")
    RT.hr()
    best = []
    for s in sounds:
        if s == "other":
            continue
        m = mean_by_sound[s]
        others = np.max([mean_by_sound[o] for o in sounds if o != s], axis=0)
        gap = m - others
        picks = [i for i in np.argsort(gap)[::-1][:SUGGEST_N] if gap[i] > 0]
        if picks:
            best.append(max(m[i] for i in picks))
    if best:
        lo = min(best)
        suggested = round(float(max(0.001, lo * 0.4)), 4)
        print(f"Weakest winning score across your sounds: {float(lo):.4f}")
        print(f"Try:  YAMNET_THRESHOLD = {suggested}")
        print()
        print(f"The current value is {RT.YAMNET_THRESHOLD}. If your scores are all smaller")
        print("than that, every clip falls through to 'other', which is what")
        print("happened to gulping and fork_scraping.")
    print()

    shutil.rmtree(tmp, ignore_errors=True)

    RT.hr()
    print("NEXT")
    RT.hr()
    print("1. Paste the LABEL_MAP and THRESHOLD above into run_test.py")
    print("2. Run:  .venv\\Scripts\\python run_test.py")
    print("3. YAMNet should stop scoring zero on gulping and pen_clicking")
    print()
    print("If a sound shows 'nothing separates this one', YAMNet genuinely")
    print("cannot tell it apart from your other sounds. That is a real finding")
    print("and an argument for fine-tuning rather than using it off the shelf.")
    RT.hr()


if __name__ == "__main__":
    main()
