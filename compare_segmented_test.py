"""
compare_segmented_test.py  --  Same comparison as compare_finetuned.py, but the
OVS test videos are segmented into real sound events first, the same way
clean_audio.py segments training data, instead of sliced into arbitrary fixed
1-second windows.

WHY THIS MATTERS

Your training clips (after clean_audio.py) are tightly cropped around actual
detected sound events. Your test clips, until now, were arbitrary 1-second
windows with 0.5s hop, which may or may not line up with where the real sound
actually is in each OVS video. That is a mismatch between what the model
learned to recognize and what shape of clip it gets tested on.

This script fixes that by running the same event-detection segmentation used
in clean_audio.py on the OVS test videos, so both training and testing treat
audio the same way. It is the test-side counterpart to that script.

RUN
    .venv\\Scripts\\python compare_segmented_test.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TFHUB_CACHE_DIR", os.path.join(HERE, "tfhub_cache"))

import numpy as np
import pandas as pd

import run_test as RT
import fine_tune_yamnet as FT
import clean_audio as CA   # reuse the same event-detection logic as training

TOP_DB = 20        # same default as clean_audio.py
PAD = 0.15
MIN_GAP = 0.15
MIN_LEN = 0.2


def segment_ovs_videos(ffmpeg, allowed_classes, tmp_dir):
    """Same idea as clean_audio.py's clean_one_file, applied to the OVS test
    videos instead of training recordings. Returns a DataFrame of segmented
    event clips with their true labels."""
    import librosa
    import soundfile as sf

    ovs = RT.find_ovs_folder()
    rows = []

    for f in sorted(os.listdir(ovs)):
        if not f.lower().endswith((".mov", ".mp4", ".m4v", ".avi")):
            continue
        key = RT.video_key(f)
        sound = RT.VIDEO_TO_SOUND.get(key)
        if sound not in allowed_classes:
            continue

        master = os.path.join(tmp_dir, f"{key}_master.wav")
        import subprocess
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", os.path.join(ovs, f),
             "-vn", "-af", "loudnorm=I=-23:LRA=7:TP=-2",
             "-ac", "1", "-ar", "44100", master], check=True)

        y, sr = librosa.load(master, sr=44100, mono=True)
        os.remove(master)

        raw_intervals = librosa.effects.split(y, top_db=TOP_DB)
        intervals = CA.merge_close_intervals(raw_intervals, sr, MIN_GAP)

        pad_samp = int(PAD * sr)
        n_events = 0
        for i, (start, end) in enumerate(intervals):
            start = max(0, start - pad_samp)
            end = min(len(y), end + pad_samp)
            if (end - start) / sr < MIN_LEN:
                continue
            clip = y[start:end]
            out_path = os.path.join(tmp_dir, f"{key}_evt{i:02d}.wav")
            sf.write(out_path, clip, sr)
            rows.append({"clip": out_path, "true_label": sound,
                        "source": key, "start_sec": round(start / sr, 3)})
            n_events += 1
        print(f"   {key:<8} {sound:<16} -> {n_events} event(s) detected")

    return pd.DataFrame(rows)


def untrained_predict(mean_scores, name_to_idx):
    best, score = "other", 0.0
    for cls, names in RT.YAMNET_MAP.items():
        valid = [n for n in names if n in name_to_idx]
        if not valid:
            continue
        s = max(mean_scores[name_to_idx[n]] for n in valid)
        if s > score:
            best, score = cls, float(s)
    return "other" if score < RT.YAMNET_THRESHOLD else best


def report(name, y_true, y_pred, classes):
    present = [c for c in classes if c in set(y_true)]
    print(f"--- {name} ---")
    header = f"{'class':<16}{'n':>6}{'recall':>9}{'precision':>11}{'F1':>8}"
    print(header)
    print("-" * len(header))
    f1s = []
    for c in present:
        n = int((y_true == c).sum())
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        print(f"{c:<16}{n:>6}{rec*100:>8.0f}%{prec*100:>10.0f}%{f1:>8.3f}")
    acc = float((y_true == y_pred).mean())
    print("-" * len(header))
    print(f"clips {len(y_true)}   accuracy {acc*100:.1f}%   macro F1 {sum(f1s)/len(f1s):.3f}")
    print()
    return sum(f1s) / len(f1s)


def main():
    RT.hr()
    print("Untrained vs fine-tuned YAMNet, on SEGMENTED test events")
    print("(not fixed 1-second windows, same event-detection as clean_audio.py)")
    RT.hr()
    print()

    import tensorflow as tf
    import librosa

    yamnet = FT.load_yamnet()
    print("YAMNet loaded.\n")

    cmap_path = os.path.join(HERE, "yamnet_class_map.csv")
    yamnet_names = pd.read_csv(cmap_path)["display_name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(yamnet_names)}

    head_path = os.path.join(HERE, "finetuned_model", "head.keras")
    classes_path = os.path.join(HERE, "finetuned_model", "classes.txt")
    if not (os.path.exists(head_path) and os.path.exists(classes_path)):
        sys.exit("Missing finetuned_model\\head.keras or classes.txt.\n"
                 "Run fine_tune_yamnet.py first.")
    head = tf.keras.models.load_model(head_path)
    with open(classes_path) as f:
        ft_classes = [line.strip() for line in f if line.strip()]
    print(f"Fine-tuned classes: {ft_classes}\n")

    ffmpeg = RT.find_ffmpeg()
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="segtest_")

    print("Segmenting OVS videos into real events (same method as clean_audio.py)...")
    df = segment_ovs_videos(ffmpeg, set(ft_classes), tmp)
    if df.empty:
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit("No events detected. Check TOP_DB or your OVS folder.")
    print(f"\n{len(df)} segmented event clips total.\n")

    print("Scoring both models on the same segmented clips...")
    untrained_preds, finetuned_preds = [], []
    for path in df["clip"]:
        wav, _ = librosa.load(path, sr=16000, mono=True)
        scores, embeddings, _ = yamnet(wav.astype(np.float32))
        mean_scores = np.mean(scores.numpy(), axis=0)
        mean_emb = np.mean(embeddings.numpy(), axis=0, keepdims=True)

        untrained_preds.append(untrained_predict(mean_scores, name_to_idx))
        probs = head.predict(mean_emb, verbose=0)[0]
        finetuned_preds.append(ft_classes[int(np.argmax(probs))])

    shutil.rmtree(tmp, ignore_errors=True)
    df["untrained"] = untrained_preds
    df["finetuned"] = finetuned_preds

    all_classes = sorted(set(df["true_label"]))

    RT.hr()
    print("RESULTS ON SEGMENTED TEST EVENTS")
    RT.hr()
    f1_u = report("untrained YAMNet, segmented test", df["true_label"], df["untrained"], all_classes)
    f1_f = report("fine-tuned YAMNet, segmented test", df["true_label"], df["finetuned"], all_classes)

    out = os.path.join(HERE, "compare_segmented_results.csv")
    try:
        df.drop(columns=["clip"]).to_csv(out, index=False)
        print(f"Saved: {out}")
    except PermissionError:
        import time
        alt = os.path.join(HERE, f"compare_segmented_{time.strftime('%H%M%S')}.csv")
        df.drop(columns=["clip"]).to_csv(alt, index=False)
        print(f"{out} was open elsewhere, saved here instead: {alt}")

    RT.hr()
    print("COMPARE AGAINST YOUR EARLIER FIXED-WINDOW RESULTS")
    RT.hr()
    print(f"segmented test, fine-tuned macro F1: {f1_f:.3f}")
    print("earlier fixed-window results, for reference:")
    print("   23 raw files              : 0.698")
    print("   cleaned/split training     : 0.672")
    print("   + augmented (pen_clicking hurt): 0.647")
    print()
    print("If this number moves meaningfully from those, the fixed-window")
    print("test clip shape was part of what was driving the earlier results,")
    print("not just the training data changes.")


if __name__ == "__main__":
    main()
