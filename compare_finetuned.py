"""
compare_finetuned.py  --  Untrained YAMNet vs fine-tuned YAMNet, same exact clips.

WHY THIS SCRIPT EXISTS

Your last two evaluations weren't a clean comparison. run_test.py slices audio
into fixed 1-second clips with a 0.5 second hop. fine_tune_yamnet.py let YAMNet
frame the audio on its own internal 0.96s/0.48s schedule instead. Same videos,
different number of windows pulled out of them: 151 clips one way, 164 the
other. Any difference in the numbers was partly the models and partly the
slicing. This script fixes that by slicing every clip exactly once, the same
way, and running BOTH models over that identical set.

It also fixes a smaller inconsistency: the untrained arm used to load a
separate yamnet.tflite file, while the fine-tuned arm loaded YAMNet a different
way through tensorflow_hub. Technically the same model, but two different
exports of it. This script loads YAMNet ONCE. A single call to it returns both
the 521 class scores (for the untrained arm) and the 1024-number embedding (for
the fine-tuned arm), so there's no question that any difference between the two
comes from the training, not from which copy of YAMNet answered.

RUN
    .venv\\Scripts\\python compare_finetuned.py

NEEDS
    yamnet_full_model\\   (or working tensorflow_hub download)
    finetuned_model\\head.keras and classes.txt   (from fine_tune_yamnet.py)
    yamnet_class_map.csv
    the OVS videos, and extra_audio\\speech and extra_audio\\other for the
    false alarm measurement
"""

import os
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TFHUB_CACHE_DIR", os.path.join(HERE, "tfhub_cache"))

import numpy as np
import pandas as pd

import run_test as RT              # ffmpeg, OVS folder, video map, clip slicing, YAMNET_MAP
import fine_tune_yamnet as FT      # load_yamnet() with embeddings

HOP = 0.5   # matches run_test.py's existing methodology, so this stays
            # comparable with every number already reported so far


# ---------------------------------------------------------------- gathering

def gather_clips(ffmpeg, tmp_dir, allowed_classes):
    """Slice OVS videos + extra_audio into 1-second clips, ONE way, for both
    models. Returns a DataFrame: clip path, true label, source id.

    Only includes sounds the fine-tuned model actually has a class for.
    Your OVS folder has 13 different trigger sounds, but the fine-tuned model
    only has 5, plus speech and other. If a video's sound isn't in the model's
    class list, the model can never predict it correctly no matter how good it
    is, so including it would drag the score down for a reason that has
    nothing to do with model quality. The untrained arm is scored on the exact
    same restricted set, so both sides face the same, fair test."""
    rows = []

    ovs = None
    try:
        ovs = RT.find_ovs_folder()
    except SystemExit:
        pass

    if ovs:
        for f in sorted(os.listdir(ovs)):
            if not f.lower().endswith((".mov", ".mp4", ".m4v", ".avi")):
                continue
            key = RT.video_key(f)
            sound = RT.VIDEO_TO_SOUND.get(key)
            if sound not in allowed_classes:
                continue
            clips, _ = RT.extract_clips(ffmpeg, os.path.join(ovs, f), tmp_dir, key)
            for c in clips:
                rows.append({"clip": c, "true_label": sound, "source": key})

    for sub in ("speech", "other"):
        if sub not in allowed_classes:
            continue
        d = os.path.join(HERE, RT.EXTRA_AUDIO_FOLDER, sub)
        if not os.path.isdir(d):
            continue
        for i, f in enumerate(sorted(os.listdir(d))):
            if not f.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
                continue
            stem = f"extra_{sub}{i}"
            try:
                clips, _ = RT.extract_clips(ffmpeg, os.path.join(d, f), tmp_dir, stem)
            except Exception as e:
                print(f"   could not read {f}: {e}")
                continue
            for c in clips:
                rows.append({"clip": c, "true_label": sub, "source": stem})

    return pd.DataFrame(rows)


# ----------------------------------------------------------------- decision

def untrained_predict(mean_scores, name_to_idx):
    """Identical logic to run_test.py's own untrained-YAMNet decision rule,
    reusing the same YAMNET_MAP and YAMNET_THRESHOLD so this arm is not a new
    or different version of that model, just re-run on matched clips."""
    best, score = "other", 0.0
    for cls, names in RT.YAMNET_MAP.items():
        valid = [n for n in names if n in name_to_idx]
        if not valid:
            continue
        s = max(mean_scores[name_to_idx[n]] for n in valid)
        if s > score:
            best, score = cls, float(s)
    return "other" if score < RT.YAMNET_THRESHOLD else best


# ------------------------------------------------------------------ metrics

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
    return acc, sum(f1s) / len(f1s)


def false_alarm_rate(df, pred_col, triggers):
    neg = df[df["true_label"].isin(["speech", "other"])]
    if len(neg) == 0:
        return None, 0
    fired = neg[pred_col].isin(triggers).mean()
    return fired, len(neg)


def mcnemar(df, col_a, col_b, name_a, name_b):
    a_ok = df[col_a] == df["true_label"]
    b_ok = df[col_b] == df["true_label"]
    both = int((a_ok & b_ok).sum())
    only_a = int((a_ok & ~b_ok).sum())
    only_b = int((~a_ok & b_ok).sum())
    neither = int((~a_ok & ~b_ok).sum())

    print(f"--- McNemar: {name_a} vs {name_b} ---")
    print(f"  both right               : {both}")
    print(f"  {name_a} right, {name_b} wrong : {only_a}")
    print(f"  {name_a} wrong, {name_b} right : {only_b}")
    print(f"  both wrong               : {neither}")
    try:
        from statsmodels.stats.contingency_tables import mcnemar as mc
        result = mc([[both, only_a], [only_b, neither]], exact=True)
        print(f"  p-value = {result.pvalue:.4f}")
        if result.pvalue > 0.05:
            print("  p > 0.05: cannot claim either model is better on this test set.")
        else:
            better = name_a if only_a > only_b else name_b
            print(f"  p < 0.05: {better} is genuinely better here, not just luck.")
    except ImportError:
        print("  (install statsmodels for a p-value: .venv\\Scripts\\pip install statsmodels)")
    print()


# -------------------------------------------------------------------- main

def main():
    RT.hr()
    print("Controlled comparison: untrained vs fine-tuned YAMNet")
    print("Same clips, same windowing, same YAMNet load, for both.")
    RT.hr()
    print()

    import tensorflow as tf
    import librosa

    yamnet = FT.load_yamnet()
    print("YAMNet loaded.\n")

    cmap_path = os.path.join(HERE, "yamnet_class_map.csv")
    if not os.path.exists(cmap_path):
        sys.exit("Missing yamnet_class_map.csv. Run get_yamnet.py first.")
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
    print(f"Fine-tuned head loaded, {len(ft_classes)} classes: {ft_classes}\n")

    ffmpeg = RT.find_ffmpeg()
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="compare_")

    print("Slicing every source into 1 second clips, one shared way...")
    print(f"Restricted to the {len(ft_classes)} classes the fine-tuned model "
         f"actually has: {ft_classes}")
    df = gather_clips(ffmpeg, tmp, set(ft_classes))
    if df.empty:
        sys.exit("No clips gathered. Check the OVS folder and extra_audio.")
    print(f"{len(df)} clips from {df['source'].nunique()} source files.")
    print(df["true_label"].value_counts().to_string())
    print()

    print("Running both models over every clip (one YAMNet call each)...")
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

    all_classes = sorted(set(df["true_label"]) | set(df["untrained"]) | set(df["finetuned"]))
    triggers = [c for c in all_classes if c not in ("speech", "other")]

    RT.hr()
    print("RESULTS, both models on the identical clip set")
    RT.hr()
    report("untrained YAMNet", df["true_label"], df["untrained"], all_classes)
    report("fine-tuned YAMNet", df["true_label"], df["finetuned"], all_classes)

    RT.hr()
    print("FALSE ALARMS")
    RT.hr()
    for name, col in [("untrained", "untrained"), ("fine-tuned", "finetuned")]:
        rate, n = false_alarm_rate(df, col, triggers)
        if rate is None:
            print(f"{name:<12} no speech/other clips found, cannot measure")
        else:
            print(f"{name:<12} fires on {rate*100:5.1f}% of {n} non-trigger clips "
                 f"(~{rate*60:.1f}/min)")
    print()

    RT.hr()
    print("IS THE DIFFERENCE REAL?")
    RT.hr()
    mcnemar(df, "untrained", "finetuned", "untrained", "finetuned")

    out = os.path.join(HERE, "compare_finetuned_results.csv")
    try:
        df.drop(columns=["clip"]).to_csv(out, index=False)
        print(f"Saved: {out}")
    except PermissionError:
        import time
        alt = os.path.join(HERE, f"compare_finetuned_{time.strftime('%H%M%S')}.csv")
        df.drop(columns=["clip"]).to_csv(alt, index=False)
        print(f"{out} was open in Excel, saved here instead: {alt}")


if __name__ == "__main__":
    main()
