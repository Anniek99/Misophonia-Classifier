"""
fine_tune_yamnet.py  --  Train a small classifier on frozen YAMNet embeddings.

Each audio file is cut into ~1 second pieces, YAMNet turns each piece into 1024
numbers, and a small new classifier learns the 7 classes from them (transfer
learning). The result is scored on the OVS test videos.

Run:
    python fine_tune_yamnet.py                       one run, fixed seed
    python fine_tune_yamnet.py --seeds 5             five seeds, mean and std
    python fine_tune_yamnet.py --debug-windows fork_scraping
                                                     save one file's windows to listen to

Needs training_data/<class>/, extra_audio/speech and extra_audio/other, and the
OVS Videos folder. Saves finetuned_model/. With --seeds it also writes
seed_summary.csv and seed_results.csv.
"""

import argparse
import os
import random
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TFHUB_CACHE_DIR", os.path.join(HERE, "tfhub_cache"))

import numpy as np

import run_test as RT   # reuses find_ffmpeg, find_ovs_folder, VIDEO_TO_SOUND, etc.

CLASSES = ["wet_chewing", "gulping", "fork_scraping", "pen_clicking",
           "sniffling", "speech", "other"]

TRAINING_DATA_FOLDER = "training_data"
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4",
            ".mov", ".m4v", ".aiff", ".aif")

MIN_SOURCES_PER_CLASS = 3   # cannot honestly split train/val below this
VAL_FRACTION = 0.2
EPOCHS = 30
BATCH_SIZE = 32
SEED = 42

# Std across seeds: 0 = population std (numpy default, used for the reported numbers),
# 1 = sample std (about 12% larger with 5 seeds). State which one you use.
STD_DDOF = 0

# YAMNet's own framing: ~0.96 s windows, a new one every ~0.48 s (only used by --debug-windows).
YAMNET_WINDOW_SEC = 0.96
YAMNET_HOP_SEC = 0.48
YAMNET_SR = 16000

MODEL_URLS = [
    "https://tfhub.dev/google/yamnet/1",
    "https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1",
]


# ------------------------------------------------------------------ seeding

def set_all_seeds(seed):
    """Seed Python, numpy and TensorFlow so the same data gives the same results."""
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ------------------------------------------------------------------ loading

def load_yamnet():
    """Load the full YAMNet (with embeddings): a local folder first, then tfhub.dev, then Kaggle."""
    local = os.path.join(HERE, "yamnet_full_model")
    if os.path.isdir(local) and os.listdir(local):
        print(f"Loading YAMNet from local folder: {local}")
        import tensorflow as tf
        return tf.saved_model.load(local)

    import tensorflow_hub as hub
    last_err = None
    for url in MODEL_URLS:
        try:
            print(f"Loading YAMNet from {url} ...")
            print("(cached after first success, in tfhub_cache/)")
            return hub.load(url)
        except Exception as e:
            last_err = e
            print(f"   failed: {type(e).__name__}: {str(e)[:120]}")

    RT.hr("!")
    print("Could not download the full YAMNet model.")
    print()
    print("This is a different file from yamnet.tflite. That one only has")
    print("class scores. This script needs embeddings too, which only come")
    print("from the full model via tensorflow_hub.")
    print()
    print("If your network keeps blocking this (common on school wifi):")
    print("  1. In a browser, open:")
    print("       https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed")
    print("  2. That downloads a .tar.gz. Extract it.")
    print(f"  3. Put the extracted contents in:")
    print(f"       {local}\\")
    print("     You should see a file called saved_model.pb directly inside")
    print("     that folder, not in a subfolder.")
    print("  4. Run this script again.")
    RT.hr("!")
    sys.exit(1)


# --------------------------------------------------------------- embeddings

def find_files(folder):
    """List the audio files in a folder."""
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(AUDIO_EXT)]


def audio_to_16k(ffmpeg, path, tmp_wav):
    """Convert an audio or video file to 16 kHz mono wav with even loudness (YAMNet needs 16 kHz)."""
    import subprocess
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", path, "-vn",
         "-af", "loudnorm=I=-23:LRA=7:TP=-2", "-ac", "1", "-ar", "16000", tmp_wav],
        check=True)


def gather_class_sources():
    """{class_name: [file paths]}, pulling extra_audio for speech/other."""
    sources = {c: find_files(os.path.join(HERE, TRAINING_DATA_FOLDER, c))
               for c in CLASSES}
    sources["speech"] += find_files(os.path.join(HERE, "extra_audio", "speech"))
    sources["other"] += find_files(os.path.join(HERE, "extra_audio", "other"))
    return sources


def save_debug_windows(wav, n_rows, cls, stem, out_dir):
    """Save each ~1 second window of one file as a wav, plus a manifest of times and labels."""
    import soundfile as sf
    os.makedirs(out_dir, exist_ok=True)
    hop = int(YAMNET_HOP_SEC * YAMNET_SR)
    win = int(YAMNET_WINDOW_SEC * YAMNET_SR)
    manifest_lines = ["index\tstart_sec\tend_sec\tground_truth_label"]
    for i in range(n_rows):
        start = i * hop
        end = min(start + win, len(wav))
        if start >= len(wav):
            break
        chunk = wav[start:end]
        fname = f"{stem}_win{i:02d}_{cls}.wav"
        sf.write(os.path.join(out_dir, fname), chunk, YAMNET_SR)
        manifest_lines.append(f"{i}\t{start/YAMNET_SR:.3f}\t{end/YAMNET_SR:.3f}\t{cls}")
    with open(os.path.join(out_dir, f"{stem}_manifest.txt"), "w") as f:
        f.write("\n".join(manifest_lines))
    return len(manifest_lines) - 1


def extract_dataset(yamnet, ffmpeg, sources, tmp_dir, debug_class=None, debug_out=None):
    """Run every file through YAMNet. Returns embeddings, labels and one source id per file."""
    import librosa

    X, y, src = [], [], []
    tmp_wav = os.path.join(tmp_dir, "extract_tmp.wav")
    debug_done = False

    # Every piece from a file gets that file's class label.
    for cls in CLASSES:
        files = sources[cls]
        if not files:
            continue
        print(f"  {cls:<16} {len(files)} file(s)")
        for i, path in enumerate(files):
            try:
                audio_to_16k(ffmpeg, path, tmp_wav)
                wav, _ = librosa.load(tmp_wav, sr=16000, mono=True)
            except Exception as e:
                print(f"      skipped {os.path.basename(path)}: {e}")
                continue
            if len(wav) < 1000:
                continue
            _, embeddings, _ = yamnet(wav.astype(np.float32))
            emb = embeddings.numpy()
            sid = f"{cls}__{i:03d}"
            X.append(emb)
            y.extend([cls] * len(emb))
            src.extend([sid] * len(emb))

            if debug_class == cls and not debug_done:
                stem = os.path.splitext(os.path.basename(path))[0]
                n_saved = save_debug_windows(wav, len(emb), cls, stem, debug_out)
                print(f"      [debug] saved {n_saved} window(s) from this file to "
                     f"{debug_out}\\  (all labeled '{cls}', listen to window 00 "
                     f"and the last window especially)")
                debug_done = True

    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)

    if not X:
        return None, None, None
    return np.concatenate(X, axis=0), np.array(y), np.array(src)


# -------------------------------------------------------------------- split

def split_by_source(y, src, val_fraction, seed):
    """Train/val split. Whole source files move together, never their frames."""
    rng = np.random.default_rng(seed)
    train_mask = np.zeros(len(y), dtype=bool)

    for cls in np.unique(y):
        cls_sources = sorted(set(src[y == cls]))
        if len(cls_sources) < MIN_SOURCES_PER_CLASS:
            print(f"  WARNING: {cls} has only {len(cls_sources)} source file(s), "
                 f"putting all of it in training. Add more files to split this class.")
            train_mask |= (y == cls)
            continue
        n_val = max(1, round(len(cls_sources) * val_fraction))
        val_sources = set(rng.choice(cls_sources, size=n_val, replace=False))
        is_val = np.isin(src, list(val_sources)) & (y == cls)
        train_mask |= (y == cls) & ~is_val

    return train_mask, ~train_mask


# -------------------------------------------------------------------- model

def build_head(n_classes):
    """The small classifier: 1024 embedding numbers in, one score per class out."""
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,), name="yamnet_embedding"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, name="logits"),
    ], name="finetuned_head")
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])
    return model


def class_weights_for(y_idx, n_classes):
    """Weight rare classes more, so the model doesn't just favor the biggest class."""
    counts = np.bincount(y_idx, minlength=n_classes)
    total = counts.sum()
    weights = {}
    for i in range(n_classes):
        weights[i] = total / (n_classes * counts[i]) if counts[i] > 0 else 0.0
    return weights


# ---------------------------------------------------------------- reporting

def score(y_true, y_pred, classes):
    """Per-class recall, precision and F1, plus accuracy and macro F1, returned as numbers."""
    present = [c for c in classes if c in set(y_true)]
    f1s, recs, precs, ns = {}, {}, {}, {}
    for c in present:
        ns[c] = int((y_true == c).sum())
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s[c] = f1
        recs[c] = rec
        precs[c] = prec
    acc = float((y_true == y_pred).mean())
    macro = sum(f1s.values()) / len(f1s) if f1s else 0.0
    return {"per_class_f1": f1s, "per_class_recall": recs, "per_class_precision": precs,
            "per_class_n": ns, "accuracy": acc, "macro_f1": macro}


def report(name, y_true, y_pred, classes):
    """Print recall, precision and F1 for each class, plus accuracy and macro F1."""
    print(f"--- {name} ---")
    present = [c for c in classes if c in set(y_true)]
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
    print(f"accuracy {acc*100:.1f}%   macro F1 {sum(f1s)/len(f1s):.3f}")
    print()


def summarize_seeds(all_results, classes_order):
    """Mean and std of recall, precision and F1 per class across seeds, plus macro F1 and accuracy."""
    classes_seen = [c for c in classes_order
                    if any(c in r["per_class_f1"] for r in all_results)]
    rows = []
    for c in classes_seen:
        f1 = np.array([r["per_class_f1"][c] for r in all_results if c in r["per_class_f1"]])
        rec = np.array([r["per_class_recall"][c] for r in all_results if c in r["per_class_recall"]])
        prec = np.array([r["per_class_precision"][c] for r in all_results if c in r["per_class_precision"]])
        n = next(r["per_class_n"][c] for r in all_results if c in r["per_class_n"])
        rows.append({
            "class": c, "n_test_pieces": n,
            "mean_recall": rec.mean(), "std_recall": rec.std(ddof=STD_DDOF),
            "mean_precision": prec.mean(), "std_precision": prec.std(ddof=STD_DDOF),
            "mean_f1": f1.mean(), "std_f1": f1.std(ddof=STD_DDOF),
            "min_f1": f1.min(), "max_f1": f1.max(),
        })
    macro = np.array([r["macro_f1"] for r in all_results])
    acc = np.array([r["accuracy"] for r in all_results])
    overall = {
        "mean_macro_f1": macro.mean(), "std_macro_f1": macro.std(ddof=STD_DDOF),
        "min_macro_f1": macro.min(), "max_macro_f1": macro.max(),
        "mean_accuracy": acc.mean(), "std_accuracy": acc.std(ddof=STD_DDOF),
        "min_accuracy": acc.min(), "max_accuracy": acc.max(),
    }
    return rows, overall


def print_seed_summary(all_results, classes_order):
    """Print the mean and std table across seeds."""
    rows, overall = summarize_seeds(all_results, classes_order)
    std_name = "population std (ddof=0)" if STD_DDOF == 0 else "sample std (ddof=1)"

    RT.hr()
    print(f"SUMMARY ACROSS {len(all_results)} SEEDS   (mean and {std_name})")
    RT.hr()
    header = (f"{'class':<15}{'n':>5}"
             f"{'recall':>9}{'std':>6}"
             f"{'precision':>11}{'std':>6}"
             f"{'F1':>8}{'std':>7}{'min':>7}{'max':>7}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['class']:<15}{r['n_test_pieces']:>5}"
             f"{r['mean_recall']*100:>8.0f}%{r['std_recall']*100:>5.0f}%"
             f"{r['mean_precision']*100:>10.0f}%{r['std_precision']*100:>5.0f}%"
             f"{r['mean_f1']:>8.3f}{r['std_f1']:>7.3f}"
             f"{r['min_f1']:>7.3f}{r['max_f1']:>7.3f}")
    print("-" * len(header))
    print(f"{'macro F1':<15}{'':>5}{'':>9}{'':>6}{'':>11}{'':>6}"
         f"{overall['mean_macro_f1']:>8.3f}{overall['std_macro_f1']:>7.3f}"
         f"{overall['min_macro_f1']:>7.3f}{overall['max_macro_f1']:>7.3f}")
    print(f"accuracy: mean {overall['mean_accuracy']*100:.1f}%   "
         f"std {overall['std_accuracy']*100:.1f}%   "
         f"range {overall['min_accuracy']*100:.1f}% to {overall['max_accuracy']*100:.1f}%")
    print()
    print("Recall std and precision std are in percentage points. The std next")
    print("to F1 is the one that matches a single STD column on a slide.")
    print()
    print("n is how many one-second test pieces each class has. It is the same")
    print("for every seed, because the test videos never change.")
    print()
    print("Mean F1 will not exactly equal 2*P*R/(P+R) from the mean P and mean R")
    print("shown here. Each seed's F1 is computed first, then those are averaged.")
    print()
    print("A small std means one run is representative. A large std means one")
    print("run alone could be lucky or unlucky, so report the mean.")
    print()


def write_seed_csvs(all_results, classes_order, out_dir):
    """Save the summary (seed_summary.csv) and every seed's numbers (seed_results.csv)."""
    import csv
    rows, overall = summarize_seeds(all_results, classes_order)

    summary_path = os.path.join(out_dir, "seed_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "n_test_pieces",
                    "mean_recall_pct", "std_recall_pct",
                    "mean_precision_pct", "std_precision_pct",
                    "mean_f1", "std_f1", "min_f1", "max_f1"])
        for r in rows:
            w.writerow([r["class"], r["n_test_pieces"],
                        f"{r['mean_recall']*100:.1f}", f"{r['std_recall']*100:.1f}",
                        f"{r['mean_precision']*100:.1f}", f"{r['std_precision']*100:.1f}",
                        f"{r['mean_f1']:.4f}", f"{r['std_f1']:.4f}",
                        f"{r['min_f1']:.4f}", f"{r['max_f1']:.4f}"])
        w.writerow(["macro_f1", "", "", "", "", "",
                    f"{overall['mean_macro_f1']:.4f}", f"{overall['std_macro_f1']:.4f}",
                    f"{overall['min_macro_f1']:.4f}", f"{overall['max_macro_f1']:.4f}"])
        w.writerow(["std_type", "population (ddof=0)" if STD_DDOF == 0 else "sample (ddof=1)"])

    per_seed_path = os.path.join(out_dir, "seed_results.csv")
    with open(per_seed_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed_value", "class", "n_test_pieces",
                    "recall_pct", "precision_pct", "f1"])
        for i, res in enumerate(all_results):
            for c in classes_order:
                if c in res["per_class_f1"]:
                    w.writerow([SEED + i, c, res["per_class_n"][c],
                                f"{res['per_class_recall'][c]*100:.1f}",
                                f"{res['per_class_precision'][c]*100:.1f}",
                                f"{res['per_class_f1'][c]:.4f}"])
            w.writerow([SEED + i, "macro_f1", "", "", "", f"{res['macro_f1']:.4f}"])
    return summary_path, per_seed_path


def print_dataset_counts(sources, y, src, classes):
    """How much data each class has: source files, and one-second pieces
    (embedding rows). Printed once so it can go in a samples-per-class table."""
    RT.hr()
    print("DATA PER CLASS (training pool)")
    RT.hr()
    header = f"{'class':<16}{'files':>7}{'pieces':>9}"
    print(header)
    print("-" * len(header))
    for c in classes:
        n_files = len(sources.get(c, []))
        n_rows = int((y == c).sum())
        print(f"{c:<16}{n_files:>7}{n_rows:>9}")
    print("-" * len(header))
    print(f"{'total':<16}{sum(len(sources.get(c, [])) for c in classes):>7}{len(y):>9}")
    print()
    print("A piece is about one second of audio. Every piece takes the label of")
    print("the folder its file came from.")
    print()


# ------------------------------------------------------- one train/eval run

def train_and_evaluate_once(X, y, src, present_classes, seed, ffmpeg, yamnet,
                            ovs_folder, verbose_report=True):
    """Split, train, validate and run the OVS test for one seed. Returns the model, classes and test scores."""
    import tensorflow as tf

    # Fix the seeds, then split train and validation by whole file, so pieces of one file never land on both sides.
    set_all_seeds(seed)

    train_mask, val_mask = split_by_source(y, src, VAL_FRACTION, seed)
    if verbose_report:
        print(f"train rows: {train_mask.sum()}   val rows: {val_mask.sum()}\n")

    cls_to_idx = {c: i for i, c in enumerate(present_classes)}
    y_idx = np.array([cls_to_idx[c] for c in y])

    X_train, y_train = X[train_mask], y_idx[train_mask]
    X_val, y_val = X[val_mask], y_idx[val_mask]

    if len(X_train) == 0:
        sys.exit("Every row ended up in validation, none in training. "
                 "Something is wrong with the split; check your data.")

    have_val = len(X_val) > 0

    # Give rare classes more weight so the biggest class doesn't win by default.
    weights = class_weights_for(y_train, len(present_classes))
    if verbose_report:
        print("Class weights (higher = rarer, matters more per mistake):")
        for c, i in cls_to_idx.items():
            print(f"   {c:<16} {weights[i]:.2f}")
        print()

    # Build the classifier and train it.
    model = build_head(len(present_classes))

    if verbose_report:
        print("Training...")
    if have_val:
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            class_weight=weights, callbacks=[callback],
            verbose=2 if verbose_report else 0)
    else:
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            class_weight=weights, callbacks=[callback],
            verbose=2 if verbose_report else 0)
    if verbose_report:
        print()

    if have_val and verbose_report:
        val_pred_idx = np.argmax(model.predict(X_val, verbose=0), axis=1)
        val_pred = np.array([present_classes[i] for i in val_pred_idx])
        val_true = np.array([present_classes[i] for i in y_val])
        RT.hr()
        print("VALIDATION (held out from the same training pool)")
        RT.hr()
        report("fine-tuned YAMNet, validation split", val_true, val_pred, present_classes)

    # Score on the OVS test videos, which the model never saw.
    real_test_score = None
    if ovs_folder:
        if verbose_report:
            RT.hr()
            print("REAL TEST: the OVS videos, which this model never saw")
            RT.hr()
        import tempfile, shutil, librosa
        tmp2 = tempfile.mkdtemp(prefix="finetune_test_")
        test_rows = []
        for f in sorted(os.listdir(ovs_folder)):
            if not f.lower().endswith((".mov", ".mp4", ".m4v", ".avi")):
                continue
            key = RT.video_key(f)
            sound = RT.VIDEO_TO_SOUND.get(key)
            # Skip videos whose sound isn't one of the trained classes.
            if sound not in present_classes:
                continue
            wav_path = os.path.join(tmp2, f"{key}.wav")
            audio_to_16k(ffmpeg, os.path.join(ovs_folder, f), wav_path)
            wav, _ = librosa.load(wav_path, sr=16000, mono=True)
            # YAMNet gives one embedding per ~1 second of audio. The classifier labels each one.
            _, emb, _ = yamnet(wav.astype(np.float32))
            preds = np.argmax(model.predict(emb.numpy(), verbose=0), axis=1)
            for p in preds:
                test_rows.append({"true": sound, "pred": present_classes[p]})
        shutil.rmtree(tmp2, ignore_errors=True)

        if test_rows:
            t_true = np.array([r["true"] for r in test_rows])
            t_pred = np.array([r["pred"] for r in test_rows])
            if verbose_report:
                report("fine-tuned YAMNet, OVS test videos", t_true, t_pred, present_classes)
                print("Note: this evaluates using YAMNet's own native ~0.96s framing,")
                print("not the 1 second / 0.5 hop slicing run_test.py uses. Trends are")
                print("comparable, exact clip counts will not match run_test.py's CSV.")
                print()
            # Keep the numbers (not just printed text) so several seeds can be averaged.
            real_test_score = score(t_true, t_pred, present_classes)
        elif verbose_report:
            print("No matching OVS videos found for the classes you trained.\n")

    return model, present_classes, real_test_score


# -------------------------------------------------------------------- main

def main():
    """Get YAMNet embeddings once, train (one seed or several), score on the OVS videos, and save the model."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1,
                    help="train+evaluate this many times, with different seeds, "
                         "and report mean/std across runs instead of one score")
    ap.add_argument("--debug-windows", dest="debug_class", default=None,
                    help="save the actual audio windows (and their ground-truth "
                         "label) for the first file in this class, e.g. "
                         "fork_scraping, so you can listen to what the model "
                         "actually saw for each guess")
    args = ap.parse_args()

    RT.hr()
    print("Fine-tuning YAMNet: extract embeddings, train a small classifier")
    RT.hr()
    print()

    # 1. Load YAMNet.
    yamnet = load_yamnet()
    print("YAMNet loaded.\n")

    # 2. Find the training files, one folder per class.
    sources = gather_class_sources()
    total_files = sum(len(v) for v in sources.values())
    empty = [c for c, v in sources.items() if not v]
    if empty:
        print("Classes with NO training audio yet:")
        for c in empty:
            print(f"   {c}  ->  put files in training_data\\{c}\\")
        print()
    if total_files == 0:
        sys.exit("No training audio found anywhere. Populate training_data\\ first.")

    print(f"{total_files} source file(s) across {len([c for c in sources if sources[c]])} classes.\n")

    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="finetune_")
    ffmpeg = RT.find_ffmpeg()

    debug_out = os.path.join(HERE, "debug_windows") if args.debug_class else None
    if args.debug_class:
        print(f"Debug mode: will save example windows from the first "
             f"{args.debug_class} file to debug_windows\\\n")

    # 3. Turn every file into YAMNet embeddings. This is the slow step, and it only runs once.
    print("Extracting embeddings (this is the slow part, be patient)...")
    X, y, src = extract_dataset(yamnet, ffmpeg, sources, tmp,
                                debug_class=args.debug_class, debug_out=debug_out)
    shutil.rmtree(tmp, ignore_errors=True)

    if X is None:
        sys.exit("Nothing usable was extracted. Check your audio files play correctly.")

    present_classes = [c for c in CLASSES if c in set(y)]
    print(f"\n{len(X)} embedding rows (~1 second each) from {total_files} files.\n")
    print_dataset_counts(sources, y, src, present_classes)

    # 4. Find the OVS test videos. Without them there is no real test.
    ovs = None
    try:
        ovs = RT.find_ovs_folder()
    except SystemExit:
        pass

    # ----------------------------------------------------- single-seed path
    if args.seeds == 1:
        model, classes_used, _ = train_and_evaluate_once(
            X, y, src, present_classes, SEED, ffmpeg, yamnet, ovs,
            verbose_report=True)

        out_dir = os.path.join(HERE, "finetuned_model")
        os.makedirs(out_dir, exist_ok=True)
        model.save(os.path.join(out_dir, "head.keras"))
        with open(os.path.join(out_dir, "classes.txt"), "w") as f:
            f.write("\n".join(classes_used))

        RT.hr()
        print("SAVED")
        RT.hr()
        print(f"  {out_dir}\\head.keras")
        print(f"  {out_dir}\\classes.txt")
        print()
        print("This is the trained classifier head only. YAMNet itself was never")
        print("changed, only downloaded and used as a fixed feature extractor.")
        print()
        print(f"This run used a fixed seed ({SEED}), so rerunning with the same")
        print("training_data folder will reproduce this exact result. Run with")
        print("--seeds 5 to see how much the score actually varies run to run.")
        return

    # ------------------------------------------------------ multi-seed path
    print(f"Running {args.seeds} seeds: train + evaluate once per seed, using")
    print("the SAME embeddings extracted above each time (extraction itself")
    print("is not random, so there is no reason to repeat it).\n")

    all_results = []
    last_model = None
    for i in range(args.seeds):
        # Each run uses the next seed after SEED.
        this_seed = SEED + i
        RT.hr()
        print(f"SEED {i+1}/{args.seeds}  (seed value = {this_seed})")
        RT.hr()
        model, classes_used, real_score = train_and_evaluate_once(
            X, y, src, present_classes, this_seed, ffmpeg, yamnet, ovs,
            verbose_report=False)
        last_model = model
        if real_score is None:
            sys.exit("No OVS test folder found. --seeds needs the real test "
                     "to compare across runs; a validation-only comparison "
                     "would be comparing each run against its own easier check.")
        all_results.append(real_score)
        print(f"  macro F1 this seed: {real_score['macro_f1']:.3f}   "
             f"accuracy: {real_score['accuracy']*100:.1f}%")
        print()

    # Mean and std across the seeds. This is the result to report.
    print_seed_summary(all_results, present_classes)

    summary_csv, per_seed_csv = write_seed_csvs(all_results, present_classes, HERE)
    print("Saved the numbers so you don't have to retype them:")
    print(f"  {summary_csv}   (the table above)")
    print(f"  {per_seed_csv}   (every seed, every class)")
    print()

    # Save the last run's model. It's one example run, not the best one.
    out_dir = os.path.join(HERE, "finetuned_model")
    os.makedirs(out_dir, exist_ok=True)
    last_model.save(os.path.join(out_dir, "head.keras"))
    with open(os.path.join(out_dir, "classes.txt"), "w") as f:
        f.write("\n".join(present_classes))
    print(f"Saved the LAST seed's model to {out_dir}\\head.keras")
    print("(the summary above is what to report; the saved model is just one")
    print("representative run, picked for convenience, not because it's best)")


if __name__ == "__main__":
    main()
