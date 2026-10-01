
REPRODUCIBILITY

Every source of randomness (Python's random module, numpy, and TensorFlow's
weight initialization / dropout / shuffling) is seeded with SEED below. Same
training_data folder in, same numbers out, every time you run this.

The train/val SPLIT was already deterministic before this update (it had its
own seed=42). What was NOT seeded before was everything inside training itself:
how the classifier's weights start, which dropout connections get switched off
each round, and what order batches get shuffled in. Those are now seeded too.

This does not make two DIFFERENT training_data folders produce the same score.
It makes the SAME folder produce the same score every time, which is what
"reproducible" means here.

RUN
    .venv\\Scripts\\python fine_tune_yamnet.py
    .venv\\Scripts\\python fine_tune_yamnet.py --seeds 5
        Runs the whole train+evaluate cycle 5 times, once per seed (the
        embeddings are only extracted once since that part is not random).
        Prints mean and standard deviation of macro F1 and per-class F1
        across the 5 runs, which is the actual sensitivity of the model to
        the random starting point, not just one lucky or unlucky run.
    .venv\\Scripts\\python fine_tune_yamnet.py --debug-windows fork_scraping
        Saves the actual ~1 second audio windows used for the FIRST source
        file in that class as individual short wav files, each paired with
        the ground-truth label it was assigned. This makes the "is silence
        getting labeled as the trigger" question something you can listen
        to directly, instead of reasoning about it secondhand.
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

# YAMNet's own documented framing: ~0.96s windows, a new one every ~0.48s.
# Used only by --debug-windows, to reconstruct approximately which audio
# samples produced each embedding row (YAMNet does not hand these back to
# us directly, it only returns the embeddings).
YAMNET_WINDOW_SEC = 0.96
YAMNET_HOP_SEC = 0.48
YAMNET_SR = 16000

MODEL_URLS = [
    "https://tfhub.dev/google/yamnet/1",
    "https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1",
]


# ------------------------------------------------------------------ seeding

def set_all_seeds(seed):
    """Seed every source of randomness this script touches. Call this before
    building or training the model (does not need to happen before the
    embedding extraction step, since YAMNet itself is frozen and
    deterministic; nothing there is affected by the seed)."""
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ------------------------------------------------------------------ loading

def load_yamnet():
    """Full YAMNet with embeddings. Not the same file as yamnet.tflite.

    Tries, in order:
      1. A local folder you downloaded by hand (see the error message below)
      2. tfhub.dev
      3. the Kaggle mirror
    """
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
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(AUDIO_EXT)]


def audio_to_16k(ffmpeg, path, tmp_wav):
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
    """Slice the already-loaded 16kHz waveform into the same approximate
    windows YAMNet used to produce n_rows embeddings, and save each one as
    its own short wav file, named with its index and the ground-truth label
    every row from this file receives. Also writes a small manifest.txt
    listing each window's start/end time and its label, so you can see at a
    glance, for example, that window 00 of a fork_scraping file is silence
    but is still labeled fork_scraping, the same way every other window
    from that file is, because labeling happens per FILE, not per window."""
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
    """Run every file through YAMNet. Returns embeddings, labels, source ids.

    Each file produces several embedding rows, one per ~0.96s internal frame.
    source_id is the file itself, and every row keeps track of which file it
    came from. That is what makes an honest train/val split possible: whole
    files move together, never individual frames.

    If debug_class is set, the FIRST file in that class also gets its windows
    saved as individual audio files via save_debug_windows, once.
    """
    import librosa

    X, y, src = [], [], []
    tmp_wav = os.path.join(tmp_dir, "extract_tmp.wav")
    debug_done = False

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
    """Balanced weights, since 'other' will usually have far more frames
    than a scarce class like gulping. Without this the model just learns to
    predict whichever class is biggest."""
    counts = np.bincount(y_idx, minlength=n_classes)
    total = counts.sum()
    weights = {}
    for i in range(n_classes):
        weights[i] = total / (n_classes * counts[i]) if counts[i] > 0 else 0.0
    return weights


# ---------------------------------------------------------------- reporting

def score(y_true, y_pred, classes):
    """Same math as report() below, but returns numbers instead of printing,
    so multi-seed runs can collect and later average them."""
    present = [c for c in classes if c in set(y_true)]
    f1s = {}
    for c in present:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s[c] = f1
    acc = float((y_true == y_pred).mean())
    macro = sum(f1s.values()) / len(f1s) if f1s else 0.0
    return {"per_class_f1": f1s, "accuracy": acc, "macro_f1": macro}


def report(name, y_true, y_pred, classes):
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


def print_seed_summary(all_results, classes_order):
    """Mean +/- standard deviation across N seeds, per class and overall.
    This is the actual answer to 'why do results vary when rerunning the
    code': they vary because training itself has random starting points.
    This shows HOW MUCH they vary, instead of just one run's number."""
    RT.hr()
    print(f"SUMMARY ACROSS {len(all_results)} SEEDS (mean +/- std)")
    RT.hr()
    classes_seen = [c for c in classes_order
                    if any(c in r["per_class_f1"] for r in all_results)]
    header = f"{'class':<16}{'mean F1':>10}{'std':>8}{'min':>8}{'max':>8}"
    print(header)
    print("-" * len(header))
    for c in classes_seen:
        vals = [r["per_class_f1"][c] for r in all_results if c in r["per_class_f1"]]
        if not vals:
            continue
        print(f"{c:<16}{np.mean(vals):>10.3f}{np.std(vals):>8.3f}"
             f"{np.min(vals):>8.3f}{np.max(vals):>8.3f}")
    macro_vals = [r["macro_f1"] for r in all_results]
    acc_vals = [r["accuracy"] for r in all_results]
    print("-" * len(header))
    print(f"{'macro F1':<16}{np.mean(macro_vals):>10.3f}{np.std(macro_vals):>8.3f}"
         f"{np.min(macro_vals):>8.3f}{np.max(macro_vals):>8.3f}")
    print(f"accuracy: mean {np.mean(acc_vals)*100:.1f}%  "
         f"std {np.std(acc_vals)*100:.1f}%  "
         f"range {np.min(acc_vals)*100:.1f}% to {np.max(acc_vals)*100:.1f}%")
    print()
    print("A small std means the score barely depends on the random starting")
    print("point, so one run is representative. A large std means one run")
    print("alone could be misleadingly lucky or unlucky, and the mean here")
    print("is the more honest number to report.")
    print()


# ------------------------------------------------------- one train/eval run

def train_and_evaluate_once(X, y, src, present_classes, seed, ffmpeg, yamnet,
                            ovs_folder, verbose_report=True):
    """Everything that depends on the random seed: split, build, train,
    validate, and run the real OVS test. Embeddings (X, y, src) are passed
    in already extracted, since extraction itself is not random and should
    not be repeated per seed. Returns the real-test score dict, or None if
    there was no OVS folder / no matching videos."""
    import tensorflow as tf

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

    weights = class_weights_for(y_train, len(present_classes))
    if verbose_report:
        print("Class weights (higher = rarer, matters more per mistake):")
        for c, i in cls_to_idx.items():
            print(f"   {c:<16} {weights[i]:.2f}")
        print()

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
            if sound not in present_classes:
                continue
            wav_path = os.path.join(tmp2, f"{key}.wav")
            audio_to_16k(ffmpeg, os.path.join(ovs_folder, f), wav_path)
            wav, _ = librosa.load(wav_path, sr=16000, mono=True)
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
            real_test_score = score(t_true, t_pred, present_classes)
        elif verbose_report:
            print("No matching OVS videos found for the classes you trained.\n")

    return model, present_classes, real_test_score


# -------------------------------------------------------------------- main

def main():
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

    yamnet = load_yamnet()
    print("YAMNet loaded.\n")

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

    print("Extracting embeddings (this is the slow part, be patient)...")
    X, y, src = extract_dataset(yamnet, ffmpeg, sources, tmp,
                                debug_class=args.debug_class, debug_out=debug_out)
    shutil.rmtree(tmp, ignore_errors=True)

    if X is None:
        sys.exit("Nothing usable was extracted. Check your audio files play correctly.")

    present_classes = [c for c in CLASSES if c in set(y)]
    print(f"\n{len(X)} embedding rows (~1 second each) from {total_files} files.\n")

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

    print_seed_summary(all_results, present_classes)

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
