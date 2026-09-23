"""
fine_tune_yamnet.py  --  Fine-tune YAMNet on your own trigger sounds.

WHAT THIS IS

Follows the official TensorFlow transfer learning tutorial:
https://www.tensorflow.org/tutorials/audio/transfer_learning_audio

YAMNet is frozen. You never change its weights. Instead:
  1. Feed each of your audio files through YAMNet.
  2. Take its 1024-number "embedding" for each ~1 second frame, instead of its
     521-class prediction. The embedding is YAMNet's internal understanding of
     the sound, before it gets squeezed down to AudioSet's specific labels.
  3. Train a small new classifier on those embeddings to recognize YOUR seven
     classes.

This is different from what run_test.py does. run_test.py asks frozen YAMNet
"which of your 521 labels is this," then translates the answer with a hand-built
map. This script asks YAMNet "what do you hear," in its own 1024-number
language, and trains something new to interpret that answer directly.

WHY THIS NEEDS A DIFFERENT DOWNLOAD

Your yamnet.tflite only outputs 521 class scores. Embeddings are not in it.
This script needs the FULL YAMNet, which has three outputs: scores, embeddings,
and a spectrogram. That version only comes from tensorflow_hub, so this script
downloads it separately and caches it in tfhub_cache/ so it only downloads once.

WHAT TO SET UP FIRST

Create this folder structure and put audio in it:

    training_data/
      wet_chewing/      (your FOAMS + Freesound wet chewing clips)
      gulping/
      fork_scraping/
      pen_clicking/
      sniffling/

speech and other are pulled automatically from extra_audio/speech and
extra_audio/other, since you already have audio there.

Any format works: wav, mp3, m4a, whatever ffmpeg can read. Files do not need to
be pre-sliced into 1 second clips. Feed it whole recordings; YAMNet slices them
internally.

RUN
    .venv\\Scripts\\python fine_tune_yamnet.py
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

MODEL_URLS = [
    "https://tfhub.dev/google/yamnet/1",
    "https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1",
]


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


def extract_dataset(yamnet, ffmpeg, sources, tmp_dir):
    """Run every file through YAMNet. Returns embeddings, labels, source ids.

    Each file produces several embedding rows, one per ~0.96s internal frame.
    source_id is the file itself, and every row keeps track of which file it
    came from. That is what makes an honest train/val split possible: whole
    files move together, never individual frames.
    """
    import librosa

    X, y, src = [], [], []
    tmp_wav = os.path.join(tmp_dir, "extract_tmp.wav")

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

    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)

    if not X:
        return None, None, None
    return np.concatenate(X, axis=0), np.array(y), np.array(src)


# -------------------------------------------------------------------- split

def split_by_source(y, src, val_fraction, seed=42):
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


# -------------------------------------------------------------------- main

def main():
    RT.hr()
    print("Fine-tuning YAMNet: extract embeddings, train a small classifier")
    RT.hr()
    print()

    import tensorflow as tf

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

    print("Extracting embeddings (this is the slow part, be patient)...")
    X, y, src = extract_dataset(yamnet, ffmpeg, sources, tmp)
    shutil.rmtree(tmp, ignore_errors=True)

    if X is None:
        sys.exit("Nothing usable was extracted. Check your audio files play correctly.")

    present_classes = [c for c in CLASSES if c in set(y)]
    print(f"\n{len(X)} embedding rows (~1 second each) from {total_files} files.\n")

    # ------------------------------------------------------------- split
    train_mask, val_mask = split_by_source(y, src, VAL_FRACTION)
    print(f"train rows: {train_mask.sum()}   val rows: {val_mask.sum()}\n")

    cls_to_idx = {c: i for i, c in enumerate(present_classes)}
    y_idx = np.array([cls_to_idx[c] for c in y])

    X_train, y_train = X[train_mask], y_idx[train_mask]
    X_val, y_val = X[val_mask], y_idx[val_mask]

    if len(X_train) == 0:
        sys.exit("Every row ended up in validation, none in training. "
                 "Something is wrong with the split; check your data.")

    have_val = len(X_val) > 0
    if not have_val:
        print("=" * 66)
        print("No validation data this run.")
        print("Every class had too few source files (fewer than 3) to hold any")
        print("out, so everything went into training instead. This is exactly")
        print("what the WARNING lines above were telling you.")
        print()
        print("This is not a crash and not fatal: the model still trains fine,")
        print("you just won't get a validation table below. Add 2-3 more files")
        print("to your smallest classes and rerun to get one.")
        print("=" * 66)
        print()

    # ------------------------------------------------------------- train
    weights = class_weights_for(y_train, len(present_classes))
    print("Class weights (higher = rarer, matters more per mistake):")
    for c, i in cls_to_idx.items():
        print(f"   {c:<16} {weights[i]:.2f}")
    print()

    model = build_head(len(present_classes))

    print("Training...")
    if have_val:
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            class_weight=weights, callbacks=[callback], verbose=2)
    else:
        # No val_loss to monitor without a validation set, so watch training
        # loss instead. Less reliable at knowing when to stop, but it will
        # not crash, and the model still comes out usable.
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            class_weight=weights, callbacks=[callback], verbose=2)
    print()

    if have_val:
        val_pred_idx = np.argmax(model.predict(X_val, verbose=0), axis=1)
        val_pred = np.array([present_classes[i] for i in val_pred_idx])
        val_true = np.array([present_classes[i] for i in y_val])

        RT.hr()
        print("VALIDATION (held out from the same training pool)")
        RT.hr()
        report("fine-tuned YAMNet, validation split", val_true, val_pred, present_classes)
    else:
        RT.hr()
        print("VALIDATION: skipped, no held-out data this run (see note above)")
        RT.hr()
        print()

    # ---------------------------------------------------- real held-out test
    ovs = None
    try:
        ovs = RT.find_ovs_folder()
    except SystemExit:
        pass

    if ovs:
        RT.hr()
        print("REAL TEST: the OVS videos, which this model never saw")
        RT.hr()
        tmp2 = tempfile.mkdtemp(prefix="finetune_test_")
        test_rows = []
        for f in sorted(os.listdir(ovs)):
            if not f.lower().endswith((".mov", ".mp4", ".m4v", ".avi")):
                continue
            key = RT.video_key(f)
            sound = RT.VIDEO_TO_SOUND.get(key)
            if sound not in present_classes:
                continue
            wav_path = os.path.join(tmp2, f"{key}.wav")
            audio_to_16k(ffmpeg, os.path.join(ovs, f), wav_path)
            import librosa
            wav, _ = librosa.load(wav_path, sr=16000, mono=True)
            _, emb, _ = yamnet(wav.astype(np.float32))
            preds = np.argmax(model.predict(emb.numpy(), verbose=0), axis=1)
            for p in preds:
                test_rows.append({"true": sound, "pred": present_classes[p]})
        shutil.rmtree(tmp2, ignore_errors=True)

        if test_rows:
            t_true = np.array([r["true"] for r in test_rows])
            t_pred = np.array([r["pred"] for r in test_rows])
            report("fine-tuned YAMNet, OVS test videos", t_true, t_pred, present_classes)
            print("Note: this evaluates using YAMNet's own native ~0.96s framing,")
            print("not the 1 second / 0.5 hop slicing run_test.py uses. Trends are")
            print("comparable, exact clip counts will not match run_test.py's CSV.")
            print()
        else:
            print("No matching OVS videos found for the classes you trained.\n")

    # ---------------------------------------------------------------- save
    out_dir = os.path.join(HERE, "finetuned_model")
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "head.keras"))
    with open(os.path.join(out_dir, "classes.txt"), "w") as f:
        f.write("\n".join(present_classes))

    RT.hr()
    print("SAVED")
    RT.hr()
    print(f"  {out_dir}\\head.keras")
    print(f"  {out_dir}\\classes.txt")
    print()
    print("This is the trained classifier head only. YAMNet itself was never")
    print("changed, only downloaded and used as a fixed feature extractor.")


if __name__ == "__main__":
    main()
