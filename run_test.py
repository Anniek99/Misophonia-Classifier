"""
run_test.py  --  Score Teachable Machine and untrained YAMNet on the OVS test videos.

Also the shared helper module: other scripts import it for ffmpeg, the OVS folder
lookup, the video-to-class map and clip slicing.

Run:  python run_test.py
Needs tm_model/ (Teachable Machine export) and the OVS Videos folder. Optional
extra_audio/ measures false alarms.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ---- Settings ----

# Test videos and Teachable Machine export. The script also searches nearby folders.
OVS_FOLDER = "OVS Videos"
MODEL_FOLDER = "tm_model"

# Speech and background clips for measuring false alarms (any audio format):
#   extra_audio/speech/   people talking
#   extra_audio/other/    room noise, cafe, traffic
EXTRA_AUDIO_FOLDER = "extra_audio"

# Folder name -> what your Teachable Machine class might be called.
CLASS_ALIASES = {
    "other": ["other", "background", "background noise", "backgroundnoise",
              "background_noise", "noise", "ambient"],
    "speech": ["speech", "talking", "voice", "speaking"],
}

# Which SSV video contains which trigger sound.
# From Samermit et al. 2022 and Mahzouni et al. 2024 Table 2.
VIDEO_TO_SOUND = {
    "Vid01": "slurping",
    "Vid04": "crunchy_chewing",
    "Vid06": "squeaking",
    "Vid07": "swishing",
    "Vid08": "sniffling",
    "Vid09": "fork_scraping",
    "Vid10": "crunchy_chewing",
    "Vid12": "sniffling",
    "Vid13": "tapping",
    "Vid16": "drumming",
    "Vid17": "wet_chewing",
    "Vid18": "gulping",
    "Vid20": "pen_clicking",
}

# Maps YAMNet's 521 AudioSet labels onto sound names.
# Only sounds your model has a class for get scored.
YAMNET_MAP = {
    # Built from measurement: tune_yamnet.py recorded which AudioSet labels fire for each class.
    # Every name exists in yamnet_class_map.csv.
    "wet_chewing":   ["Liquid", "Drip", "Fill (with liquid)", "Squish",
                      "Chewing, mastication"],
    # AudioSet has no gulping class. These are weak relatives, so low scores are expected.
    "gulping":       ["Hiccup", "Burping, eructation", "Gargling"],
    "fork_scraping": ["Dishes, pots, and pans", "Cutlery, silverware",
                      "Chink, clink", "Glass"],
    "pen_clicking":  ["Finger snapping", "Hands", "Scissors", "Clapping"],
    "sniffling":     ["Breathing", "Snort", "Spray", "Hiss"],
    "speech":        ["Speech", "Child speech, kid speaking",
                      "Narration, monologue", "Conversation"],
    # extra sounds, only used if you add classes for them
    "crunchy_chewing": ["Crumpling, crinkling", "Crunch", "Chewing, mastication"],
    "slurping":        ["Gargling", "Liquid", "Stream"],
    "swishing":        ["Gargling", "Liquid", "Splash, splatter"],
    "squeaking":       ["Squeak", "Scrape"],
    "tapping":         ["Tap", "Tick", "Clicking"],
    "drumming":        ["Typing", "Computer keyboard", "Patter"],
}

CLIP_SECONDS = 1.0
CLIP_HOP = 0.5          # 0.5 = clips overlap by half, giving more test samples
YAMNET_THRESHOLD = 0.0194  # derived by tune_yamnet.py from measured scores
                         # tune_yamnet.py suggests a value from your audio.

# ============================================================================


def hr(char="="):
    """Print a divider line."""
    print(char * 78)


def find_ffmpeg():
    """Use system ffmpeg if present, otherwise the one bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit(
            "Could not find ffmpeg.\n"
            "Run this, then try again:\n"
            "    .venv\\Scripts\\pip install imageio-ffmpeg"
        )


def find_ovs_folder():
    """Look in the configured spot, then in obvious nearby places."""
    candidates = [
        OVS_FOLDER,
        os.path.join(os.path.expanduser("~"), "Desktop", "OVS Videos"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "OVS Videos"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OVS Videos"),
        "OVS Videos",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    sys.exit(
        f"Could not find the OVS Videos folder.\n"
        f"Looked in:\n  " + "\n  ".join(candidates) +
        f"\n\nOpen run_test.py and set OVS_FOLDER to the right path."
    )


def find_model_files():
    """Find the .tflite and labels.txt wherever they are."""
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in [os.path.join(here, MODEL_FOLDER), here,
                   os.path.join(here, "data", "tm_model")]:
        if not os.path.isdir(folder):
            continue
        tflite = [f for f in os.listdir(folder) if f.lower().endswith(".tflite")]
        labels = [f for f in os.listdir(folder) if f.lower() == "labels.txt"]
        if tflite and labels:
            return os.path.join(folder, tflite[0]), os.path.join(folder, labels[0])
    return None, None


def video_key(filename):
    """Vid17_O.MOV -> Vid17 . Handles .mov and .MOV and stray spaces."""
    m = re.match(r"(Vid\d+)", os.path.basename(filename), re.IGNORECASE)
    if not m:
        return None
    num = re.search(r"\d+", m.group(1)).group()
    return f"Vid{int(num):02d}"


def extract_clips(ffmpeg, video_path, out_dir, stem):
    """Pull the audio out of a video and cut it into 1-second WAV clips."""
    master = os.path.join(out_dir, f"{stem}_master.wav")
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", video_path, "-vn",
         "-af", "loudnorm=I=-23:LRA=7:TP=-2", "-ac", "1", "-ar", "44100", master],
        check=True,
    )

    import wave
    with wave.open(master) as w:
        duration = w.getnframes() / w.getframerate()

    clips = []
    start = 0.0
    i = 0
    while start + CLIP_SECONDS <= duration + 1e-6:
        clip = os.path.join(out_dir, f"{stem}_c{i:03d}.wav")
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{start:.3f}",
             "-t", str(CLIP_SECONDS), "-i", master,
             "-ac", "1", "-ar", "44100", clip],
            check=True,
        )
        clips.append(clip)
        start += CLIP_HOP
        i += 1

    os.remove(master)
    return clips, duration


def main():
    """Slice the OVS videos into 1 second clips, score YAMNet and Teachable Machine on them, and print the results."""
    hr()
    print("TheraPhono: Teachable Machine vs YAMNet on the OVS videos")
    hr()
    print()

    ffmpeg = find_ffmpeg()
    ovs = find_ovs_folder()
    model_path, labels_path = find_model_files()

    print(f"videos : {ovs}")
    print(f"model  : {model_path}")
    print(f"ffmpeg : {ffmpeg}")
    print()

    # ---------------------------------------------------------------- labels
    if not model_path:
        print("!! No .tflite plus labels.txt found. Running YAMNet only.")
        my_classes, tm_labels = [], []
    else:
        with open(labels_path) as f:
            tm_labels = [ln.strip().split(" ", 1)[-1] for ln in f if ln.strip()]
        my_classes = [c for c in tm_labels]
        print(f"Your model has {len(tm_labels)} classes:")
        for c in tm_labels:
            print(f"   - {c}")
        print()

    # Normalize class names so "Wet Chewing" matches "wet_chewing"
    def norm(s):
        return re.sub(r"[^a-z]", "", s.lower())

    canon = {}
    for lab in tm_labels:
        canon[norm(lab)] = lab

    def match_class(sound):
        """Find the Teachable Machine label for a sound name, allowing aliases."""
        if norm(sound) in canon:
            return canon[norm(sound)]
        for alt in CLASS_ALIASES.get(sound, []):
            if norm(alt) in canon:
                return canon[norm(alt)]
        return None

    sound_to_tm = {}
    for sound in set(VIDEO_TO_SOUND.values()):
        m = match_class(sound)
        if m:
            sound_to_tm[sound] = m

    # ---------------------------------------------------------------- videos
    videos = []
    for f in sorted(os.listdir(ovs)):
        if not f.lower().endswith((".mov", ".mp4", ".m4v", ".avi")):
            continue
        key = video_key(f)
        if key is None:
            continue
        videos.append((key, VIDEO_TO_SOUND.get(key), os.path.join(ovs, f)))

    scored = [v for v in videos if v[1] and (not tm_labels or v[1] in sound_to_tm)]
    unknown = [v for v in videos if not v[1]]
    no_class = [v for v in videos if v[1] and tm_labels and v[1] not in sound_to_tm]

    print(f"Found {len(videos)} videos.")
    if tm_labels:
        print(f"  {len(scored)} match a class in your model  -> scored")
    else:
        print(f"  {len(scored)} have a known sound  -> scored")
    if no_class:
        print(f"  {len(no_class)} are known sounds with no class in your model:")
        for k, s, _ in no_class:
            print(f"       {k}  ({s})")
    if unknown:
        print(f"  {len(unknown)} not in the mapping: "
              + ", ".join(k for k, _, _ in unknown))
    print()

    missing = [s for s in sound_to_tm
               if s not in {v[1] for v in scored}]
    if missing:
        print("!! WARNING: these classes have NO test video, so they cannot be scored:")
        for s in missing:
            need = [k for k, v in VIDEO_TO_SOUND.items() if v == s]
            print(f"       {s}   (need {' or '.join(need)} from osf.io/3ysfh)")
        print()

    # ------------------------------------------------- extra speech/background
    extra_files = []
    extra_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              EXTRA_AUDIO_FOLDER)
    AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma",
                 ".mp4", ".mov", ".aiff", ".aif")
    if os.path.isdir(extra_root):
        for sub in sorted(os.listdir(extra_root)):
            subdir = os.path.join(extra_root, sub)
            if not os.path.isdir(subdir):
                continue
            tm_name = match_class(sub) if tm_labels else sub
            if tm_labels and not tm_name:
                print(f"  extra_audio\\{sub}: no matching class in your model, skipped")
                continue
            if tm_name:
                sound_to_tm[sub] = tm_name
            files = [f for f in sorted(os.listdir(subdir))
                     if f.lower().endswith(AUDIO_EXT)]
            for f in files:
                extra_files.append((sub, os.path.join(subdir, f)))
            if files:
                print(f"  extra_audio\\{sub}: {len(files)} file(s)")
            else:
                print(f"  extra_audio\\{sub}: EMPTY, put audio files in it")
        print()
    else:
        print("  (no extra_audio folder, so false alarms cannot be measured)")
        print(f"     make it here: {extra_root}")
        print("     with subfolders 'speech' and 'other'")
        print()

    if not scored and not extra_files:
        print()
        hr("!")
        print("Nothing to score. Your Teachable Machine class names do not match")
        print("any of the sound names this script knows about.")
        print()
        print("Your classes:      " + ", ".join(tm_labels))
        print("Names it expects:  " + ", ".join(sorted(set(VIDEO_TO_SOUND.values()))))
        print()
        print("Two ways to fix it:")
        print("  1. Rename your classes in Teachable Machine and re-export, or")
        print("  2. Open run_test.py and edit VIDEO_TO_SOUND so the values match")
        print("     the names you actually used.")
        print()
        print("Capitalization and spaces do not matter. 'Wet Chewing' matches")
        print("'wet_chewing' already.")
        hr("!")
        sys.exit(1)

    # ---------------------------------------------------------------- clips
    tmp = tempfile.mkdtemp(prefix="theraphono_")
    print("Extracting audio and slicing into 1-second clips...")
    rows = []
    for key, sound, path in scored:
        clips, dur = extract_clips(ffmpeg, path, tmp, key)
        for c in clips:
            rows.append({"clip": c, "video": key, "true_sound": sound})
        print(f"   {key:<7} {sound:<16} {dur:5.1f}s -> {len(clips):3d} clips")

    for i, (sound, path) in enumerate(extra_files):
        stem = f"extra{i:02d}_" + re.sub(r"[^A-Za-z0-9]", "", os.path.splitext(
            os.path.basename(path))[0])[:24]
        try:
            clips, dur = extract_clips(ffmpeg, path, tmp, stem)
        except Exception as e:
            print(f"   could not read {os.path.basename(path)}: {e}")
            continue
        for c in clips:
            rows.append({"clip": c, "video": stem, "true_sound": sound})
        print(f"   {stem[:7]:<7} {sound:<16} {dur:5.1f}s -> {len(clips):3d} clips")

    print(f"\n{len(rows)} clips total\n")

    import librosa
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(rows)

    # ---------------------------------------------------------------- YAMNet
    have_yamnet = False
    yamnet_predict = None

    def _interpreter(path):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except Exception:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        it = Interpreter(model_path=path)
        it.allocate_tensors()
        return it

    here = os.path.dirname(os.path.abspath(__file__))
    local_model = os.path.join(here, "yamnet.tflite")
    local_map = os.path.join(here, "yamnet_class_map.csv")

    try:
        if os.path.exists(local_model) and os.path.exists(local_map):
            # Preferred path: a plain downloaded file, no tensorflow_hub.
            print("Loading YAMNet from yamnet.tflite ...")
            yit = _interpreter(local_model)
            yin = yit.get_input_details()[0]
            yout = yit.get_output_details()[0]
            need = int(yin["shape"][-1])
            ynames = pd.read_csv(local_map)["display_name"].tolist()
            yidx = {n: i for i, n in enumerate(ynames)}

            in_rank = len(yin["shape"])

            def yamnet_scores(path):
                wav, _ = librosa.load(path, sr=16000, mono=True)
                wav = wav.astype(np.float32)
                if len(wav) < need:
                    wav = np.pad(wav, (0, need - len(wav)))
                outs = []
                for st in range(0, max(1, len(wav) - need + 1), need):
                    chunk = wav[st:st + need]
                    if len(chunk) < need:
                        break
                    # YAMNet's tflite wants a flat [15600] tensor.
                    # Other builds want [1, 15600]. Match whatever it declared.
                    yit.set_tensor(yin["index"],
                                   chunk if in_rank == 1 else chunk.reshape(1, -1))
                    yit.invoke()
                    o = yit.get_tensor(yout["index"])
                    outs.append(o[0] if o.ndim == 2 else o)
                return np.mean(outs, axis=0) if outs else np.zeros(len(ynames))
        else:
            # Fallback: tensorflow_hub. Often fails on Windows.
            print("Loading YAMNet via tensorflow_hub ...")
            import tensorflow_hub as hub
            yam = None
            last = None
            for url in ("https://tfhub.dev/google/yamnet/1",
                        "https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1"):
                try:
                    yam = hub.load(url)
                    break
                except Exception as e:
                    last = e
            if yam is None:
                raise last
            ynames = pd.read_csv(yam.class_map_path().numpy().decode())["display_name"].tolist()
            yidx = {n: i for i, n in enumerate(ynames)}

            def yamnet_scores(path):
                wav, _ = librosa.load(path, sr=16000, mono=True)
                sc, _, _ = yam(wav.astype(np.float32))
                return np.mean(sc.numpy(), axis=0)

        active_map = {s2: [n for n in names if n in yidx]
                      for s2, names in YAMNET_MAP.items()
                      if s2 in set(df["true_sound"])}

        def yamnet_predict(path):
            mean = yamnet_scores(path)
            best, score = "other", 0.0
            for sound, names in active_map.items():
                if not names:
                    continue
                v = max(mean[yidx[n]] for n in names)
                if v > score:
                    best, score = sound, float(v)
            return ("other" if score < YAMNET_THRESHOLD else best)

        print("Running YAMNet...")
        df["yamnet"] = [yamnet_predict(p) for p in df["clip"]]
        have_yamnet = True

    except Exception as e:
        print()
        hr("!")
        print("Could not download YAMNet:")
        print(f"   {type(e).__name__}: {str(e)[:160]}")
        print()
        print("FIX: run this once, then run run_test.py again:")
        print("        .venv\\Scripts\\python get_yamnet.py")
        print()
        print("It downloads YAMNet as a plain file and skips tensorflow_hub,")
        print("which is usually what is broken.")
        print()
        print("Continuing with your Teachable Machine model only.")
        hr("!")
        print()

    # ------------------------------------------------- Teachable Machine
    have_tm = False
    if model_path:
        print("Running Teachable Machine...")
        try:
            # Newer TensorFlow moved the interpreter out to ai_edge_litert.
            try:
                from ai_edge_litert.interpreter import Interpreter
            except Exception:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
            interp = Interpreter(model_path=model_path)
            interp.allocate_tensors()
            inp = interp.get_input_details()
            outp = interp.get_output_details()
            shape = list(inp[0]["shape"])

            if len(shape) > 2:
                print()
                hr("!")
                print(f"Your model wants input shape {shape}, which is a spectrogram,")
                print("not raw audio. Feeding it would require reproducing Teachable")
                print("Machine's exact settings, which is not worth your time.")
                print("Write this shape down for your mentor. YAMNet results below.")
                hr("!")
                print()
            else:
                tm_need = int(shape[-1])
                tm_to_sound = {v: k for k, v in sound_to_tm.items()}

                def tm_predict(path):
                    wav, _ = librosa.load(path, sr=44100, mono=True)
                    wav = wav.astype(np.float32)
                    if len(wav) < tm_need:
                        wav = np.pad(wav, (0, tm_need - len(wav)))
                    interp.set_tensor(inp[0]["index"], wav[:tm_need].reshape(1, -1))
                    interp.invoke()
                    probs = interp.get_tensor(outp[0]["index"])[0]
                    lab = tm_labels[int(np.argmax(probs))]
                    return tm_to_sound.get(lab, lab)

                df["teachable"] = [tm_predict(p) for p in df["clip"]]
                have_tm = True
        except Exception as e:
            print(f"   Teachable Machine failed: {e}")
            print("   Continuing with YAMNet only.")
    print()

    models = []
    if have_yamnet:
        models.append(("YAMNet (untrained)", "yamnet"))
    if have_tm:
        models.append(("Teachable Machine", "teachable"))

    if not models:
        sys.exit("Neither model ran. Nothing to report. See the messages above.")

    # ============================================================ REPORT
    def pct(a, b):
        return f"{a/b*100:5.1f}%" if b else "    -"

    def metrics(col, cls):
        """precision, recall, f1 for one class under one model."""
        tp = int(((df[col] == cls) & (df["true_sound"] == cls)).sum())
        fp = int(((df[col] == cls) & (df["true_sound"] != cls)).sum())
        fn = int(((df[col] != cls) & (df["true_sound"] == cls)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return tp, fp, fn, prec, rec, f1

    present = [c for c in sorted(df["true_sound"].unique())]

    hr()
    print("1. OVERALL")
    hr()
    print(f"{'model':<24}{'correct':>9}{'total':>8}{'accuracy':>11}{'macro F1':>11}")
    print("-" * 78)
    for name, col in models:
        ok = int((df[col] == df["true_sound"]).sum())
        f1s = [metrics(col, c)[5] for c in present]
        macro = sum(f1s) / len(f1s) if f1s else 0
        print(f"{name:<24}{ok:>9}{len(df):>8}{pct(ok, len(df)):>11}{macro:>11.3f}")
    print()
    print("Macro F1 averages performance across all classes equally, so a model")
    print("cannot look good just by nailing the biggest class. It is the number")
    print("to report, not accuracy.")
    print()

    hr()
    print("2. PER CLASS, FULL METRICS")
    hr()
    print("recall    = of all the real X, how many did it catch")
    print("precision = when it says X, how often is it right")
    print("F1        = the two combined")
    print()
    for name, col in models:
        print(f"--- {name} ---")
        print(f"{'class':<16}{'clips':>7}{'TP':>5}{'FP':>5}{'FN':>5}"
              f"{'recall':>9}{'precis':>9}{'F1':>8}")
        print("-" * 78)
        for c in present:
            tp, fp, fn, pr, rc, f1 = metrics(col, c)
            n = int((df["true_sound"] == c).sum())
            print(f"{c:<16}{n:>7}{tp:>5}{fp:>5}{fn:>5}"
                  f"{rc*100:>8.1f}%{pr*100:>8.1f}%{f1:>8.3f}")
        print()

    hr()
    print("3. SIDE BY SIDE, RECALL PER CLASS")
    hr()
    header = f"{'class':<16}{'clips':>7}"
    for name, _ in models:
        header += f"{name[:18]:>21}"
    header += "   better"
    print(header)
    print("-" * 78)
    for c in present:
        n = int((df["true_sound"] == c).sum())
        line = f"{c:<16}{n:>7}"
        vals = []
        for _, col in models:
            ok = int(((df[col] == c) & (df["true_sound"] == c)).sum())
            vals.append(ok / n if n else 0)
            line += f"{ok:>12} ({ok/n*100:4.0f}%)" if n else f"{'-':>21}"
        if len(vals) == 2:
            if abs(vals[0] - vals[1]) < 0.02:
                line += "   tie"
            else:
                line += f"   {models[0][0].split()[0] if vals[0] > vals[1] else models[1][0].split()[0]}"
        print(line)
    print()

    hr()
    print("4. PER VIDEO")
    hr()
    header = f"{'video':<26}{'class':<16}{'clips':>7}"
    for name, _ in models:
        header += f"{name.split()[0][:10]:>12}"
    print(header)
    print("-" * 78)
    for key in sorted(df["video"].unique()):
        sub = df[df["video"] == key]
        c = sub["true_sound"].iloc[0]
        line = f"{key[:25]:<26}{c:<16}{len(sub):>7}"
        for _, col in models:
            ok = int((sub[col] == c).sum())
            line += f"{ok/len(sub)*100:>11.0f}%"
        print(line)
    print()

    # ---------------------------------------------------------------- table 4
    for name, col in models:
        hr()
        print(f"6. WHAT {name.upper()} GUESSED WRONG")
        hr()
        bad = df[df[col] != df["true_sound"]]
        if bad.empty:
            print("Nothing wrong.\n")
            continue
        counts = defaultdict(int)
        for _, r in bad.iterrows():
            counts[(r["true_sound"], r[col])] += 1
        print(f"{'true sound':<20}{'guessed':<20}{'times':>7}")
        print("-" * 78)
        for (t, g), n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"{t:<20}{g:<20}{n:>7}")
        print()

    # ---------------------------------------------------------- extra videos
    if no_class:
        hr()
        print("7. SOUNDS WITH NO CLASS IN YOUR MODEL")
        hr()
        print("Not scored, but worth seeing what the models call them.")
        print()
        extra_rows = []
        for key, sound, path in no_class:
            clips, _ = extract_clips(ffmpeg, path, tmp, key)
            for c in clips:
                extra_rows.append({"clip": c, "video": key, "true_sound": sound})
        if extra_rows:
            ex = pd.DataFrame(extra_rows)
            if have_yamnet:
                ex["yamnet"] = [yamnet_predict(p) for p in ex["clip"]]
            if have_tm:
                ex["teachable"] = [tm_predict(p) for p in ex["clip"]]
            for key in sorted(ex["video"].unique()):
                sub = ex[ex["video"] == key]
                sound = sub["true_sound"].iloc[0]
                line = f"{key:<9}{sound:<18}"
                for _, col in models:
                    top = sub[col].value_counts().idxmax()
                    pct = sub[col].value_counts().max() / len(sub) * 100
                    line += f"{top:>18} ({pct:3.0f}%)"
                print(line)
            print()

    # ---------------------------------------------------------------- save
    here2 = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here2, "results.csv")
    try:
        df.drop(columns=["clip"]).to_csv(out, index=False)
        print(f"Per-clip results saved to: {out}")
    except PermissionError:
        import time
        alt = os.path.join(here2, f"results_{time.strftime('%H%M%S')}.csv")
        df.drop(columns=["clip"]).to_csv(alt, index=False)
        print(f"results.csv is open in Excel, so saved here instead:")
        print(f"   {alt}")

    shutil.rmtree(tmp, ignore_errors=True)

    hr()
    print("HOW TO READ THIS")
    hr()
    print("Look at PER SOUND, not just the overall accuracy. A model can score")
    print("well overall and still be useless at one specific trigger.")
    print()
    print("The WRONG GUESSES table tells you which kind of problem you have:")
    print("  guessed 'other'          -> the model is not detecting the sound")
    print("  guessed a different sound -> it detects something but confuses it")
    print()
    print("These are only a few videos per sound, so treat this as a smoke test,")
    print("not a final accuracy number. Say that out loud to your mentor.")
    hr()


if __name__ == "__main__":
    main()
