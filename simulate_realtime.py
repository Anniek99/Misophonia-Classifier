"""
simulate_realtime.py  --  Replay audio as a stream and measure detection delay and false alarms.

Runs the trained classifier over continuous audio. Reports how long after a
trigger starts the display would update, and how often it fires when it should
stay quiet. This runs on a laptop. Inference time on the Raspberry Pi has to be
measured on the Pi.

Run:
    python simulate_realtime.py
    python simulate_realtime.py --sweep      try several hop and agreement settings
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from collections import deque

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TFHUB_CACHE_DIR", os.path.join(HERE, "tfhub_cache"))

import numpy as np

import run_test as RT
import fine_tune_yamnet as FT

WINDOW_SEC = 0.975   # matches YAMNet's own embedding frame. Not a free parameter.
SAMPLE_RATE = 16000


def load_stream(ffmpeg, path, tmp):
    """Load a video or audio file as one continuous 16 kHz waveform."""
    out = os.path.join(tmp, "stream.wav")
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", path, "-vn",
         "-af", "loudnorm=I=-23:LRA=7:TP=-2", "-ac", "1", "-ar", str(SAMPLE_RATE), out],
        check=True)
    import librosa
    wav, _ = librosa.load(out, sr=SAMPLE_RATE, mono=True)
    os.remove(out)
    return wav.astype(np.float32)


def run_stream(wav, predict_fn, hop_sec, n_agree, hold_sec):
    """Slide a window over the audio like the device would. A label shows when the last few predictions agree, then stays for the hold time."""
    win = int(WINDOW_SEC * SAMPLE_RATE)
    hop = int(hop_sec * SAMPLE_RATE)

    # The last few predictions. A label shows only when they all agree.
    recent = deque(maxlen=n_agree)
    events = []
    displayed = None
    display_until = -1.0
    infer_times = []

    pos = win
    while pos <= len(wav):
        t_now = pos / SAMPLE_RATE
        chunk = wav[pos - win:pos]

        # Time each prediction, to see how heavy it is.
        t0 = time.perf_counter()
        pred = predict_fn(chunk)
        infer_times.append(time.perf_counter() - t0)

        recent.append(pred)

        # Show a new label if the last predictions agree, it isn't 'other', and the hold time is over.
        if t_now >= display_until and len(recent) == n_agree \
                and len(set(recent)) == 1 and recent[0] != "other":
            if recent[0] != displayed:
                events.append({"time": t_now, "label": recent[0]})
                displayed = recent[0]
                display_until = t_now + hold_sec

        pos += hop

    return events, infer_times


def main():
    """Replay the test audio as a stream, for one setting or a sweep, and print delay and false alarms."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop", type=float, default=0.25)
    ap.add_argument("--agree", type=int, default=2)
    ap.add_argument("--hold", type=float, default=3.0)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    # Load YAMNet and the trained classifier.
    yamnet = FT.load_yamnet()
    print("YAMNet loaded.")

    import tensorflow as tf
    head_path = os.path.join(HERE, "finetuned_model", "head.keras")
    classes_path = os.path.join(HERE, "finetuned_model", "classes.txt")
    if not (os.path.exists(head_path) and os.path.exists(classes_path)):
        sys.exit("Missing finetuned_model\\head.keras or classes.txt.\n"
                 "Run fine_tune_yamnet.py first.")
    head = tf.keras.models.load_model(head_path)
    with open(classes_path) as f:
        classes = [line.strip() for line in f if line.strip()]
    print(f"Fine-tuned head loaded, classes: {classes}\n")

    # Label one chunk of audio: YAMNet first, then the trained classifier.
    def predict(chunk):
        _, embeddings, _ = yamnet(chunk)
        emb = np.mean(embeddings.numpy(), axis=0, keepdims=True)
        probs = head.predict(emb, verbose=0)[0]
        return classes[int(np.argmax(probs))]

    ffmpeg = RT.find_ffmpeg()
    ovs = RT.find_ovs_folder()
    tmp = tempfile.mkdtemp(prefix="sim_")

    # Test audio: the OVS videos, plus quiet background to count false alarms.
    clips = []
    for f in sorted(os.listdir(ovs)):
        if not f.lower().endswith((".mov", ".mp4", ".avi")):
            continue
        key = RT.video_key(f)
        sound = RT.VIDEO_TO_SOUND.get(key)
        if sound and sound in classes:
            clips.append((key, sound, os.path.join(ovs, f)))

    quiet_dir = os.path.join(HERE, "extra_audio", "other")
    if os.path.isdir(quiet_dir):
        for f in sorted(os.listdir(quiet_dir)):
            if f.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
                clips.append(("QUIET_" + f[:12], None, os.path.join(quiet_dir, f)))

    if not clips:
        sys.exit("No audio found.")

    print("Loading audio as continuous streams...")
    streams = []
    for key, sound, path in clips:
        wav = load_stream(ffmpeg, path, tmp)
        streams.append((key, sound, wav, len(wav) / SAMPLE_RATE))
        print(f"   {key:<22} {str(sound):<15} {len(wav)/SAMPLE_RATE:6.1f}s")
    print()

    # One setting, or with --sweep a grid of hop and agreement values.
    configs = ([(h, a) for h in (0.5, 0.25, 0.1) for a in (1, 2, 3)]
               if args.sweep else [(args.hop, args.agree)])

    for hop, agree in configs:
        RT.hr()
        print(f"hop = {hop}s   agree = {agree} windows   hold = {args.hold}s")
        RT.hr()
        print(f"{'clip':<22}{'expect':<15}{'latency':>10}{'fires':>7}{'FA/min':>9}")
        print("-" * 78)

        lat, fa, infer_all = [], [], []
        for key, sound, wav, dur in streams:
            ev, times = run_stream(wav, predict, hop, agree, args.hold)
            infer_all += times

            hits = [e for e in ev if e["label"] == sound] if sound else []
            l = (hits[0]["time"] - WINDOW_SEC) if hits else None
            wrong = [e for e in ev if e["label"] != sound]
            rate = len(wrong) / (dur / 60)
            fa.append(rate)
            if l is not None:
                lat.append(l)

            latstr = f"{l:8.2f}s" if l is not None else ("   never" if sound else "       -")
            print(f"{key:<22}{str(sound):<15}{latstr:>10}{len(ev):>7}{rate:>9.1f}")

        ms = np.array(infer_all) * 1000
        print("-" * 78)
        print(f"median latency  : {np.median(lat):.2f}s" if lat else
              "median latency  : nothing detected")
        print(f"false alarms    : {np.mean(fa):.1f} per minute (mean)")
        print(f"inference time  : {np.median(ms):.1f}ms median, "
              f"{np.percentile(ms,95):.1f}ms p95   (ON THIS MACHINE, not the Pi)")
        print(f"CPU duty cycle  : {np.median(ms)/1000/hop*100:.0f}% of one core")
        print()

    shutil.rmtree(tmp, ignore_errors=True)

    RT.hr()
    print("READING THIS")
    RT.hr()
    print("latency  : your study tests 0, 500 and 1000ms delay. Whatever delay")
    print("           still shows a PAVS benefit is your real budget, not a")
    print("           number borrowed from someone else's hardware.")
    print()
    print("FA/min   : more than about 1 per minute during quiet is unwearable.")
    print("           Raising 'agree' cuts this at the cost of one hop of latency.")
    print()
    print("duty     : share of one CPU core, measured here, not on the Pi.")
    print("           Run this same script on the Pi before committing to numbers.")
    RT.hr()


if __name__ == "__main__":
    main()
