"""
augment_scaper.py  --  Generate new training clips from the ones you have.

WHAT THIS DOES

For every real event clip you have, this makes several new synthetic versions
of it: pitch shifted up or down slightly, sped up or slowed down slightly,
mixed against a randomly chosen background at a randomized loudness. Same
underlying sound, genuinely different audio. This is the standard way to
stretch a small dataset further without recording anything new.

Uses Scaper (Salamon et al.), a real academic tool built for exactly this,
used in DCASE sound-event-detection challenges.
https://scaper.readthedocs.io/en/latest/tutorial.html

WHY EACH OUTPUT CLIP IS SHORT

A tempting design is: generate a long soundscape (several seconds of
background) with the event placed somewhere inside it. Don't do that here.

fine_tune_yamnet.py labels every ~1 second frame of a training file with that
file's class. If a generated clip were mostly background with a short event
buried in it, most of its frames would be background, but they'd all get
labeled as the trigger. That's label noise, and it would work against you.

So each output file here is sized to be just the event plus a small pad on
each side, roughly what you'd get from clean_audio.py. The background mixing
and pitch/time perturbation still happen, so the augmentation is real, but
almost every frame in the output genuinely contains the sound.

SETUP

1. Install the SoX command line tool (a separate program, like ffmpeg).

   Try this first:
       winget install --id ChrisBagwell.SoX -e

   Close and reopen your terminal, then check:
       sox --version

   If that prints a version number, you're done. If PowerShell says it can't
   find sox, this specific winget package has a known bug where it doesn't
   register on PATH. Fall back to this instead:

   a. Download: https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2-win32.zip/download
   b. Extract the zip somewhere permanent, e.g. C:\\Tools\\sox
   c. Add that folder to PATH: search "environment variables" in the Start
      menu, edit Path under your user variables, add the folder.
   d. Close and reopen your terminal. Check sox --version again.

2. Install Python packages:
       .venv\\Scripts\\pip install scaper jams pyloudnorm

3. Set up two folders:

       augment_source\\<class>\\      real event clips, ideally already run
                                      through clean_audio.py
       augment_backgrounds\\<any_name>\\   a few different background
                                      recordings. Needs at least one
                                      subfolder, same shape as
                                      augment_source. Copy your
                                      extra_audio\\other file into
                                      augment_backgrounds\\room\\ to start,
                                      and add more subfolders with different
                                      backgrounds if you have them. More
                                      variety here helps more than anything
                                      else in this script.

RUN
    .venv\\Scripts\\python augment_scaper.py --per-class 20
"""

import argparse
import os
import random
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

# scaper 1.6.5 uses the pre-NumPy-2.0 alias np.Inf, removed in NumPy 2.0+.
# This shim is contained to this script and does not touch your installed
# NumPy version or anything else in your environment.
if not hasattr(np, "Inf"):
    np.Inf = np.inf

SOURCE_FOLDER = "augment_source"
BG_FOLDER = "augment_backgrounds"
OUT_FOLDER = "training_data_augmented"

AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg")


def stage_as_wav(ffmpeg, src_root, staged_root):
    """Copy every audio file under src_root into staged_root as WAV.

    Scaper reads audio through soundfile/libsndfile, which only understands
    WAV, FLAC, and a few others. It cannot open .m4a, .mp3, or .mov directly,
    and fails with a cryptic 'Format not recognised' error deep inside a
    generate() call, the same failure no matter which file triggers it. Every
    other script in this project runs files through ffmpeg first for exactly
    this reason. This does the same, into a throwaway staging copy, so the
    files you actually keep (your real training_data and extra_audio) are
    never touched or converted in place.
    """
    import subprocess

    if not os.path.isdir(src_root):
        return None

    for cls in os.listdir(src_root):
        src_dir = os.path.join(src_root, cls)
        if not os.path.isdir(src_dir):
            continue
        out_dir = os.path.join(staged_root, cls)
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            if not f.lower().endswith(AUDIO_EXT):
                continue
            stem = os.path.splitext(f)[0]
            out_path = os.path.join(out_dir, f"{stem}.wav")
            if f.lower().endswith(".wav"):
                shutil.copy(os.path.join(src_dir, f), out_path)
            else:
                subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error", "-i",
                     os.path.join(src_dir, f), "-ac", "1", "-ar", "44100", out_path],
                    check=True)
    return staged_root


def find_class_folders(root):
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                 if os.path.isdir(os.path.join(root, d))
                 and any(f.lower().endswith(AUDIO_EXT)
                        for f in os.listdir(os.path.join(root, d))))


def clip_duration(path):
    import soundfile as sf
    info = sf.info(path)
    return info.frames / info.samplerate


def generate_for_class(scaper_mod, cls, source_root, src_dir, bg_dir, out_dir, n,
                       pad, snr_range, pitch_range, time_range, seed_base):
    os.makedirs(out_dir, exist_ok=True)
    src_files = [f for f in os.listdir(src_dir) if f.lower().endswith(AUDIO_EXT)]
    if not src_files:
        return 0, "no source clips"

    durations = [clip_duration(os.path.join(src_dir, f)) for f in src_files]
    max_stretch = time_range[1] if time_range else 1.0

    made = 0
    for i in range(n):
        # Pick the source file OURSELVES instead of letting Scaper's
        # source_file=('choose', []) pick it internally. This is the fix:
        # every clip now gets sized around the ACTUAL duration of the one
        # file it will actually contain, not the longest file anywhere in
        # the class. Cycling through the pool (rather than random pick)
        # also means every source file gets roughly even representation
        # across the n clips generated, instead of leaving it to chance.
        chosen_file = src_files[i % len(src_files)]
        chosen_dur = durations[i % len(src_files)]
        soundscape_dur = chosen_dur * max_stretch + 2 * pad

        # fg_path is the SHARED PARENT folder, not this class's subfolder.
        # Scaper discovers valid labels by scanning subfolders of fg_path,
        # so it has to point at the folder that contains every class, and
        # the class itself is selected through the label argument below.
        sc = scaper_mod.Scaper(
            duration=soundscape_dur,
            fg_path=source_root,
            bg_path=bg_dir,
            random_state=seed_base + i,
        )
        sc.ref_db = -25
        sc.add_background(
            label=("choose", []),
            source_file=("choose", []),
            source_time=("uniform", 0, 5),
        )
        sc.add_event(
            label=("const", cls),
            source_file=("const", os.path.join(src_dir, chosen_file)),
            source_time=("const", 0),
            event_time=("const", pad),
            event_duration=("const", chosen_dur),
            snr=("uniform", snr_range[0], snr_range[1]),
            pitch_shift=("uniform", pitch_range[0], pitch_range[1]) if pitch_range else None,
            time_stretch=("uniform", time_range[0], time_range[1]) if time_range else None,
        )

        out_path = os.path.join(out_dir, f"{cls}_synth{i:03d}.wav")
        tmp_jams = os.path.join(out_dir, f"_tmp{i}.jams")
        try:
            sc.generate(out_path, tmp_jams, disable_sox_warnings=True,
                       no_audio=False)
            made += 1
        except Exception as e:
            msg = str(e)
            if "SoX could not be found" in msg:
                return made, "SOX_MISSING"
            print(f"      skipped one: {type(e).__name__}: {msg[:100]}")
        finally:
            if os.path.exists(tmp_jams):
                os.remove(tmp_jams)

    return made, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=20,
                    help="how many synthetic clips to generate per class")
    ap.add_argument("--pad", type=float, default=0.15,
                    help="seconds of background context kept around each event")
    ap.add_argument("--no-pitch", action="store_true",
                    help="skip pitch shifting, in case SoX gives you trouble with it")
    ap.add_argument("--no-stretch", action="store_true",
                    help="skip time stretching")
    args = ap.parse_args()

    try:
        import scaper
    except ImportError:
        sys.exit(
            "scaper is not installed.\n"
            "    .venv\\Scripts\\pip install scaper jams pyloudnorm"
        )

    if not os.path.isdir(SOURCE_FOLDER):
        sys.exit(
            f"{SOURCE_FOLDER}\\ does not exist.\n"
            f"Expected something like:\n"
            f"    {SOURCE_FOLDER}\\wet_chewing\\*.wav\n"
            f"    {SOURCE_FOLDER}\\gulping\\*.wav\n"
            f"If you already ran clean_audio.py, point this at that output, "
            f"or copy those files in."
        )

    bg_classes = find_class_folders(BG_FOLDER)
    if not bg_classes:
        sys.exit(
            f"No background audio found.\n"
            f"{BG_FOLDER}\\ needs at least one subfolder containing audio, "
            f"for example:\n"
            f"    {BG_FOLDER}\\room\\your_file.wav\n"
            f"Copy your extra_audio\\other file in there to start."
        )

    import shutil as _shutil
    import tempfile

    ffmpeg = _shutil.which("ffmpeg")
    if ffmpeg is None:
        sys.exit(
            "ffmpeg is not on PATH. It's needed to convert non-WAV audio "
            "(like .m4a or .mp3) before Scaper can read it."
        )

    stage_root = tempfile.mkdtemp(prefix="scaper_stage_")
    staged_source = os.path.join(stage_root, "source")
    staged_bg = os.path.join(stage_root, "bg")

    print("Converting source and background audio to WAV first.")
    print("Scaper cannot read .m4a, .mp3, or other compressed formats")
    print("directly, so this makes throwaway WAV copies to work from.")
    print("Your original files in augment_source and augment_backgrounds")
    print("are not touched.")
    print()
    stage_as_wav(ffmpeg, SOURCE_FOLDER, staged_source)
    stage_as_wav(ffmpeg, BG_FOLDER, staged_bg)

    classes = find_class_folders(staged_source)
    if not classes:
        _shutil.rmtree(stage_root, ignore_errors=True)
        sys.exit("Nothing converted successfully. Check your source files play correctly.")

    pitch_range = None if args.no_pitch else (-2, 2)
    time_range = None if args.no_stretch else (0.9, 1.1)

    print(f"Classes found: {classes}")
    print(f"Generating {args.per_class} synthetic clips per class.")
    print(f"pitch shift: {'off' if pitch_range is None else f'{pitch_range} semitones'}")
    print(f"time stretch: {'off' if time_range is None else f'{time_range}x'}")
    print()

    random.seed(0)
    total = 0
    for cls in classes:
        src_dir = os.path.join(staged_source, cls)
        out_dir = os.path.join(OUT_FOLDER, cls)
        made, error = generate_for_class(
            scaper, cls, staged_source, src_dir, staged_bg, out_dir, args.per_class,
            args.pad, (3, 15), pitch_range, time_range, seed_base=hash(cls) % 10000)

        if error == "SOX_MISSING":
            print(f"{cls:<18} STOPPED: SoX is not on PATH. See the setup")
            print("                   instructions at the top of this file.")
            sys.exit(1)
        elif error:
            print(f"{cls:<18} {error}")
        else:
            print(f"{cls:<18} {made}/{args.per_class} generated -> {out_dir}")
            total += made

    _shutil.rmtree(stage_root, ignore_errors=True)

    print()
    print(f"{total} new synthetic clips written to {OUT_FOLDER}\\")
    print()
    print("These are new audio files, sized to match what clean_audio.py")
    print("produces. Copy the ones you want into training_data\\<class>\\")
    print("alongside your real recordings, then rerun fine_tune_yamnet.py.")
    print()
    print("Listen to a few before trusting them blindly. Occasionally a big")
    print("pitch shift makes a sound stop resembling the original.")


if __name__ == "__main__":
    main()
