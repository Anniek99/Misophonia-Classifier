"""
get_yamnet.py  --  Download YAMNet without tensorflow_hub.

tensorflow_hub fails a lot on Windows: SSL certificate problems, school network
filters, and tfhub.dev being deprecated. This skips it and downloads the model
as a plain file instead. Once it works, the model is saved next to this script
and never downloads again.

RUN:
    .venv\\Scripts\\python get_yamnet.py

If every method fails, this prints exactly what went wrong at each step, which
is the useful thing to bring to your mentor.
"""

import os
import ssl
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "yamnet.tflite")
CLASSMAP_PATH = os.path.join(HERE, "yamnet_class_map.csv")

MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite",
    "https://storage.googleapis.com/mediapipe-assets/yamnet_audio_classifier_with_metadata.tflite",
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/android/lite-model_yamnet_classification_tflite_1.tflite",
]

CLASSMAP_URLS = [
    "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
]


def fetch(urls, dest, label):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"  {label}: already have it ({os.path.getsize(dest)//1024} KB)")
        return True

    # Some Windows Python installs have no working certificate bundle.
    contexts = [("normal", None)]
    try:
        import certifi
        contexts.append(("certifi", ssl.create_default_context(cafile=certifi.where())))
    except Exception:
        pass
    unverified = ssl.create_default_context()
    unverified.check_hostname = False
    unverified.verify_mode = ssl.CERT_NONE
    contexts.append(("no SSL check", unverified))

    for url in urls:
        for cname, ctx in contexts:
            try:
                print(f"  trying {url[:70]}...  [{cname}]")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                kw = {"timeout": 120}
                if ctx is not None:
                    kw["context"] = ctx
                with urllib.request.urlopen(req, **kw) as r:
                    data = r.read()
                if len(data) < 1000:
                    print(f"     too small ({len(data)} bytes), skipping")
                    continue
                with open(dest, "wb") as f:
                    f.write(data)
                print(f"     OK, saved {len(data)//1024} KB")
                return True
            except Exception as e:
                print(f"     failed: {type(e).__name__}: {str(e)[:90]}")
    return False


def main():
    print("=" * 70)
    print("Downloading YAMNet")
    print("=" * 70)
    print()

    print("1. The class list (521 sound names)")
    ok_map = fetch(CLASSMAP_URLS, CLASSMAP_PATH, "class map")
    print()

    print("2. The model itself")
    ok_model = fetch(MODEL_URLS, MODEL_PATH, "model")
    print()

    if ok_model and ok_map:
        print("=" * 70)
        print("Both downloaded. Checking the model loads...")
        try:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except Exception:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
            it = Interpreter(model_path=MODEL_PATH)
            it.allocate_tensors()
            inp = it.get_input_details()[0]
            out = it.get_output_details()[0]
            print(f"  input  : {list(inp['shape'])}")
            print(f"  output : {list(out['shape'])}")
            print()
            print("Works. Now run:")
            print("    .venv\\Scripts\\python run_test.py")
        except Exception as e:
            print(f"  Model downloaded but will not load: {e}")
            return 1
        print("=" * 70)
        return 0

    print("=" * 70)
    print("COULD NOT DOWNLOAD")
    print("=" * 70)
    print()
    if not ok_map:
        print("  class map : FAILED")
    if not ok_model:
        print("  model     : FAILED")
    print()
    print("Almost always a network problem. Things to try, in order:")
    print()
    print("  1. Phone hotspot instead of school or cafe wifi.")
    print("  2. .venv\\Scripts\\pip install --upgrade certifi")
    print("  3. Download by hand in a browser:")
    for u in MODEL_URLS[:1]:
        print(f"       {u}")
    for u in CLASSMAP_URLS[:1]:
        print(f"       {u}")
    print(f"     then save them into:")
    print(f"       {MODEL_PATH}")
    print(f"       {CLASSMAP_PATH}")
    print()
    print("Option 3 always works, since a browser handles certificates itself.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
