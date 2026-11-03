# TheraPhono: misophonia trigger sound detector

TheraPhono is a wearable that listens for a misophonia trigger sound and shows a matching harmless video (a PAVS). This repo is the sound detector. It labels each one-second piece of audio as one of five trigger sounds, speech, or other.

It uses Google's YAMNet to turn audio into numbers, then trains a small classifier on those numbers. That's transfer learning. The script is called `fine_tune_yamnet.py`, but YAMNet itself is never changed.

## How to replicate the main result

You need a Windows computer and the data zip from Annie. The audio and videos aren't in this repo because they're large and have their own licenses.

**1. Install Python.** Run this, then close and reopen the terminal:

```
winget install Python.Python.3.13
```

**2. Get the code.**

```
git clone https://github.com/Anniek99/Misophonia-Classifier.git
cd Misophonia-Classifier
```

**3. Install the packages.**

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**4. Unzip the data into this folder** so it looks like this:

```
Misophonia-Classifier/
  training_data/
    wet_chewing/  gulping/  fork_scraping/  pen_clicking/  sniffling/
  extra_audio/
    speech/
    other/
  OVS Videos/
  tm_model/            (only needed for run_test.py)
```

**5. Run it.**

```
.venv\Scripts\python fine_tune_yamnet.py --seeds 5
```

Turning the audio into YAMNet numbers is the slow part. It happens once, even with five seeds. If YAMNet won't download, the script prints what to download by hand.

**6. Compare with the expected result.** Mean across the five runs (std is the population std):

| Class | Test pieces | Recall | Precision | F1 |
|---|---|---|---|---|
| wet_chewing | 33 | 51% | 97% | 0.655 ± 0.113 |
| gulping | 26 | 69% | 68% | 0.676 ± 0.023 |
| fork_scraping | 26 | 37% | 44% | 0.395 ± 0.041 |
| pen_clicking | 27 | 74% | 51% | 0.596 ± 0.049 |
| sniffling | 52 | 83% | 79% | 0.807 ± 0.047 |
| **Macro F1** | 164 | | | **0.626 ± 0.024** |

Mean accuracy is 65.5%. The macro F1 for each run was 0.619, 0.672, 0.600, 0.617 and 0.622. The seeds are 42, 43, 44, 45 and 46. Two separate runs on the original computer matched exactly. A different computer or TensorFlow version may differ a little.

The run also writes `seed_summary.csv` (the table) and `seed_results.csv` (every run), and saves the last run's model to `finetuned_model/`.

## What each script is for

| Script | What it does | Needs | Makes |
|---|---|---|---|
| `fine_tune_yamnet.py` | Trains the classifier and scores it on the OVS test videos. This is the main result | `training_data/`, `extra_audio/`, `OVS Videos/` | `finetuned_model/`, plus `seed_summary.csv` and `seed_results.csv` with `--seeds` |
| `run_test.py` | Scores Teachable Machine and untrained YAMNet on the same videos. Also holds helper functions the other scripts use | `OVS Videos/`, `tm_model/`, and the files from `get_yamnet.py` | `results.csv` |
| `get_yamnet.py` | Downloads YAMNet as plain files, for `run_test.py` | Internet | `yamnet.tflite`, `yamnet_class_map.csv` |
| `compare_finetuned.py` | Untrained YAMNet vs the trained classifier on identical clips, with false alarms and a significance test | `finetuned_model/`, `yamnet_class_map.csv`, `OVS Videos/`, `extra_audio/` | `compare_finetuned_results.csv` |
| `simulate_realtime.py` | Replays audio as a stream to measure how long the device takes to react, and how often it fires by mistake | `finetuned_model/`, `OVS Videos/`, `extra_audio/other/` | A table printed on screen |

Run `fine_tune_yamnet.py` before `compare_finetuned.py` and `simulate_realtime.py`, because they load the model it saves.

## Other commands

| Command | What it does |
|---|---|
| `.venv\Scripts\python fine_tune_yamnet.py` | One run with seed 42. Expect macro F1 0.619 and accuracy 62.8% |
| `.venv\Scripts\python fine_tune_yamnet.py --debug-windows fork_scraping` | Saves the first fork_scraping file's windows to `debug_windows/`, with times and labels, so you can listen to them |
| `.venv\Scripts\python get_yamnet.py` | Downloads the files `run_test.py` needs |
| `.venv\Scripts\python run_test.py` | Baselines. Expect Teachable Machine near 0.20 macro F1 and YAMNet with a label map near 0.41 |
| `.venv\Scripts\python compare_finetuned.py` | Same-clip comparison. Rerun it with the current data before quoting numbers |
| `.venv\Scripts\python simulate_realtime.py --sweep` | Tries several hop and agreement settings |

The `run_test.py` numbers come from an earlier, larger test (7 classes, 358 pieces), so they don't compare directly with the table above.

To train on a different folder, change `TRAINING_DATA_FOLDER` near the top of `fine_tune_yamnet.py`.

## The data

**Training set for the reported results:** 53 files, cut into 1,270 one-second pieces. That's wet_chewing 8 files, gulping 6, fork_scraping 27, pen_clicking 6, sniffling 4, speech 1 and other 1. Every piece takes its file's label, including quiet pieces. Training and validation are split by whole file, never by piece.

**Test set:** six OVS videos. Vid08 and Vid12 are sniffling, Vid09 is fork scraping, Vid17 is wet chewing, Vid18 is gulping and Vid20 is pen clicking. Test counts are one-second pieces, not videos. Only the five trigger classes are scored, because every test video is a trigger sound and speech and other have no test examples.

The videos come from the Sound-Swapped Video database (https://osf.io/3ysfh/).

## Limits

- The test set is small. Wet chewing has one test video and sniffling has two, so per-class numbers are rough.
- Speech and other have one training file each. False alarms on everyday sound haven't been measured properly.
- Results vary between runs because training starts from random weights. The seeds are fixed, and the five-run average is the number to report.
- Timing was measured on a laptop. Inference time on the Raspberry Pi has to be measured on the Pi.

Earlier experiments are in the git history at the tag `archive-experiments`: audio segmentation, augmentation, Teachable Machine data prep, and the script that built the YAMNet label map in `run_test.py`.

## References

- YAMNet: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- Samermit et al. (2022). Development and evaluation of a sound-swapped video database for misophonia. *Frontiers in Psychology*, 13, 890829.
- Mahzouni (2025). *Multisensory Temporal Integration of Simple, Neutral, and Trigger Stimuli in Misophonia.* UC Santa Cruz. https://escholarship.org/uc/item/9jd5f5vm
