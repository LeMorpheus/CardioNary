# Test_Data — reference test set

**This folder contains no patient data.** It is the held-out public-dataset
material the system is tested against, exported from the golden vectors compiled
into the two firmware images.

Clinical recordings live in `Real_World_Data/` and are kept entirely separate.
Nothing here came from a patient, and nothing here should ever be mixed with what
is in that folder.

---

## Contents

| Path | Count | What it is |
|---|---|---|
| `pulmo_images/` | 120 PNG | 96×96 greyscale chest radiographs, Kermany corpus, held-out fold 0 |
| `cardio_audio/` | 53 WAV | 3.0 s heart sounds, 8 kHz mono int16, Yaseen corpus, held-out |
| `index.json` | — | Per-file ground truth, offline INT8 prediction and original source filename |

Filenames carry the vector id and the ground-truth class, e.g.
`vec025_normal.png`, `vec012_as.wav`.

These are byte-identical to the arrays compiled into the firmware — verified by
round-trip comparison against the generated C. Uploading one through the
dashboard therefore exercises the live path against a known answer.

---

## Where this material originates

| Stage | Location |
|---|---|
| Source corpora | `data/` (Kermany chest X-ray, Yaseen heart sound) |
| Vector manifests, with ground truth and offline predictions | `vectors/chiron_{cardio,pulmo}_vectors_manifest.json` |
| Generated C arrays compiled into firmware | `firmware/chiron_{cardio,pulmo}/vectors/*.c` |
| **Exported as ordinary files** | **`Test_Data/` (this folder)** |
| On-device sweep results | `reports/hw_selftest_{cardio,pulmo}.json` |
| Class-balanced subset shipped with the defect report | `silabs_aiml_defect_report/reproduce/golden_subset.npz` |

---

## Expected results when uploading through the dashboard

Compare a live result against the **device's own flash-sweep answer** in
`reports/hw_selftest_*.json`, not against the offline INT8 prediction in the
manifest. The two differ on exactly two pulmonary vectors, and both differences
are known and reproducible:

| Vector | Offline INT8 | On device |
|---|---|---|
| `vec003` | Normal | **UNCERTAIN** — confidence 0.514, below the reporting floor |
| `vec025` | Pneumonia | **Normal** — confidence 0.549 |

A live upload of either file reproduces the device column, not the offline
column. That is correct behaviour: the live path runs the same front-end and the
same model on the same silicon.

Measured on 2026-08-25 over the live upload path:

| Path | Result |
|---|---|
| Image upload vs on-device flash path | 10/10 |
| Audio upload vs on-device flash path | 5/5, spanning all five cardiac classes |

**Audio replay note.** When replaying a file from `cardio_audio/`, send
`normalise=false`. These vectors are already at the training signal level;
the default gain stage exists for microphone input, which sits around −25 dBFS,
and applying it to a golden vector changes the samples and so changes the answer.

---

## Regenerating

The exporter parses the generated C arrays and the manifests. Re-run it if the
vector set is ever rebuilt:

```bash
python tools/export_test_data.py
```
