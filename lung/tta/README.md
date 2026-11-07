# lung/tta — test-time adaptation

**Owner:** Aayush Pratap Asnani (`A`)  
**Stage:** S1 / S3 (Proposal 4.3)

X-ray machines, exposure settings and hospitals differ. A model trained on one
source will meet others. This workstream adapts the model at test time, from a new
site's own unlabelled images, so it holds up across equipment.

---

## Method: Batch-Normalisation Adaptation

Re-estimate batch-normalisation statistics using $N$ unlabelled clinic images in a single forward pass:
1. Load the float model (`models/chiron_pulmo_v1_float.keras`). The original model is never modified.
2. Find every `BatchNormalization` layer (including the 52 layers nested inside MobileNetV2), set `trainable = True` and `momentum = 0.0`.
3. Feed the $N$ clinic images with `training=True`. Each layer's running mean and variance are directly replaced by the batch mean and variance; no gradients or backpropagation are computed.
4. Set all BN layers back to `trainable = False`.
5. **Collapse Guard:** If the adapted copy labels $> 95\%$ of the $N$ images as a single class while the original did not, revert all batch-normalisation statistics to the original values.

---

## Scripts & CLI

- `lung/tta/bn_adapt.py`:
  - `apply_bn_adaptation(model, clinic_images, collapse_threshold=0.95)`
  - `apply_synthetic_shift(images, dark_factor=0.7, contrast_factor=0.5)`
  - `evaluate_synthetic_shift(model_path, n_adapt=32, seed=0)`
- `lung/eval/timing.py`:
  - Mean inference latency per image (`time.perf_counter`)
  - Peak memory footprint (`tracemalloc`)
  - Edge budget verification (< 1 MB model size, < 1 s inference time)

---

## S1 Synthetic Shift Experiment

- **Data:** Kermany fold 0 held-out images.
- **Shift:** Darker ($\times 0.7$) and flatter (contrast halved around mean).
- **Protocol:** Adapt on $N = 32$ images; score the remaining images. The images used for adaptation are never scored (rule 7: honesty).
