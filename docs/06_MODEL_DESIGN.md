# Model Design — DSP Front-End, CNN Architectures, Quantisation, Memory

Governing constraint: **the on-device feature extraction must reproduce the training-time
feature extraction closely enough that on-device predictions match the Python reference.**
If that does not hold, an excellent offline model is worthless on hardware. Every choice
below is made to protect that property.

---

## 0. Toolchain facts this design depends on — VERIFIED 2026-08-12

Confirmed by generating and building real projects with the installed SDK
(see [03_DEV_ENVIRONMENT.md §5](03_DEV_ENVIRONMENT.md)):

| Fact | Status |
|---|---|
| AI/ML extension **3.0.1** installed | Yes |
| **MVP v1 accelerated TFLM kernels compile for EFR32MG26** | `tflite_micro_mvpv1_accelerator`, `conv2d_compiled`, `fully_connected_compiled`, `pooling2d_compiled` all in the linked image |
| Device-targeted codegen for `efr32xg26` | `autogen/core/platform/efr32xg26` |
| Reference audio classifier footprint on BRD2608A | `.text` 207 kB, `.bss` 141 kB, **373 kB heap still free** |
| `cmsis_dsp` available from SiSDK | Yes |

> Note: **TFLM is code-generated per model.** Adding `tensorflow_lite_micro` alone does *not*
> produce a buildable TFLM — the tflite-micro tree is emitted into `autogen/core/` by the
> `ml_compiler` from an actual model. The workflow is: add the **`ml_model`** component, drop
> `<name>.tflite` **and** `<name>.mlconf` into `config/tflite/`, and SLC generates only the
> kernels the model needs.
>
> **Consequence for planning: the firmware will not build until a model exists.** Phase 4 of
> [10_IMPLEMENTATION_PLAN.md](10_IMPLEMENTATION_PLAN.md) pushes a *dummy* `.tflite` through
> this pipeline early precisely to de-risk it.
>
> A second consequence is a bonus: because codegen is model-driven, the operator whitelist in
> §1 below is not merely a performance preference — **unlisted operators pull in extra
> generated code and lose acceleration.** Staying inside the list keeps the image small.

---

## 0b. MEASURED results (2026-08-13) — cardio model built and quantised

| Metric | Cardio (`chiron_cardio_v1`) |
|---|---|
| Task | Yaseen 2018, 5 classes |
| Cross-validated accuracy | **98.9 % ± 1.1 %** (5-fold) |
| INT8 held-out accuracy | **98.06 %** |
| float → int8 delta | **+0.00 pp** |
| float/int8 agreement | **100.00 %** |
| Parameters | 13,509 |
| `.tflite` size | **27.7 kB** |
| Operators | `CONV_2D`, `DEPTHWISE_CONV_2D`, `AVERAGE_POOL_2D`, `SOFTMAX` |

The architecture below is **narrower than originally specified** (width 16, 4 blocks,
13.5 k params rather than 41 k) because it already saturates the task. Do not scale it up
without evidence that the additional capacity improves measured performance.

> Note: **Classifier-head trap, discovered in conversion.** Writing the head as
> `AveragePooling2D → Reshape((ch,)) → Dense` makes Keras compute the reshape target at
> runtime, emitting **`SHAPE`, `STRIDED_SLICE` and `PACK`**. None has an MVP kernel, and
> dynamic shapes are hostile to a static TFLM arena.
>
> **Fix: use a 1×1 `Conv2D` on the `(1, 1, ch)` tensor instead.** It is numerically identical
> to the Dense layer, keeps the graph entirely inside the accelerated operator set, and made
> the model smaller (28.6 → 27.7 kB). The output tensor is then `[1, 1, 1, n_classes]`, which
> the firmware reads as `n_classes` consecutive int8 values.

---

## 1. Operator whitelist — design backwards from the accelerator

Silicon Labs ships MVP-accelerated TFLM kernels for exactly these operators
(`tensorflow_lite_micro_accelerated_kernels.slcc`, `src/kernels/mvp1/`, verified in the
installed extension):

```
add   conv   depthwise_conv   fully_connected   mul   pooling   transpose_conv
```

The component's own description confirms the fallback behaviour: *"Remaining kernels fall back
to using optimized or reference kernel implementations."*

**Both CNNs use only:** `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`, `ADD`,
`AVERAGE_POOL_2D`, `RESHAPE`, `SOFTMAX`.

- `SOFTMAX` has no MVP kernel but falls back to CMSIS-NN and runs once on a ≤6-element
  vector — irrelevant.
- **No** `BATCH_NORM` at inference (folded into the preceding conv by the TFLite converter).
- **No** `MEAN`-based global average pooling — use `AVERAGE_POOL_2D` with a full-size kernel,
  which *is* accelerated. This is a small detail that silently costs 2–3× if configured incorrectly.
- **No** LSTM/GRU. The Fraiwan 99.62 % result used CNN+BDLSTM, but the CNN-alone variant
  scored **99.04 %** — a 0.58 pp cost to stay entirely inside the accelerated operator set and
  keep a static, arena-friendly memory profile. That is a good trade.

---

## 2. DSP front-end

### 2.1 Parameters

| Parameter | Cardio | Pulmo |
|---|---|---|
| Sample rate | 8 kHz | 8 kHz |
| Window | 3.0 s (24,000 samples) | 5.0 s (40,000 samples) |
| Band-pass | 20–800 Hz | 80–2000 Hz |
| FFT size | **256** (32 ms) | **512** (64 ms) |
| Hop | 128 (16 ms) | 256 (32 ms) |
| Window fn | Hamming | Hamming |
| Mel bins | **40** | **64** |
| Frames produced | 187 | 156 |
| **Model input** | **40 × 96 × 1** | **64 × 128 × 1** |
| Frame→input reduction | average-pool 2 frames, crop | crop centre 128 |

Both are powers of two in FFT size so `arm_rfft_q15` applies directly, and both mel counts
are what the training pipeline uses — no resampling of the feature matrix on device.

### 2.2 Fixed-point strategy (the part that usually breaks)

| Stage | Offline (Python) | On-device (C) |
|---|---|---|
| Band-pass | `scipy` Butterworth, **`filtfilt`** | cannot use zero-phase |
| FFT | `numpy` float64 | `arm_rfft_q15` (Q15) |
| Magnitude | float | `arm_cmplx_mag_q15` |
| Mel filterbank | float matmul | Q15 matmul with pre-scaled bank |
| Log | `np.log(x + 1e-6)` | fixed-point `log2` LUT × ln2 |
| Quantise | TFLite scale/zero-point | same scale/zero-point |

> Note: **The `filtfilt` trap.** `scipy.signal.filtfilt` is zero-phase and non-causal. There is no
> real-time equivalent. If training uses `filtfilt` and firmware uses a causal IIR, the
> features differ and on-device accuracy drops for no visible reason.
>
> **Rule: the training pipeline must use the *same causal* filter the firmware will use.**
> Implement the band-pass once as a **cascaded biquad (SOS)**, apply it **forward-only** in
> Python (`scipy.signal.sosfilt`, not `sosfiltfilt`), and port the identical SOS coefficients
> to `arm_biquad_cascade_df1_q15`. Non-negotiable.

Same principle for every other stage: **the Python reference is written to be portable to
fixed point**, and a bit-comparison test guards it (see
[11_VERIFICATION_AND_TEST.md](11_VERIFICATION_AND_TEST.md) §3).

### 2.3 Streaming, not buffering
The 5 s Pulmo window is 80 kB as int16. We never hold it in RAM:

```
flash (golden vector, int16 PCM)
   │  read 512 samples at a time
   ▼
biquad band-pass (persistent state)  →  Hamming  →  rfft_q15  →  mag  →  mel  →  log
   │
   ▼  append one column
feature matrix  64 × 128 int8  (8 kB, the only large buffer)
```

Peak audio-path RAM: ~4 kB of ring buffer + 8 kB feature matrix.

---

## 3. CNN architectures

Depthwise-separable, BN folded, ReLU6, global average pool, single dense head.

### 3.1 Cardio — `chiron_cardio_v1`

Input **40 × 96 × 1** (int8)

| # | Layer | Out shape | Params |
|---|---|---|---|
| 1 | Conv2D 3×3, 32, s=2, ReLU6 | 20 × 48 × 32 | 320 |
| 2 | DWSepConv 3×3 → 64, s=2, ReLU6 | 10 × 24 × 64 | 2,400 |
| 3 | DWSepConv 3×3 → 96, s=2, ReLU6 | 5 × 12 × 96 | 6,816 |
| 4 | DWSepConv 3×3 → 128, s=2, ReLU6 | 3 × 6 × 128 | 13,280 |
| 5 | DWSepConv 3×3 → 128, s=1, ReLU6 | 3 × 6 × 128 | 17,664 |
| 6 | AveragePool2D 3×6 | 1 × 1 × 128 | 0 |
| 7 | Reshape → Dense 5 | 5 | 645 |
| | **Total** | | **≈ 41 k** |

INT8 weights ≈ **41 kB**. Add TFLite flatbuffer overhead → **~50 kB flash**.

### 3.2 Pulmo — `chiron_pulmo_v1`

Input **64 × 128 × 1** (int8)

| # | Layer | Out shape | Params |
|---|---|---|---|
| 1 | Conv2D 3×3, 32, s=2, ReLU6 | 32 × 64 × 32 | 320 |
| 2 | DWSepConv 3×3 → 64, s=2, ReLU6 | 16 × 32 × 64 | 2,400 |
| 3 | DWSepConv 3×3 → 96, s=2, ReLU6 | 8 × 16 × 96 | 6,816 |
| 4 | DWSepConv 3×3 → 128, s=2, ReLU6 | 4 × 8 × 128 | 13,280 |
| 5 | DWSepConv 3×3 → 160, s=1, ReLU6 | 4 × 8 × 160 | 21,952 |
| 6 | DWSepConv 3×3 → 160, s=1, ReLU6 | 4 × 8 × 160 | 27,200 |
| 7 | AveragePool2D 4×8 | 1 × 1 × 160 | 0 |
| 8 | Reshape → Dense 6 | 6 | 966 |
| | **Total** | | **≈ 73 k** |

INT8 weights ≈ **73 kB** → **~85 kB flash**.

Both are small enough that **flash is not the constraint** — the constraint is having enough
capacity to learn without overfitting 213 patients. If validation shows underfitting, widen
channels before adding depth; if overfitting, cut channels and raise augmentation. Both
directions have ample memory headroom.

### 3.3 Tensor arena (peak activation memory)

TFLM allocates a single arena sized by the largest simultaneous set of live tensors.

| Model | Largest activation | Estimated arena | Allocated |
|---|---|---|---|
| Cardio | layer 1 out: 20 × 48 × 32 = 30,720 B | ~55 kB | **64 kB** |
| Pulmo | layer 1 out: 32 × 64 × 32 = 65,536 B | ~110 kB | **128 kB** |

> Do **not** guess the arena at ship time. `MicroInterpreter::arena_used_bytes()` after
> `AllocateTensors()` returns the true figure — log it over VCOM at boot and size the static
> buffer to that value + 10 %. Record both numbers in the report.

The Silicon Labs **ML Profiler** (Simplicity Studio → AI Tools) independently reports
RAM/flash fit and per-layer timing; run it and paste its output into the report as third-party
evidence.

---

## 4. Quantisation

**Full-integer INT8**, TFLite post-training quantisation:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_gen          # ≥ 200 TRAINING-fold windows
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8
```

Rules:
- The representative dataset comes from the **training fold only** (test data must never touch
  the quantiser).
- It must cover **all classes** and the full loudness range, or activation ranges will be
  miscalibrated for rare classes.
- **Measure float→int8 accuracy drop and report it.** Expect < 1 pp. If the drop exceeds
  2 pp, the cause is almost always a layer with a very wide dynamic range — fix it with
  quantisation-aware training (QAT) rather than by accepting the loss.

`inference_input_type = int8` matters: it means the firmware feeds the log-mel matrix in
directly, with no float conversion step on device.

---

## 5. Per-node budget summary

| | Cardio (BRD2608A) | Pulmo (BRD2709A) |
|---|---|---|
| Model flash | ~50 kB | ~85 kB |
| TFLM runtime + MVP kernels | ~120 kB | ~120 kB |
| BLE stack + app + bootloader | ~350 kB | ~350 kB |
| CMSIS-DSP + mel tables | ~40 kB | ~40 kB |
| Golden vectors | 8 MB external SPI flash | ~1.9 MB internal (24 × 5 s) |
| **Flash used (internal)** | **~560 kB / 3200 kB** | **~2.5 MB / 3200 kB** |
| Tensor arena | 64 kB | 128 kB |
| BLE stack RAM | ~40 kB | ~40 kB |
| DSP scratch + feature matrix | ~20 kB | ~24 kB |
| **RAM used** | **~124 kB / 512 kB** | **~192 kB / 512 kB** |

Every figure has ≥ 2× headroom. This design is not memory-limited — which is the point:
it removes an entire class of demo-day failure.

### 5.1 Measured baselines (BRD2608A, 2026-08-12)

These are real numbers from linked images, not estimates:

| Build | `.text` | static RAM | free heap |
|---|---|---|---|
| `bt_soc_empty` (BLE only) | 209,568 B | ~10.6 kB | **501 kB** |
| AI/ML audio classifier (mic + microfrontend + TFLM + MVP, MicriumOS) | 207,216 B | ~141 kB | **373 kB** |

The second row is the important one: a *complete* on-device audio classification application
on our exact silicon costs ~202 kB flash and ~137 kB RAM. Chiron's per-node budget above sits
inside that envelope, so the flash/RAM risk for this project is effectively closed.

Note also that SiSDK's memory manager claims all unused RAM as `.memory_manager_heap` — the
TFLM arena and BLE stack allocate from it. Do not read a large `.bss`/heap figure as
"RAM exhausted"; read `arena_used_bytes()` instead.

---

## 6. Expected inference latency

Rough estimate at 78 MHz with MVP-accelerated kernels; **must be replaced with measured
values** (GPIO toggle + logic analyser — see [02_HARDWARE_INVENTORY.md](02_HARDWARE_INVENTORY.md) §2).

| Stage | Cardio | Pulmo |
|---|---|---|
| Band-pass + framing | ~5 ms | ~10 ms |
| FFT + mel + log (all frames) | ~25 ms | ~45 ms |
| TFLM `Invoke()` (MVP) | ~20 ms | ~40 ms |
| **Total per window** | **~50 ms** | **~95 ms** |

Well under real time (3 s / 5 s windows), so a future continuous-audio phase has large margin.

---

## 7. Training recipe

| Setting | Value |
|---|---|
| Framework | TensorFlow / Keras (required for the `.tflite` → TFLM path) |
| Optimiser | AdamW, lr 3e-3, cosine decay |
| Batch | 64 |
| Epochs | 150, early stop on val macro-F1, patience 20 |
| Loss | Categorical cross-entropy; **class-weighted for Pulmo** |
| Label smoothing | 0.05 |
| Regularisation | dropout 0.2 before the dense head, weight decay 1e-4 |
| Seeds | 3 seeds × 10 folds; report mean ± std |

Deliverable per model: `float.h5`, `int8.tflite`, `metrics.json`, `confusion_matrix.png`,
`training_curves.png`, and `arena_used_bytes` from the device.

---

## 8. Why not one shared-backbone model?

The earlier project proposal compared a dual-model pipeline (A) against a shared-backbone
branching model (B) and selected A. This phase reaches the same conclusion for a stronger
reason: **the two models live on two different chips.** A shared backbone is not merely
harder to train, it is architecturally impossible here. Pipeline A is the only option, and
it brings genuine benefits:

- The two models are **independently trainable, testable and re-flashable**. Retraining the
  lung model cannot regress the heart model.
- Different input shapes and different band-passes, each tuned to its own physiology.
- Two nodes is a better demo than one — it shows a real multi-node BLE topology.
