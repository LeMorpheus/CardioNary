# The Machine Learning Behind CardioNary

**Team Chiron · BITS Pilani Hyderabad**

This document describes only the machine learning: the problems, the data, the
features, the models, how they were trained, how they were measured, and what was
deliberately not attempted. It is written to be read start to finish by someone who
knows some machine learning but nothing about this project.

One idea runs through all of it. **A number is only worth what its evaluation
protocol is worth.** Most of the work described here is not model building. It is
the work of making sure the numbers mean what they appear to mean — and, twice,
discovering that they did not, and throwing the result away.

---

## Table of contents

1. [The two learning problems](#1-the-two-learning-problems)
2. [The data](#2-the-data)
3. [Leakage and confounds — the part that decides whether any of this is real](#3-leakage-and-confounds--the-part-that-decides-whether-any-of-this-is-real)
4. [A task we killed with our own evidence](#4-a-task-we-killed-with-our-own-evidence)
5. [Feature engineering: the heart-sound model](#5-feature-engineering-the-heart-sound-model)
6. [Feature engineering: the chest X-ray model](#6-feature-engineering-the-chest-x-ray-model)
7. [The two architectures](#7-the-two-architectures)
8. [The training protocol](#8-the-training-protocol)
9. [Results](#9-results)
10. [Compression: 8-bit post-training quantisation](#10-compression-8-bit-post-training-quantisation)
11. [Teaching the models to abstain](#11-teaching-the-models-to-abstain)
12. [Interpretability, and three experiments that failed](#12-interpretability-and-three-experiments-that-failed)
13. [Measuring distribution shift instead of assuming it away](#13-measuring-distribution-shift-instead-of-assuming-it-away)
14. [What we refused to model](#14-what-we-refused-to-model)
15. [Honest limitations](#15-honest-limitations)
16. [Reproducibility](#16-reproducibility)

---

## 1. The two learning problems

Two independent supervised classification models. They share a training pipeline, an
evaluation protocol and a code base, and nothing else — different modalities,
different architectures, different failure modes.

| | **Heart-sound model** | **Chest X-ray model** |
|---|---|---|
| Input | 3 seconds of heart audio | One chest radiograph |
| Output | 5 classes | 2 classes |
| Classes | Normal · Aortic Stenosis · Mitral Stenosis · Mitral Regurgitation · Mitral Valve Prolapse | No pneumonia pattern · Pattern consistent with pneumonia |
| Type | Multi-class, balanced | Binary, imbalanced |
| Learned representation | Time–frequency | Spatial |
| Parameters | 13,509 | 412,773 |

Neither model is a diagnosis. Both are screening aids whose output a clinician
reviews. That is not a legal disclaimer bolted on at the end — it shapes real design
decisions later in this document, particularly the decision to let a model refuse to
answer.

### Why the heart problem is multi-class and not "normal vs abnormal"

A binary normal/abnormal heart-sound model is a much easier problem and a much less
useful one. Telling a clinician "something is wrong" adds almost nothing to what the
clinician already heard. Naming *which* valve lesion the sound resembles is a
genuinely different task, and it is the one worth solving. It also makes the
evaluation honest: with five balanced classes, chance is 20%, so a high score cannot
come from exploiting a majority class.

### Why the lung problem is binary

It was not meant to be. The lung task began as multi-class disease classification
from lung *audio*, and we abandoned it on evidence. Section 4 tells that story,
because it is the single most instructive thing in this project.

---

## 2. The data

### Heart sounds — Yaseen 2018

| Property | Value |
|---|---|
| Size | 1,000 recordings, 200 per class, perfectly balanced |
| Native format | 8 kHz, mono, 1.16–3.99 seconds per clip |
| Classes | Normal, Aortic Stenosis, Mitral Stenosis, Mitral Regurgitation, Mitral Valve Prolapse |
| Chance accuracy | 0.200 |

This is a curated teaching corpus: clean recordings, well-positioned, textbook
morphology. That matters enormously for how the results should be read, and it is
stated every time a number from it is quoted.

**One honest caveat, given up front.** Yaseen publishes one recording per case with
no patient identifier. We therefore treat each file as its own patient. That is the
strongest split the corpus permits, but it is not the same as a verified
patient-disjoint split, and we say so rather than letting the phrase
"patient-disjoint" do work it has not earned.

### Chest radiographs — Kermany paediatric corpus

| Property | Value |
|---|---|
| Size | 5,856 images from 2,790 subjects |
| Cohort | Children aged 1–5, one hospital, one imaging pipeline |
| Classes | Normal, Pneumonia |
| Majority-class baseline (subject level) | 0.600 |

The single-hospital, single-cohort property is not a limitation here. It is the
reason this corpus is usable at all — see the next section.

---

## 3. Leakage and confounds — the part that decides whether any of this is real

A classifier will learn whatever separates the classes. If something other than the
disease separates them, it will learn that instead, and it will score beautifully
while having learned nothing of value. Almost all of the effort in this project went
into removing those shortcuts and proving they were gone.

### 3.1 Anti-leakage rules, written as assertions rather than as prose

Documentation does not prevent leakage; failing tests do. The preprocessing stage
refuses to build a dataset unless all of the following hold:

| Rule | Enforced how |
|---|---|
| Every patient identifier appears exactly once in the manifest | assertion on uniqueness |
| Every patient belongs to exactly one fold | assertion that fold count per patient is 1 |
| Train and test patient sets are disjoint in every fold | explicit set-intersection assertion per fold |
| No two recordings share identical content | content hash, duplicates rejected |
| All labels lie inside the declared class set | set-containment assertion |
| Subjects with comorbidities are excluded where the task requires it | assertion on the comorbidity flag |
| Multiple renderings of the same acoustic event never straddle a split | only one rendering code admitted |

That last rule is subtle and worth spelling out. One of the lung-audio corpora ships
each recording three times, filtered three different ways. Those are not three
recordings — they are one event heard three ways. Allowing two of them into training
and one into test would produce a spectacular score from pure memorisation.

### 3.2 The audit of the chest X-ray corpus

Two findings, both of which changed what we shipped.

**The official train/test split leaks.** 264 subjects — 967 images — appear in
*both* official folders. Most published results on this corpus use that split, which
is a plausible source of the 96–98% figures in the literature. We discarded the
official split entirely, recombined the corpus, and assigned our own patient-disjoint
folds.

**The subject identifier is not global.** The numbering restarts per subset, so
`person1` in one folder is not `person1` in another. Concretely, 979 of 1,674 person
identifiers carry both bacterial and viral images — clinically implausible for one
child. Grouping by this identifier may therefore merge two children into one
"subject". We kept that grouping anyway, because the error direction is conservative:
merging two patients costs a little data efficiency but *cannot* create leakage,
whereas splitting one patient into two can. The cost is that a clean three-class
bacterial/viral/normal task is not defensible on this corpus, which is why the model
ships binary.

### 3.3 Why this corpus survives an audit that others fail

Both classes come from one hospital, one paediatric cohort, one imaging pipeline.
There is no source variable to shortcut on, by construction.

We checked what happens when that control is absent, using a published study that
holds the disease images fixed and changes only where the *healthy* class comes from:

| Setting | Accuracy |
|---|---|
| Healthy children vs diseased adults (source-confounded) | 0.975 |
| Healthy adults vs diseased adults (source-controlled) | **0.747** |

22.8 points of "accuracy" came from anatomy and acquisition, not disease. The same
study's decisive control is a classifier separating *healthy children* from *healthy
adults* — no pathology in either class — which reaches 0.974. A model can score 97%
on a task where the label carries no medical information at all.

That is the standard we held our own corpora to.

---

## 4. A task we killed with our own evidence

The lung model was supposed to be multi-class disease classification from lung
sounds. Published work reports up to 99.62% on exactly these corpora. We could not
reproduce anything like it under an honest protocol, and the reason turned out to be
more interesting than the result.

### 4.1 The idea: a baseline that cheats on purpose

Before trusting any accuracy number, build the dumbest possible classifier that has
access only to the confound and none of the signal. Here, that is a classifier that
reads which stethoscope model recorded the file — from the filename — and never opens
the audio at all. Its accuracy is the **hurdle**. Any real model must beat it, or it
has demonstrated nothing.

### 4.2 Five experiments, every one patient-disjoint

| # | Task | Units | Classes | Hurdle | Our CNN | Δ | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | Heart sounds | 1,000 clips | 5 | 0.200 (chance) | **0.989** | **+78.9** | ship |
| 2 | Lung audio, 4-class | 92 patients | 4 | 0.370 | 0.446 | +7.6 | weak |
| 3 | Lung audio, 6-class | 123 patients | 6 | **0.675** (stethoscope-only) | 0.643 | **−3.2** | fail |
| 4 | Lung audio, one stethoscope | 59 patients | 5 | 0.441 | 0.391 | **−5.0** | fail |
| 5 | Lung audio, merged corpora | 201 patients | 6 | **0.582** (stethoscope-only) | 0.592 | +1.0 | fail |

Row 3 is the one that ends the argument. **A classifier that reads the stethoscope
brand and ignores the audio entirely beats our neural network.** Row 5 looks like a
narrow win, but +1.0 point sits well inside the ±7.2-point fold standard deviation —
statistically indistinguishable from the cheating baseline.

The per-class recall of the merged model shows the mechanism directly:

| Class | Recall | Which corpus it comes from |
|---|---|---|
| COPD | 0.877 | mostly corpus B |
| Normal | 0.650 | both |
| Heart Failure | 0.444 | corpus A only |
| Asthma | 0.258 | corpus A only |
| **Pneumonia** | **0.000** | corpus B only |
| **Bronchiectasis** | **0.000** | corpus B only |

Two classes are never predicted at all. The model has not learned six diseases; it
has learned to tell two datasets apart and is guessing within each.

### 4.3 What we did about it

We deleted the task. The lung problem moved to chest radiography on a corpus where
the confound cannot exist, and the honest 96–98% there replaced a dishonest 99% that
would not have survived a single competent question.

**This is the most transferable lesson in the project.** Ninety-nine percent is not a
result. Ninety-nine percent *against a measured hurdle, under a patient-disjoint
split, with the shortcut baseline reported alongside it* is a result.

---

## 5. Feature engineering: the heart-sound model

### 5.1 Why a time–frequency representation

A murmur is defined by two things at once: **when** it happens inside the cardiac
cycle, and **what frequencies** it contains. Aortic stenosis produces turbulence
during ejection; mitral stenosis produces a low rumble during filling. A plain
frequency spectrum throws away the timing. A raw waveform throws away the pitch
structure the network needs. A spectrogram keeps both, so that is what the model
sees.

The mel scale spaces the frequency axis the way hearing does — fine resolution low
down, coarser higher up. Heart sounds live between roughly 20 and 600 Hz, so this
spends the model's resolution exactly where the signal is.

### 5.2 The front end, in full

| Stage | Setting |
|---|---|
| Sample rate | 8,000 Hz (native; no resampling) |
| Analysis window | 3.0 s = 24,000 samples |
| Band-pass | Butterworth, 20–800 Hz, order 4 (two second-order sections) |
| Filter application | **Causal, forward-only** |
| FFT size | 256 (32 ms) |
| Hop | 128 (16 ms) |
| Window function | Hamming |
| Mel filterbank | 40 triangular bands, area-normalised |
| Frames | 186 produced, zero-padded to 192 |
| Compression | natural log, floor 1e-6 |
| **Model input** | **40 × 192, single channel** |

The filterbank is written out explicitly rather than called from a library, so the
exact same table is used everywhere the features are computed. There is one
definition of "the features", and everything reads it.

### 5.3 The one rule that is non-negotiable

**The filter used at training time must be the same filter used at inference time.**

The tempting choice is zero-phase filtering, which removes phase distortion by
running the filter forwards and then backwards. It is better filtering, and it is
useless here, because it is non-causal — it needs the whole recording, including the
future, before it can produce the first output sample. Anything computing features
incrementally cannot do it.

If training uses zero-phase filtering and inference uses a causal filter, the two
pipelines produce different features from the same audio, and accuracy collapses at
inference for no visible reason. This is textbook training–serving skew, and it is
invisible: nothing errors, the model just quietly gets worse.

So the training pipeline uses the causal filter, deliberately accepting slightly
worse filtering in exchange for the two pipelines being identical.

---

## 6. Feature engineering: the chest X-ray model

Almost none, and that is intentional.

| Stage | Setting |
|---|---|
| Colour | Convert to greyscale |
| Crop | Centre-crop to square **before** resizing |
| Resize | Bilinear, to 96 × 96 |
| Scale | Divide by 255 → [0, 1] |
| Histogram equalisation | Tried and rejected |
| **Model input** | **96 × 96, single channel** |

Two decisions are worth explaining.

**Centre-crop before resize.** Chest films in this corpus range from 384 × 127 to
2916 × 2713. Resizing those to a square directly stretches the anatomy differently
for every image — the model would have to learn to undo an arbitrary aspect-ratio
distortion before it could learn anything about lungs. Cropping to square first keeps
the lung fields and leaves the geometry consistent.

**No per-image normalisation.** Every image is scaled by the same constant, so
brightness levels remain comparable *across* images. This costs a little robustness
and buys something more valuable: intensity-based measurements computed later
(Section 12) are meaningful, because a value of 140 means the same thing in every
image.

Convolutional layers are extremely good at learning their own low-level features from
pixels. Hand-designed image features would have added complexity and thrown
information away.

---

## 7. The two architectures

### 7.1 Heart sounds — a small depthwise-separable CNN, trained from scratch

Input 40 × 192 × 1.

| # | Layer | Channels |
|---|---|---|
| 1 | Conv2D 3×3, stride 2, BatchNorm, ReLU6 | 16 |
| 2 | SeparableConv2D 3×3, stride 2, BatchNorm, ReLU6 | 32 |
| 3 | SeparableConv2D 3×3, stride 2, BatchNorm, ReLU6 | 64 |
| 4 | SeparableConv2D 3×3, stride 2, BatchNorm, ReLU6 | 128 |
| 5 | Average pool over the full remaining map | — |
| 6 | Dropout 0.3 | — |
| 7 | 1×1 Conv2D → 5, Softmax | 5 |
| | **Total parameters** | **13,509** |

**Why so small?** Because it saturates the task. An earlier specification called for
roughly 41,000 parameters; the narrower model matched it. Extra capacity that does
not improve a measured number is not free — it is more to overfit with and more to
explain.

**Depthwise-separable convolution** factors a standard convolution into a per-channel
spatial filter followed by a 1×1 mixing convolution. It gives most of the
representational power for a fraction of the multiply-accumulates, which is the right
trade for a small model on a small input.

**One implementation detail worth knowing.** The classifier head is written as a 1×1
convolution on the (1, 1, channels) tensor rather than the more obvious
`Reshape → Dense`. The two are numerically identical, but the reshape version makes
the framework compute the reshape target at runtime, which puts shape-manipulation
operations into the exported graph and makes the whole thing dependent on dynamic
shapes. Rewriting the head as a 1×1 convolution removed all of them and made the
exported model smaller — 28.6 kB down to 27.7 kB — for exactly zero change in what it
computes.

### 7.2 Chest X-rays — transfer learning

Input 96 × 96 × 1.

| # | Stage |
|---|---|
| 1 | Fixed 1×1 convolution, constant weights, **not trainable** — replicates the single greyscale plane into 3 channels |
| 2 | MobileNetV2 backbone, width multiplier α = 0.35, ImageNet-pretrained, no classifier top |
| 3 | First 40 layers frozen; everything above fine-tuned |
| 4 | Average pool over the full remaining map |
| 5 | Dropout 0.3 |
| 6 | 1×1 Conv2D → 2, Softmax |
| | **Total parameters: 412,773** |

**Why pretrained weights are the biggest legitimate lever available.** The backbone
arrives already knowing edges, textures and shapes from a million natural images.
5,856 chest films are not enough to learn general vision from scratch — but they are
plenty to learn a decision boundary on top of vision that already exists. Measured
against the from-scratch model on the identical protocol, transfer learning plus
augmentation was worth **+6.2 points** at image level.

**The greyscale trick.** ImageNet weights expect three input channels. Rather than
storing three copies of every image, a fixed 1×1 convolution with constant weights
replicates the single plane into three. It is not learned, it costs nothing, and the
data stays single-channel everywhere it is stored.

### 7.3 Why not one shared model with two heads

They are different modalities with different input shapes, different amounts of
training data, and completely different failure modes. Keeping them separate means
they are independently trainable and independently testable: retraining the lung
model cannot silently regress the heart model. That property is worth more than any
parameter sharing would have bought.

---

## 8. The training protocol

The same protocol runs both models. The protocol is the product; the models are what
falls out of it.

### 8.1 Cross-validation, not a single lucky split

Five folds. For the X-ray corpus every fold is patient-disjoint; for the heart corpus
every fold is disjoint at the level of the recording, which is the finest granularity
the corpus supports. Every reported figure is the mean across folds, with the
standard deviation printed next to it — because a single hold-out number on a small
corpus is a coin flip dressed up as a result.

### 8.2 Nothing from the test fold touches the training pipeline. Ever.

| Quantity | Fitted on |
|---|---|
| Normalisation mean and standard deviation | training fold only |
| Class weights | training fold label counts only |
| Quantisation calibration set (Section 10) | training fold only |
| Augmentation | training generator only — never applied to evaluation data |

The normalisation rule catches people constantly. Computing the mean and standard
deviation over the whole dataset before splitting leaks test-set statistics into
training. It is a small leak, it inflates results, and it is completely avoidable.

### 8.3 Optimisation

| Setting | Value |
|---|---|
| Optimiser | Adam |
| Loss | Categorical cross-entropy, **label smoothing 0.05** |
| Class balancing | Per-sample weights, inverse to class frequency |
| Learning-rate schedule | Reduce on plateau — factor 0.5, patience 12, floor 1e-5 |
| Regularisation | Dropout 0.3 before the head; augmentation |
| Batching | Infinite shuffled generator, augmentation applied per batch |

**Label smoothing** replaces hard 0/1 targets with slightly soft ones. It discourages
the network from driving its outputs to saturation, which matters here for a specific
reason: an over-confident model makes the confidence threshold in Section 11
meaningless. We want calibrated confidence, not maximal confidence.

**Class weights as per-sample weights.** The framework rejects the usual class-weight
argument when training from a generator, so the weights are emitted per sample
instead. Mathematically identical, and worth mentioning because silently dropping
class balancing on an imbalanced corpus is an easy and expensive mistake.

### 8.4 Augmentation, decided separately for each modality

This is a place where copying a recipe from one modality to the other would have been
wrong.

**Audio — deliberately mild, and ablated.** Additive gain jitter in the log domain
(equivalent to a multiplicative gain on the signal), plus light SpecAugment-style
frequency and time masking. It is mild because published work found that heavy
augmentation *degrades* lung-sound classification. We did not take that on trust — the
pipeline has an ablation flag, and **the heart-sound figures reported in this document
come from the run with augmentation switched off**, because that is what measured
better.

**Images — aggressive, because the variation is real.** Rotation ±10°, zoom
0.90–1.10, translation ±8%, exposure gain and offset ±10%. Patient positioning, zoom
and exposure genuinely vary between real films, so these transforms generate
plausible new examples rather than corrupting the signal.

**One augmentation is deliberately absent: horizontal flip.** The heart sits on the
left. Mirroring a chest film manufactures dextrocardia — a real but rare condition —
and teaches the model that left–right position carries no information. It is the
single most commonly applied image augmentation, and it is wrong here.

### 8.5 Two levels of evaluation

The X-ray model is scored twice:

- **Image level** — one prediction per radiograph.
- **Patient level** — the softmax outputs of all of a patient's images are averaged,
  and the argmax of that average is the patient's prediction.

Patient level is the number that matters, because a clinician gets one answer per
patient, not one per image. It is also the level at which the published literature
reports, so our number stands beside theirs rather than being quietly more
favourable.

---

## 9. Results

### 9.1 Heart sounds

| Metric | Value |
|---|---|
| **5-fold cross-validated accuracy** | **98.89% ± 1.14%** |
| Window-level cross-validated accuracy | 98.67% ± 1.03% |
| Held-out fold, full precision | 98.06% |
| Held-out fold, 8-bit | **98.06%** |
| Chance | 0.200 |
| Margin over chance | **+78.9 points** |

Per-class, on the held-out fold (990 windows):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 1.000 | 1.000 | 1.000 |
| Aortic Stenosis | 1.000 | 0.990 | 0.995 |
| Mitral Stenosis | 0.995 | 0.979 | 0.987 |
| Mitral Regurgitation | 0.957 | 1.000 | 0.978 |
| Mitral Valve Prolapse | 0.995 | 0.975 | 0.985 |

Normal is separated perfectly. The only meaningful confusion is Mitral Valve Prolapse
and Mitral Stenosis being drawn towards Mitral Regurgitation — which is exactly the
confusion a clinician would predict, since all three are mitral-valve lesions. A
confusion structure that matches physiology is weak evidence that the model learned
something real rather than something incidental.

**How this number should be read.** It is high because the corpus is clean and
balanced, not because the problem is solved. Bedside recordings are considerably
harder. Section 13 is about measuring exactly that gap instead of ignoring it.

### 9.2 Chest X-rays

The from-scratch model is included because the comparison is the point.

| | From scratch | **Transfer learning + augmentation** |
|---|---|---|
| Full precision, image level | 0.9031 | **0.9661** |
| Full precision, patient level | — | **0.9803** |
| 8-bit, image level | 0.9015 | **0.9637** |
| 8-bit, **patient level** | — | **0.9785** |
| Parameters | 13,122 | 412,773 |
| Model size | 27.2 kB | 610.4 kB |

Per-class recall after quantisation: **Normal 0.960, Pneumonia 0.965.** The
from-scratch model's weak Normal recall — 0.868, pooled across folds — is fixed, and
fixed by improving both classes rather than trading one against the other.

The from-scratch model's pooled cross-validated confusion, at patient level:

| | predicted Normal | predicted Pneumonia |
|---|---|---|
| **Normal** (1,116) | 969 | 147 |
| **Pneumonia** (1,674) | 64 | **1,610** |

Note the shape of the errors even in the weaker model: far more false alarms than
missed pneumonia. For a screening tool that is the correct direction to fail in, and
it is worth checking rather than assuming.

---

## 10. Compression: 8-bit post-training quantisation

Both models are converted to full-integer 8-bit arithmetic after training.

### 10.1 The method

Standard post-training quantisation. Weights and activations are mapped to 8-bit
integers using a scale and zero-point per tensor, derived by running a small
**calibration set** through the trained model and observing the range each tensor
actually takes.

### 10.2 The calibration set is where this goes wrong

Two rules, and both matter more than the conversion itself:

1. **It comes from the training fold only.** Test data must never touch the
   quantiser. Calibrating on test data is leakage — quieter than most, and just as
   invalidating.
2. **It must span every class and the full loudness range.** The sample is drawn
   class-balanced, then sorted by mean energy so it spans quiet to loud rather than
   clustering wherever the corpus happens to be dense. Calibrate only on loud
   examples and the activation ranges will be wrong for quiet ones — which shows up
   as a mysterious accuracy loss on exactly the rare classes you care about.

Calibration sets used: 300 windows for the heart model, 120 images for the X-ray
model.

### 10.3 Results, and the metric that actually matters

| | Heart sounds | Chest X-rays |
|---|---|---|
| Full precision | 98.06% | 96.61% image / 98.03% patient |
| 8-bit | **98.06%** | **96.37% image / 97.85% patient** |
| Accuracy change | **0.00 points** | −0.24 image / −0.18 patient |
| **Prediction agreement** | **100.00%** | **99.27%** |

**Agreement is the number to look at, not accuracy.** Two models can score identically
while disagreeing on a quarter of the cases — one gets a case right that the other
gets wrong, and it cancels out in the average. Agreement asks a stricter question:
*on how many individual cases do the full-precision and 8-bit models produce the same
answer?* 100% agreement means the quantised heart model is not merely as good as the
original, it is making the same decisions on every case.

Four times smaller for a fifth of a point is an extremely good trade. If the drop had
exceeded about two points, the correct response would have been quantisation-aware
training, not accepting the loss.

---

## 11. Teaching the models to abstain

A softmax classifier always returns a class. Given a photograph of a wall, it will
return a heart valve lesion, and it will not look worried. For a screening tool used
by a non-specialist, "I do not know" is a far more useful output than a confident
wrong answer.

### 11.1 A confidence floor

Predictions below a confidence threshold are reported as **uncertain** rather than as
a class. The operating point is 0.70. The behaviour as it is tightened is the
important property:

Raise the threshold to 0.99 and the X-ray model abstains on **114 of 120** held-out
images rather than guessing. Coverage collapses; the answers that survive stay
correct. That is the right failure mode — the model degrades into silence, not into
confident error.

### 11.2 Where a confidence floor is not enough — a negative result

We fed both models eight degenerate inputs: all black, all white, mid grey, uniform
noise, salt-and-pepper, gradients, checkerboards.

| Model | Degenerate inputs answered confidently |
|---|---|
| Heart sounds | 2 of 8 |
| Chest X-rays | **8 of 8** |

The X-ray model assigns confidence up to 0.996 to pure noise. **A confidence floor is
therefore not an out-of-distribution detector for this model** — the model is
confidently wrong, not uncertainly wrong, so no threshold on its own confidence can
catch it. That finding is why out-of-distribution detection is treated as a separate
mandatory component rather than as something the softmax gives you for free, and it
is why we measured the shift directly in Section 13.

---

## 12. Interpretability, and three experiments that failed

Alongside each classification, the system computes secondary measurements. Every one
of them is **classical signal or image processing, or a probe of the already-trained
model.** None is a second neural network. Each extra network would be more to train,
more to validate, and one more thing capable of being confidently wrong.

The rule applied throughout: **each indicator was validated against the labels before
it was allowed on screen.** Three failed that test. All three are documented here,
because the negative results were more useful than the positive ones.

### 12.1 Cardiac cycle segmentation — works

Within a heartbeat the gap from the first sound to the second (systole) is *shorter*
than the gap from the second back to the first (diastole). So the sequence of
inter-peak gaps splits into two interleaved groups, and the group with the smaller
mean is systole. No labels, no learning, no training data: a Shannon-energy envelope,
peak detection, and an alternating-gap rule.

### 12.2 Murmur timing — works, and matches physiology

Murmur energy is measured as sustained energy where there should be relative quiet:
the RMS in the interior of each gap, trimmed 25% at each end to exclude the heart
sounds themselves, expressed relative to the heart sounds' own RMS.

Measured across the 53 held-out cases:

| Class | Systolic ratio | Diastolic ratio | Physiology expects |
|---|---|---|---|
| Normal | 0.001 ± 0.001 | 0.006 ± 0.017 | neither |
| Aortic Stenosis | **0.859 ± 0.131** | 0.078 ± 0.161 | systolic |
| Mitral Stenosis | 0.058 ± 0.043 | **0.124 ± 0.046** | diastolic |
| Mitral Regurgitation | **0.486 ± 0.170** | 0.019 ± 0.021 | systolic |
| Mitral Valve Prolapse | **0.320 ± 0.252** | 0.080 ± 0.120 | systolic |

The measurement recovers the correct phase for all five classes, including the
diastolic predominance that defines mitral stenosis. Converted to a call with a
threshold of 0.06: **49 of 52 cases match physiology.**

**A calibration failure worth recording.** The first threshold tried was 0.18. That
sits *above* the mitral-stenosis diastolic ratio, so all nine mitral-stenosis cases
were reported as "no murmur" — for the one class whose defining sign is a diastolic
murmur. The measurement had been right the whole time; the cut-off was wrong. This is
precisely why every indicator is validated against labels before display, and why an
output that looks plausible is not evidence of an output that works.

### 12.3 Occlusion sensitivity on the X-ray model — works

Blank each cell of a 3×3 grid in turn, re-run the quantised model, and measure how
far the confidence drops. Where it drops most is where the model was looking. A 3×3
grid is deliberately coarse: at 96 × 96, claiming finer localisation would be false
precision.

Where the model attends, by true class:

| Class | Upper | Middle | Lower |
|---|---|---|---|
| Normal | **36** | 19 | 5 |
| Pneumonia | 9 | **43** | 8 |

An interpretable, clinically sensible difference — for pneumonia the model attends to
the mid-lung fields — obtained from the existing model with no retraining and no
gradients, which means it works on the quantised graph exactly as shipped.

### 12.4 Failure one: occlusion by cardiac phase — withdrawn

**The idea.** Blank the systolic frames of the input spectrogram, then the diastolic
frames, re-run the model, and see which phase the prediction depends on.

**The result.** It answered "systole" for every class, Normal included. Normalising
the confidence drop per occluded frame — diastole is simply longer — barely helped.

**The interpretation.** It measures where the signal *energy* is, not where the
diagnostic *information* is. It is retained in the case record with an explicit
`"discriminative": false` flag and is never displayed. The signal-processing phase
ratio in 12.2 is used instead, because that one demonstrably works.

### 12.5 Failure two: opacity burden by percentile — withdrawn as circular

The first implementation thresholded each image at its own 70th percentile and
reported the fraction of pixels above it. That is tautological: taking the top 30% of
pixels guarantees a "burden" near 30% regardless of what the image contains. Measured
output: Normal 29.5%, Pneumonia 29.3%. The statistic was reporting its own threshold.

It was replaced with an absolute cut on the shared intensity scale — which is
available precisely because Section 6 declined to normalise each image individually.

### 12.6 Failure three: simple image statistics do not separate the classes

Measured over the 120 held-out images:

| Statistic | Normal | Pneumonia | Verdict |
|---|---|---|---|
| Mean intensity | 138.8 ± 12.2 | 140.8 ± 20.5 | no separation |
| Lower-zone mean | 144.7 ± 14.5 | 149.1 ± 18.5 | no separation |
| Dense area, absolute threshold | 14.5 ± 9.3% | 14.5 ± 15.2% | no separation |
| **Texture heterogeneity** | **40.1 ± 4.6** | **34.8 ± 7.2** | weak, ~0.9 SD |

Only texture shows any signal at all — consistent with consolidation filling in
normally-aerated, high-contrast lung — and it overlaps far too much to decide
anything alone.

**The honest conclusion: at 96 × 96 the network separates these classes at 96% and
simple thresholding cannot.** The descriptive statistics are shown as *descriptions
of the image*, with an explicit note that they do not detect pneumonia. The
temptation to present a plausible-sounding number as if it were diagnostic is exactly
what this project spent its time refusing.

---

## 13. Measuring distribution shift instead of assuming it away

Both models were trained on curated public corpora. Real screening data is not
curated. The interesting question is not *whether* there is a distribution gap — of
course there is — but whether it can be **measured**, so the system can act on it.

We collected 50 heart recordings from 10 people under written consent. They have not
been used to train or evaluate any model, and no accuracy figure anywhere in this
project depends on them. Their entire purpose is to characterise the gap.

### 13.1 Finding a statistic that separates the two sources

Recordings made through a different acoustic chain carry more energy above 1 kHz —
handling noise, room tone, coupling differences — than the curated corpus does.
Measuring the fraction of total energy above 1 kHz on both sets:

| Set | Flagged by a threshold of 0.0015 |
|---|---|
| Training corpus | **0%** |
| Field recordings | **82%** |

A single scalar, computed with an FFT and no learning, separates the two sources
cleanly enough to act on.

### 13.2 What the system does with that

For recordings from the unvalidated source, the system **shows the operator the
spectrogram and asserts no class at all.**

This is worth being precise about, because it is a deliberate product decision made
on measured evidence. The model would happily return one of five valve lesions with a
high confidence value. That number would be meaningless — not because the model is
bad, but because it has never been evaluated on this signal chain, and an unvalidated
class presented on a screen labelled "analysis" is a claim we cannot support. So the
system declines, and shows the signal instead.

Closing that gap — characterising the new acoustic chain and either adapting or
retraining — is the top item of future work. It is stated as the project's headline
limitation rather than buried.

---

## 14. What we refused to model

Every item below is something a user could plausibly ask this system for. None is
computed, and the interface says so explicitly.

| Asked for | Why it is not computed |
|---|---|
| Valve disease **severity**, valve area, pressure gradient | Requires echocardiography. Not inferable from sound, at any accuracy |
| Cardiomegaly | Needs cardiac borders and a cardiothoracic ratio; the centre-crop does not reliably contain them |
| Pleural effusion | Needs the costophrenic angles, which are cropped out |
| Pneumothorax | Needs a pleural line, far below 96 × 96 resolution |
| Pulmonary oedema, congestion, atelectasis | **The model has no training data for these.** A binary pneumonia classifier cannot diagnose them, and no descriptive statistic can either |
| Murmur shape (crescendo, decrescendo) | Envelope shape is unreliable at this signal-to-noise ratio |
| Bacterial vs viral pneumonia | The corpus's subject identifiers cannot support a defensible three-class split (Section 3.2) |

**The line that must not be crossed:** "the input contains a measurable feature" is
not "the model can diagnose a disease". Everything above is a place where those two
statements could be confused, so each one is named and refused.

---

## 15. Honest limitations

1. **The heart corpus is curated.** Clean, balanced teaching recordings. 98.9% on it
   is a real number about that corpus and an optimistic number about a clinic.
2. **The heart split is recording-disjoint, not verified patient-disjoint,** because
   the corpus publishes no patient identifiers. It is the strongest split available,
   and it is weaker than the phrase usually implies.
3. **No clinical validation.** Every figure in this document comes from public
   held-out data. There is no measured sensitivity or specificity on real patients.
4. **The X-ray corpus is one hospital, one paediatric cohort.** That is what makes it
   confound-free, and it is also exactly why generalisation to other hospitals,
   scanners and age groups is unproven.
5. **Transfer to a new acoustic chain is unproven**, quantified in Section 13, and the
   reason the system currently declines to classify those recordings.
6. **The X-ray model has no usable uncertainty signal of its own** (Section 11.2), so
   out-of-distribution detection has to come from outside the model.
7. **The reported spread is across folds, not across seeds.** A multi-seed run would
   give a better estimate of variance than a single seed's fold spread does.
8. **Subject grouping in the X-ray corpus is conservative and imperfect** — it may
   merge two children into one subject. That direction cannot cause leakage, but it
   does mean the patient count is approximate.

---

## 16. Reproducibility

Every number in this document comes from a stored artefact rather than from a
notebook someone ran once.

| Artefact | What it holds |
|---|---|
| Cross-validation records | Per-fold accuracy, macro-F1, mean, standard deviation, confusion matrices, fold count, augmentation flag |
| Quantisation records | Full-precision accuracy, 8-bit accuracy, delta, **prediction agreement**, calibration set size, per-tensor scales and zero-points, operator list |
| Manifests | Every recording and image with its subject identifier, label and fold assignment |
| Front-end definitions | One dataclass per modality, printable, carrying every parameter in Sections 5 and 6 |
| Descriptor outputs | Every secondary measurement per case, including the withdrawn ones, flagged non-discriminative |

The two front ends are each defined exactly once, in one place, and every stage that
computes features reads that definition. There is no second copy of "what the
features are" to drift out of sync — which is the same discipline as the causal-filter
rule in Section 5.3, applied to the whole pipeline.

---

## In one paragraph

Two supervised classifiers: a 13,509-parameter convolutional network reading log-mel
spectrograms of heart sound at five classes, cross-validating at 98.89% ± 1.14%, and a
412,773-parameter fine-tuned MobileNetV2 reading 96 × 96 chest radiographs at two
classes, reaching 97.85% per patient after 8-bit quantisation with 99.27% agreement
against its full-precision original. Both are patient-disjoint wherever the corpus
permits it, both are calibrated only on training data, both abstain rather than guess
below a confidence floor, and both are annotated by classical signal and image
processing rather than by further networks. Along the way an entire task was deleted
because a classifier reading the stethoscope brand off the filename beat our network,
two interpretability measures were withdrawn for measuring artefacts of their own
construction, and the gap between curated corpora and real recordings was measured
rather than assumed away. **The models are ordinary. The evaluation protocol is the
contribution.**
