# CardioNary — Question and Answer Preparation

**For the judging session. Plain language, short answers, every number real.**

Read the rule at the top of §0 before anything else. If you remember nothing else
from this document, remember that.

---

## 0. Three rules for the room

**1. Never state a number you cannot point at.** Every figure here was measured on
the hardware sitting in front of you. If you are asked something not in this
document, say *"we did not measure that"*. That answer costs nothing. A guessed
number that a judge checks costs everything.

**2. Agreement, not accuracy, is our headline.** Accuracy says how good the model
is. Agreement says whether the chip computes what the trainer computed. Judges who
know embedded ML care about the second one, and ours is perfect on cardiac.

**3. When we chose not to do something, say so plainly.** We killed our own lung
audio task on evidence. That is a strength, not an admission. Tell it as a
decision, not an apology.

---

## 1. What is the product

**CardioNary — a cardiopulmonary screening station.**

Two small microcontroller boards, each running a neural network on the chip
itself. One listens to heart sounds. One reads chest X-rays. They send results
over Bluetooth Low Energy to a Raspberry Pi, which shows them on a web page any
browser can open.

**One sentence:** *A screening aid that runs entirely on two microcontrollers,
with no cloud, no internet, and no app to install.*

**What it is not:** a diagnosis. It is a screening aid, and every result is for a
clinician to review. Say this early and mean it.

---

## 2. The problem, and who has it

**The problem.** A stethoscope examination is a skill. It varies between
examiners, it is never recorded, and a soft murmur missed in a busy clinic is not
caught later — it is caught years later, as heart failure. Chest X-rays in small
clinics often wait for a radiologist who is not there.

**Two targets:**

- **Pneumonia in children under five.** It is the leading infectious cause of
  death in that age group, and those deaths cluster after the newborn period.
  Our imaging model is trained on exactly that age band, one to five years.
- **Valvular heart disease**, including the mitral damage left by rheumatic
  fever — still common wherever untreated strep throat is common.

**Who it is for.** Primary health centres, rural clinics, and screening camps —
places with a nurse or a general physician but no cardiologist and no
radiologist. The operator is the person holding the stethoscope, not an engineer.

**Where we identified it.** India has a very high burden of both conditions and a
severe shortage of specialists outside cities. The gap is not diagnosis — it is
*triage*: deciding who needs to be sent onward.

---

## 3. Why edge AI, and why we are edge-AI native

**Why not cloud:**

1. **The clinics that need this most have the worst connectivity.** A system that
   needs a network fails exactly where it is needed.
2. **Privacy is architectural.** Raw audio and raw images never leave the
   microcontroller. There is nothing to intercept because nothing is sent.
3. **Cost.** No servers, no bandwidth, no per-inference fee. There is no
   marginal cost to pass on.
4. **Latency.** 57 ms on the chip. No round trip.

**Why we are genuinely edge-native, not "edge-flavoured":**

The complete inference happens on the microcontroller — feature extraction and
the neural network both. The Raspberry Pi does **no** classification. It moves
bytes and draws a web page. If you unplug the Pi, the boards still classify and
still print results over their serial port.

> **If a judge presses:** "prove nothing runs on the Pi." Point at the boot log
> on the serial console: the board loads the model, reports its own memory arena,
> and prints each result before any of it reaches the Pi.

---

## 4. The two models

| | Cardiac | Pulmonary |
|---|---|---|
| Board | xG26 Dev Kit, BRD2608A | xG26 Explorer Kit, BRD2709A |
| Chip | EFR32MG26, Cortex-M33 at 78 MHz | same |
| Task | 5 classes of heart sound | pneumonia / not pneumonia |
| Architecture | Custom small CNN | MobileNetV2, width 0.35 |
| Parameters | **13,509** | **412,773** |
| Model size (INT8) | **27.7 kB** | **610 kB** |
| Input | 40 x 192 log-mel, from 3 s at 8 kHz | 96 x 96 greyscale |
| **RAM used while running** | **48,412 bytes** | **177,252 bytes** |
| Inference time | **57 ms** | **644 ms** |
| Feature extraction time | 201 ms | 2 ms |
| Dataset | Yaseen (heart sounds) | Kermany (paediatric chest X-ray) |

Five cardiac classes: **Normal, Aortic Stenosis, Mitral Stenosis, Mitral
Regurgitation, Mitral Valve Prolapse.**

### Why these architectures

**Cardiac — small custom CNN.** The input is a 40 x 192 spectrogram, which is a
small image with strong local structure. A handful of convolution layers is
enough. We deliberately kept it at 13.5 k parameters because a bigger network
gave no measurable improvement, and every extra parameter costs flash, RAM and
milliseconds on a device that has to fit a Bluetooth stack alongside it.

**Pulmonary — MobileNetV2 at width 0.35.** Chest X-rays need more capacity than
heart sounds: the useful signal is texture spread across the whole lung field,
not a short repeating pattern. MobileNetV2 uses depthwise-separable convolutions,
which give most of the accuracy of a full convolution for a fraction of the
multiply-accumulates — exactly the trade a microcontroller needs. Width 0.35 is
the smallest setting that still cleared our accuracy gate.

**The honest general answer:** *the architecture follows the data.* A short
1-D-ish repeating signal needs a small network. A full radiograph needs a real
vision backbone. We did not pick a model and then look for a problem.

---

## 5. Features — exactly what each model looks at

This is a favourite question. Be precise.

### Cardiac — time-frequency domain

The board computes a **log-mel spectrogram** on the chip using CMSIS-DSP:

| Step | Detail |
|---|---|
| Input | 3.0 s of audio, 8 kHz, 24,000 samples |
| Window | Hamming |
| Transform | Short-Time Fourier Transform |
| Mel filterbank | **40 mel bands** |
| Frames | **192** |
| Output | 40 x 192 matrix, log scale, quantised to INT8 |

**Domain: time-frequency.** Not time domain, not frequency domain — both at once.
That is the whole point: a murmur is defined by *when* in the cardiac cycle it
occurs as well as *what frequencies* it contains. A pure FFT would lose the
timing. A pure waveform would lose the pitch.

### What a mel spectrogram is (simple version)

Take the sound. Chop it into short overlapping slices. For each slice, work out
how much energy sits at each frequency. Stack those slices side by side and you
get a picture: time across, frequency up. **Mel** means the frequency axis is
spaced the way human hearing works — fine detail at low frequencies, coarser
higher up. Heart sounds live at low frequencies, so this spacing spends its
resolution where the signal is.

> **Which spectrogram is on the screen?** The one on the dashboard is a **plain
> STFT log-magnitude spectrogram, 0–1000 Hz**, drawn from the same audio the chip
> classified. It is for the operator to look at. The model's own input is the
> 40-band **mel** version computed on the board. Do not claim the screen is
> showing the model's input — it is a display of the same signal.

### Pulmonary — spatial domain

No transform at all. The image is resized to 96 x 96 greyscale and the firmware
applies one operation: `int8_value = pixel - 128`. The convolution layers learn
their own features from raw pixels.

**Domain: spatial.** Convolution kernels learn edges, then textures, then the
opacity patterns that distinguish consolidation from clear lung.

### Secondary (derived) features — and the logic behind them

Beyond the class, the cardiac path also reports **whether a murmur is present and
whether it falls in systole or diastole.**

**How, and why it is legitimate:** this is *classical signal processing on the
same audio*, not a second neural network. We find S1 and S2 by envelope peak
detection, which gives us the cardiac cycle boundaries. Energy between S1 and S2
is systolic; energy between S2 and the next S1 is diastolic. Comparing those
energies against the total tells us whether a murmur is present and where.

**Why bother:** it turns "Mitral Stenosis, 94%" into something a clinician can
check against physiology. Mitral stenosis produces a *diastolic* murmur. If the
model says mitral stenosis and the timing says systolic, something is wrong, and
the operator can see that.

**How well it works: it matches physiology on 49 of 52 cases.** That is measured,
and we quote it as 49/52, never as "about 94%".

For the pulmonary model we also compute where the network is looking, using
occlusion attribution — blank out a patch, see how much the output changes.
Normal images draw attention to the upper field in 36 of 60 cases; pneumonia
images to the middle field in 43 of 60.

---

## 6. Accuracy, and the two kinds of it

**This is the most important technical section. Learn it.**

| Model | Accuracy on device | Agreement with the trained model |
|---|---|---|
| Cardiac | **53/53 = 100.00%** | **53/53 = 100.00%** |
| Pulmonary | **115/120 = 95.83%** | **118/120 = 98.33%** |

Cross-validated cardiac performance: **98.9% ± 1.1%.**

### The difference, explained simply

- **Accuracy** = did the device get the right answer, against the true label?
  Bounded by how good the model is.
- **Agreement** = did the device produce the *same* answer as the same model
  running on a PC, given byte-identical input?

**Agreement has a known correct answer: 100%.** Accuracy does not. If agreement
is below 100%, something in the port is broken — the DSP, the quantisation, the
kernels. That is why we measure it.

### How we measured both

1. Hold out data the model never trained on — 53 heart sounds, 120 X-rays.
2. Run them through the model **on a PC** and record every prediction. This goes
   in a manifest file.
3. Compile those exact same bytes into the firmware as "golden vectors".
4. Run them **on the chip** and send every result back over Bluetooth.
5. Compare, case by case.

**Software accuracy** comes from step 2. **Hardware accuracy** comes from step 4.
**Coherence between them is the agreement number.**

### Why cardiac is 100% and what to say about it

Say this: *"100% on 53 held-out cases. Cross-validated it is 98.9% ± 1.1%. Yaseen
is a clean, balanced teaching corpus, so this is an easier task than bedside
audio would be, and we say so in our report."*

Do not round it down and do not apologise for it. Give the cross-validated figure
in the same breath and you have answered the objection before it is made.

### The two pulmonary cases where device and PC disagree

Both are known and reproducible:

| Vector | PC says | Device says |
|---|---|---|
| 3 | Normal | **UNCERTAIN** — confidence 0.514, just below our reporting floor |
| 25 | Pneumonia | **Normal** — confidence 0.549 |

Both sit right at the decision boundary. This is expected INT8 rounding, not a
bug: a case the model is barely sure about can tip either way when the arithmetic
is 8-bit. It is confined to low-confidence cases, which is exactly where it
should be.

---

## 7. Memory and hardware efficiency

**They will ask how much RAM the model uses while running. Know these numbers.**

| | Cardiac | Pulmonary |
|---|---|---|
| **Tensor arena (RAM at inference)** | **48,412 B** | **177,252 B** |
| Total chip RAM | 512 kB | 512 kB |
| Model in flash | 27.7 kB | 610 kB |
| Total chip flash | 3,200 kB | 3,200 kB |
| Inference | 57 ms | 644 ms |

The **tensor arena** is the single block of RAM TensorFlow Lite Micro uses for
all intermediate activations. The device measures and reports its own arena at
boot — we do not estimate it. It prints on the serial console every time.

**Headroom:** the larger model uses about a third of available RAM and under a
fifth of flash, alongside a full Bluetooth stack. Nothing here is squeezed.

**Accelerator:** both chips carry Silicon Labs' **MVP** (Matrix Vector Processor),
and our convolution layers run on it.

---

## 8. Why chest X-ray for lungs, and the decision we reversed

**Be ready for this. It is our best story.**

We originally planned lung *sounds*. We ran five patient-disjoint experiments and
found that public lung-audio datasets are contaminated: **ICBHI recordings are 77%
identifiable by the recording device alone.** We then trained a baseline that was
shown *only which device made the recording* and nothing else.

**That device-only baseline scored 0.675. Our actual neural network scored
0.643.** A model that knew nothing about lungs beat the model that did.

That means a good accuracy number on that data would be measuring the microphone,
not the disease. So we withdrew the task rather than publish it.

We moved to the **Kermany paediatric chest X-ray corpus**, where both classes come
from one hospital and one imaging pipeline — so that particular confound cannot
exist by construction.

**Say it like this:** *"We deleted our own result because it was measuring the
wrong thing. That is why we trust the number we do report."*

---

## 9. Datasets and demographics

| | Cardiac | Pulmonary |
|---|---|---|
| Corpus | Yaseen heart sound | Kermany chest X-ray |
| Classes | 5 | 2 |
| Population | Adult, curated teaching recordings | **Paediatric, ages 1–5** |
| Source | Public research dataset | Single hospital, single imaging pipeline |
| Split | Patient-disjoint | Patient-disjoint, held-out fold |

**Splitting is patient-disjoint** — no patient appears in both training and test.
Without that, the model can memorise a person rather than a disease and the
accuracy is meaningless.

**Demographic limits, state them honestly:** the cardiac corpus is adult and
curated; the pulmonary corpus is paediatric and from one institution. Neither is
Indian data. Performance on an Indian clinic population is **unmeasured**, and
that is exactly what our own data collection is for.

---

## 10. How we trained

1. **Select and audit the dataset**, including the leakage checks in §8.
2. **Split patient-disjoint**, so no subject crosses the boundary.
3. **Train in TensorFlow/Keras** in float32.
4. **Quantise to INT8** using post-training quantisation with a representative
   calibration set drawn from training data only.
5. **Verify quantisation offline** — compare float and INT8 predictions on the PC.
6. **Compile for the device** with the Silicon Labs AI/ML extension, which
   generates MVP-accelerated kernels.
7. **Export golden vectors** — held-out cases plus their offline predictions.
8. **Run on hardware and measure agreement.** This is the gate. Anything below
   100% agreement means the port is wrong and gets debugged before anything else.

**Why INT8:** an 8-bit integer model is four times smaller than float32, and the
MVP accelerator works on integers. Without quantisation neither model fits or runs
at usable speed.

---

## 11. Product, dimensions, cost, usage

**Today:** two development kits plus a Raspberry Pi 4 on a bench, USB powered.

**Intended product:** a single handheld unit — one Silicon Labs xG26 module, a MEMS
microphone in a stethoscope chest piece, a rechargeable cell, in a moulded
enclosure. The gateway becomes any phone, tablet or laptop with a browser.

| | Figure |
|---|---|
| Enclosure size (design target) | **14.8 × 6.2 × 2.8 cm** |
| Mass (design target) | about **185 g** |
| Bill of materials per node | **₹2,999** at 100 units · **₹1,799** at 1,000 · **₹1,399** at 10,000 — itemised in [24_BILL_OF_MATERIALS.md](24_BILL_OF_MATERIALS.md) |
| Subscription | none |
| Per-study fee | none |
| Connectivity required | none |

The size and mass are the **design target for V2**, not a measured object. Nothing
has been moulded yet. Say so if asked.

**Why the cost matters:** the node costs less than the stethoscope it sits behind,
and because inference is on the endpoint there is no per-inference cost to charge
for.

**How it is used:** the operator places the stethoscope, presses one button,
and reads the result on any browser on the local network. No app to install. Full
disease names, never abbreviations. Six languages.

**Unique selling point, in one line:** *complete on-device cardiopulmonary
screening for ₹1,799.00 a node at a thousand units, that works with no network at all — and that tells
you when it does not know.*

---

## 12. Advantages over existing solutions

| Existing approach | Limitation | What we do |
|---|---|---|
| Cloud AI diagnostics | Needs connectivity; sends patient data off-site | Everything on the chip; nothing transmitted |
| Digital stethoscopes | Record and replay; the clinician still decides alone | Classifies on the device and shows what it measured |
| Radiology AI on workstations | Needs a workstation and a radiology department | Runs on a ₹1,799 node |
| Phone-app screening | Needs a phone per operator, app installs, updates | Any browser; no install |
| Research prototypes | Report accuracy only | We report accuracy **and** agreement, and we publish our failures |

**The differentiator judges will remember:** most systems give an answer no matter
what. Ours refuses. Raise the reporting threshold to 0.99 and the pulmonary node
declines on 114 of 120 cases rather than guessing. And a live microphone recording
shows the operator the signal without asserting a class at all, because the model
was never validated on that signal chain.

---

## 13. Why the neural network is logical and efficient

**Logical** — the architecture follows the data:

- Heart sound is a short, repeating, low-frequency pattern. Convert it to a small
  time-frequency image and a small CNN is the right tool.
- A chest X-ray is a large texture problem. That needs a real vision backbone,
  so we used one designed for mobile hardware.

**Efficient** — every choice is a measured trade:

| Choice | Reason |
|---|---|
| INT8 instead of float32 | 4x smaller, and the MVP accelerator is integer-only |
| Depthwise-separable convolutions | Most of the accuracy for a fraction of the multiply-accumulates |
| 96 x 96 input, not 224 x 224 | 224 x 224 would not fit; 96 x 96 still clears our gate |
| 13.5 k parameters for cardiac | Bigger gave no measurable gain; smaller is faster and cooler |
| 40 mel bands | Enough resolution for heart-sound frequencies without wasting compute |
| MVP accelerator | Convolution in hardware instead of on the CPU |

**Number to quote:** the pulmonary model has 412,773 parameters and runs in
177 kB of RAM in 644 ms on a 78 MHz microcontroller.

---

## 14. Where AI is used, and where it is not

**AI is used for:** the two classification models, and nothing else.

**Deliberately not AI:**

- Murmur presence and timing — classical signal processing (§5)
- Attention maps — occlusion, not a second network
- The out-of-range check on live audio — a measured spectral statistic
- Everything in the gateway and dashboard

**Why this matters:** every one of those could have been another neural network.
Each would have added memory, latency, and one more thing that can be confidently
wrong. Using signal processing where signal processing suffices is an engineering
decision we can defend.

---

## 15. Live input — what we added and its honest limits

Three ways to feed the system:

1. **Reference cases** — held-out data in the board's own flash.
2. **Upload** a chest X-ray or a heart sound file.
3. **Record** a heart sound live from the MEMS microphone.

In every case, the sample goes to the microcontroller and the microcontroller
classifies it. The Pi never classifies.

**The important honesty point.** For a **live microphone recording** we show the
spectrogram and **no class at all.**

**Why:** the cardiac model was trained on clean curated recordings. A MEMS
microphone on a stethoscope is a different transducer, a different coupling and a
different noise floor. The model was never validated on it. A softmax classifier
always returns a class, so it *will* answer — confidently and possibly wrongly.

We measured the gap rather than guessing: energy above 1 kHz is far higher in our
own field recordings than in the training corpus. A threshold at 0.0015 flags 82%
of our real-world recordings and 0% of the training set.

**Say it like this:** *"We show the operator the signal we recorded. We do not
show a class, because the model has not been validated on this signal chain and
we are not going to present a number we cannot stand behind."*

That is a stronger answer than any accuracy figure.

---

## 16. Security and privacy

| Control | Status |
|---|---|
| Raw audio and images never leave the chip | By design |
| Results encrypted before transmission | **Implemented** |
| Model integrity checked at every boot | **Implemented** |
| Database on the gateway holds no names | By design |

Every result is sealed before it goes over the air: **AES-128 counter mode** for
confidentiality, **AES-CMAC** for authenticity, encrypt-then-MAC. The wire format
is 28 bytes: a 4-byte counter, 16 bytes of ciphertext, an 8-byte tag.

**Total cost: 952 bytes of flash**, using primitives the Bluetooth stack already
carries.

The firmware also verifies a keyed checksum over the model at every boot. If the
model has been altered, the node refuses to run. We know it works because it
caught a real change during development.

**Honest limitation if asked:** one shared key for the demo fleet, which is fine
for a demonstration and not for deployment. Per-device keys and signed models are
in the V2 plan.

---

## 17. Biggest challenges — faced and coming

### Already faced

| Challenge | What happened |
|---|---|
| **Dataset confound** | Lung audio was 77% predictable from the recording device. We deleted the task. |
| **Compiler bug** | The vendor's optimiser silently miscompiled our model — 50.00% accuracy at 0.98 confidence, no warning. Caught by the agreement metric. Documented and reported. |
| **Toolchain bug** | The code generator omitted a required header. We wrote a patch. |
| **Boot failure** | A board flashed clean and never booted — wrong link address. Found by reading the vector table. |
| **Test set gap** | Our self-test tool was silently parsing encrypted results as plaintext after we added cryptography. |

### Still ahead

| Challenge | Why it is hard |
|---|---|
| **Does the model survive a real chest?** | The single largest open risk. Live audio is a different signal chain and we have not yet shown the model transfers. |
| **Clinical validation** | Needs ethics approval, a blinded protocol and a clinician reference standard. |
| **Positive predictive value** | At 95% sensitivity and 95% specificity with 2% prevalence, only about 28% of positives are true. That is arithmetic, it applies to every screening test, and we would rather raise it ourselves than have it raised for us. |
| **Regulatory class** | Software producing a disease-associated output attracts CDSCO classification. |

---

## 18. Version 2 in one minute

V1 proved the digital half: the models run correctly on the silicon, quantisation
costs nothing measurable, and the whole chain works. Every input it has ever
classified came from a file.

**V2 is the version that digitises a patient:**

1. **Live acquisition** — microphone into the microcontroller, with a signal
   quality gate that refuses a bad recording instead of classifying it.
2. **Calibrated uncertainty** — conformal prediction, so "not conclusive" carries
   a real guarantee rather than being a threshold we picked.
3. **Fleet security** — per-device keys, signed models, secure boot.
4. **Clinical pilot** — pre-registered, blinded, with sensitivity and specificity
   rather than accuracy.
5. **One custom board**, with power measured rather than estimated.

The middle of the pipeline does not change. That is the return on having built V1
properly.

---

## 19. Fast answers to likely questions

**"How much RAM does the model use while running?"**
48,412 bytes cardiac, 177,252 bytes pulmonary, of 512 kB. The device reports it
itself at boot.

**"Is this real-time?"**
57 ms cardiac inference, 644 ms pulmonary. Feature extraction adds 201 ms on
cardiac, 2 ms on pulmonary.

**"What if the model is wrong?"**
It is a screening aid, not a diagnosis; every result is reviewed by a clinician.
And it declines rather than guesses — raise the threshold to 0.99 and the
pulmonary node abstains on 114 of 120.

**"Why not just use a phone?"**
A phone costs 20 times more, needs charging, needs an app, and its microphone
applies automatic gain control and noise suppression that destroy exactly the
low-frequency content we need.

**"Why is your cardiac accuracy 100%?"**
100% on 53 held-out cases; cross-validated 98.9% ± 1.1%. Yaseen is a clean
balanced corpus, so the task is easier than bedside audio.

**"Have you tested on real patients?"**
No. We have collected recordings under consent, and they have not been used to
train or evaluate anything. No result we report depends on them.

**"What is the accelerator doing?"**
The MVP runs our convolution layers in hardware instead of on the CPU.

**"How do you know the chip computes the same thing as your PC?"**
We compile the held-out cases into the firmware, run them on the chip, and compare
every prediction against the PC's. Cardiac 53/53. Pulmonary 118/120, both
differences at the decision boundary.

**"What is your biggest weakness?"**
That we have not yet shown the model transfers to live audio from our own
microphone. We know it, we have measured the gap, and the dashboard refuses to
show a class for live recordings because of it.

---

## 20. Numbers to memorise

| | Cardiac | Pulmonary |
|---|---|---|
| Accuracy on device | **53/53 = 100%** | **115/120 = 95.83%** |
| Agreement with PC | **100%** | **98.33%** |
| Cross-validated | 98.9% ± 1.1% | 96.37% image / 97.85% patient |
| Parameters | 13,509 | 412,773 |
| Model size | 27.7 kB | 610 kB |
| **RAM at inference** | **48,412 B** | **177,252 B** |
| Inference | 57 ms | 644 ms |
| Front-end | 201 ms | 2 ms |
| Reference cases in flash | 53 | 120 |

**Others:** 512 kB RAM and 3,200 kB flash per chip · 78 MHz Cortex-M33 ·
28-byte sealed record · 952 bytes of crypto · ₹1,799.00 per node at 1,000 units ·
cold boot to both nodes connected in 57 seconds · murmur timing matches
physiology 49/52 · device-only lung-audio baseline 0.675 beat our CNN's 0.643.

---

## 21. Slide by slide — what each slide invites a judge to ask

The deck is frozen. This section walks it in order and gives the question each
slide is most likely to provoke, with a short answer. If a judge interrupts a
slide, the answer is here.

---

### Slide 1 — Title

**"Why the name?"**
Cardiopulmonary and revolutionary. It is a screening device for the heart and the
lungs, and the point of it is that the intelligence sits on the device rather than
in a data centre.

**"What do you mean by edge AI?"**
The neural network runs on the microcontroller that captured the signal. Nothing
is sent anywhere to be analysed. Not a phone, not a server, not the cloud.

---

### Slide 2 — The problem

**"Where do these numbers come from?"**
UNICEF under-five mortality estimates for pneumonia, and the Global Burden of
Disease 2021 and 2023 studies for valve disease. Both are on the slide.

**"Is screening really the bottleneck, or is it treatment?"**
For these two conditions it is screening. Pneumonia in children is treatable and
cheap to treat. Severe aortic stenosis is operable. The failure is that nobody
listens or looks at the right moment, and that step needs a trained ear or a
radiologist who is not there.

**"Why these two diseases and not something bigger?"**
Because both are picked up by a signal a cheap sensor can capture — a sound and
a picture — and both have public data good enough to train on honestly. We chose
two problems we could actually finish and verify.

---

### Slide 3 — Our solution

**"Why two boards? Isn't that twice the cost?"**
Today it is two, because the two models have different needs and we wanted both
verified on real silicon. They fit on one board: run one at a time, together they
need 638 kB of flash (19.9%) and 173 kB of RAM (33.8%). That merge is item 3 on
the roadmap and it is why the costing in the bill of materials assumes one module.

**"You say nothing leaves the board. What about the 28 bytes?"**
The 28 bytes are the answer, not the patient. Four bytes of counter, sixteen bytes
of encrypted result, eight bytes of authentication tag. No audio, no image, no
name. The recording itself is consumed in RAM and overwritten.

**"952 bytes of flash for security — how do you know?"**
Two firmware builds, one with the security layer and one without, and the linker
map differenced. It is 952 bytes, which is 0.03% of the 3,200 kB on the chip.

**"No app, no account — so how does a browser find it?"**
The Pi serves the dashboard on the local network. You type its address. There is
no login because there is nothing to log in to and no account anywhere.

---

### Slide 4 — Why Edge AI

**"You claim low power, but everything here is USB powered. What is the battery
life?"**
We have not measured it. The design uses a 78 MHz Cortex-M33 with a hardware
matrix accelerator and INT8 arithmetic instead of an application processor, which
is the architectural reason it can run from a small cell — but we have not put a
meter on it, and we will not quote a figure we did not take.

**"You say it works offline. Have you actually tested that?"**
Yes. There is no cloud service behind this to disconnect from. Unplug the gateway
and both boards keep classifying and reporting over BLE; the results simply have
nowhere to be displayed until the gateway comes back.

**"Zero recordings transmitted — is that a design claim or a measured one?"**
Both. The firmware has no path that transmits a sample. The only characteristic
that notifies is the 28-byte result, and it is specified in the GATT document.

---

### Slide 5 — System architecture

**"If there is no cloud, why do you need a Raspberry Pi at all?"**
Only to draw the web page and keep a local case list. It moves bytes; it does not
think. No inference of any kind happens on it. If it dies you lose the display,
not the screening.

**"Six languages — is that real?"**
The dashboard text is translated. It is interface language, not clinical
translation, and we describe it as exactly that.

---

### Slide 6 — Signal flow

**"Why a spectrogram for the heart but raw pixels for the lung?"**
A murmur is defined by *when* a sound happens inside the heartbeat and *what
frequencies* it contains, so the model needs a time-frequency view — 40 mel bands
across 192 frames. Pneumonia is a pattern in space, so the image goes in almost as
it is; the only preparation is centring the brightness.

**"Stage 4 is not a network. Why not?"**
Because it does not need to be. Locating S1 and S2, or blanking a patch and
watching the output move, is classical processing. A second network would cost
memory, latency, and give us one more thing capable of being confidently wrong.

---

### Slide 7 — Why each model sits on the kit it does

**"Could you not just put the lung model on the Dev Kit and use one board?"**
Yes, and that is the plan. We ran them separately for V1 so each could be verified
independently on its own silicon. The combined footprint is already known: 638 kB
flash, 173 kB RAM.

**"48 kB of recorded audio — how long is that?"**
Three seconds at 8 kHz, 16 bits per sample: 24,000 samples, 48,000 bytes. That is
one input to the model.

---

### Slide 8 — How we trained the models

**"Five folds. Why not ten?"**
Five was what we ran, and the spread is already reported: 98.9% ± 1.1%. More folds
would tighten the estimate slightly; it would not change the conclusion.

**"Yaseen does not publish patient IDs. So how is your heart split
patient-disjoint?"**
This is the honest answer and it is worth giving fully: it is not, strictly. Yaseen
publishes one recording per case with no patient identifier, so we treat each file
as its own patient. That is the strongest split the corpus allows. The X-ray split
*is* genuinely patient-disjoint, and we re-split it ourselves because the official
Kermany split leaks 264 subjects and 967 images between train and test.

**"You used an ImageNet-pretrained MobileNetV2. Is that not cheating?"**
No, it is transfer learning and it is standard. The pretrained weights supply
generic edge and texture filters; the pneumonia decision is learned from the chest
X-rays. It is also why a 412,773-parameter model reaches 96.37% on a small corpus.

**"Why 8-bit and not 16?"**
Because 8-bit costs us nothing measurable and buys a four-times reduction. The
heart model is identical before and after (98.06% → 98.06%). The lung model gives
up 0.24 points. The chip's accelerator is an integer engine, so 8-bit is also the
fast path.

---

### Slide 9 — Results

**"Cardiac 100% — I do not believe it."**
You should not, as a claim about the world. It is 51 of 53 reported and all
correct at the 0.70 floor, and 53 of 53 with the floor off. Cross-validated it is
98.9% ± 1.1%. Yaseen is a clean, balanced teaching corpus, so the task is easier
than a bedside recording. That is exactly why we are careful about live audio.

**"Where does the 0.70 reporting floor come from?"**
It is a deliberate operating point, not a fit. Below it the device reports
UNCERTAIN instead of a class. Push it to 0.99 and the lung node declines on 114 of
120 — the behaviour degrades into abstention, not into wrong answers.

**"Why is the heart front-end 201 ms when the lung front-end is 2 ms?"**
They are different amounts of work. The heart needs a real short-time Fourier
transform and a mel filterbank across 192 frames, computed on the CPU. The lung
front-end is one subtraction pass over 9,216 pixels.

**"Why does the smaller input take longer to classify on the lung side?"**
The lung model has 412,773 parameters against 13,509. It is thirty times the
arithmetic, so 644 ms against 57 ms.

**"What are the two pulmonary disagreements?"**
Two cases where the device and the PC land on opposite sides of the decision
boundary. We report them rather than hide them, and the live upload path reproduces
them exactly — which is itself evidence the live path is faithful.

---

### Slide 10 — Beyond the class label

**"Is the lung heat map a real explainability method?"**
It is occlusion sensitivity: blank a patch, measure how much the output moves,
repeat across the image. It is a standard method and it is honest about what it
is — it shows where the network was looking, not why.

**"49 of 52 — what about the other three?"**
Three cases where the murmur timing we measured did not match what the class
implies. We report the number rather than the fraction that flatters us. It is a
cross-check for the operator, not a second opinion.

---

### Slide 11 — Comparison with existing solutions

**"You claim a first. How can you possibly know?"**
We say *as far as we can establish from published products and peer-reviewed
work*, and we mean the wording. We searched the commercial products and the
literature and found cardiac-only microcontroller work, and cloud or phone-based
cardiopulmonary work, but nothing running both networks entirely on
microcontrollers. If a judge knows of one, we would genuinely like the reference.

**"The STM32H7 research node is cheaper than yours."**
It is, and it is one model, heart only, with no interface, no encryption and no
product around it. Our figure includes the microphone, the chest piece, the
battery, the enclosure and the assembly.

**"Where do the competitor prices come from?"**
Street prices, converted at approximately ₹84 to the dollar. They are indicative,
and we present them as such. Our own figure is the only one on that row that is a
costed bill of materials.

---

### Slide 12 — Product, cost and business model

**"Is ₹1,799 a quote from a vendor?"**
No. It is a costed bill of materials — twelve named line items, priced at
1,000-piece distributor pricing, with assembly and amortised tooling included. It
is not a quotation and it is not a selling price. The breakdown is in
[24_BILL_OF_MATERIALS.md](24_BILL_OF_MATERIALS.md) and we are happy to hand it over.

**"Why show three volumes?"**
Because a single number hides the thing that matters. At 100 units it is ₹2,999,
mostly because the enclosure has to be printed rather than moulded. At 10,000 it
is ₹1,399. Quoting only the best case would be the misleading version.

**"What would you sell it for?"**
We have not set a price. The bill of materials excludes engineering,
certification, duty and margin, so a selling price is higher. What we can commit
to is the structure: hardware cost plus margin on the screening node, and no
recurring fee of any kind.

**"Have you built the handheld unit?"**
No. The dimensions on that slide are a design target for V2. What exists today is
two development kits and a Raspberry Pi, and we say so.

---

### Slide 13 — Real world validation

**"You collected fifty recordings. Why is none of your accuracy based on them?"**
Because they are unlabelled by a clinician and because our model was never trained
on this microphone. Using them to claim accuracy would be dishonest in two separate
ways. They exist to characterise the gap, not to inflate a number.

**"Then what did you learn from them?"**
That the gap is real and measurable. Energy above 1 kHz separates our recordings
from the training corpus: a threshold of 0.0015 flags 82% of our field recordings
and 0% of the training set. That measurement is why the dashboard refuses to show
a class for a live capture.

**"Fifty recordings from ten people is not fifty independent samples."**
Correct, and we do not present it as fifty patients. Ten people, five recordings
each, collected with written consent.

**"Consent and ethics?"**
Written consent from every participant. The recordings have not been used to train
or evaluate any model, no result in the deck depends on them, and the patient
photographs carry no faces or identifiers.

---

### Slide 14 — Limitations and roadmap

**"Your device cannot classify a live recording. Is it not useless in a clinic?"**
Today, for live cardiac audio, it shows the operator the signal and does not assert
a class. That is the correct behaviour for a model that has not been validated on
that microphone, and it is a deliberate choice rather than a missing feature. The
X-ray path and the reference-case path classify normally. Closing that gap is
items 1, 2 and 5 on the roadmap.

**"Twenty days for six items — is that realistic?"**
Items 3 and 4 are engineering we have already scoped, and item 3's memory
budget is measured. Items 1, 2 and 5 depend on a clinician's availability, which is
the real risk. We would rather show you the plan and the risk than a plan without
one.

**"The lung node has no camera. How does an X-ray reach it?"**
From a radiology system, or a photograph of a film. Until that path is as native as
the microphone, the lung node depends on an X-ray already existing. It is on the
slide because it is a genuine limitation.

---

## 22. What we have not measured

Read this list before the session. If a judge asks for one of these, the answer is
*"we did not measure that"* — said plainly, without improvising a number.

| Not measured | What we can say instead |
|---|---|
| Battery life, and power in mW | The architectural reasons it should be low: 78 MHz Cortex-M33, hardware accelerator, INT8 throughout, no application processor |
| Actual enclosure size and mass | 14.8 × 6.2 × 2.8 cm and about 185 g are **design targets** for V2; nothing has been moulded |
| BLE range, and behaviour at range | Both nodes connect and hold on the bench; we have not characterised distance |
| Operating temperature, drop, ingress | No environmental testing of any kind has been done |
| Clinical sensitivity and specificity on real patients | None. Every accuracy figure is on public held-out data |
| Accuracy on audio from our own microphone | Not established — this is the headline limitation, and it is why live captures assert no class |
| Long-run stability | The 1,000-inference soak test is planned and has not been run |
| A vendor quotation for the bill of materials | It is a costing at distributor pricing, not a quote |

**The sentence to use:** *"We did not measure that. Here is what we did measure,
and here is why we think it points the same way."*
