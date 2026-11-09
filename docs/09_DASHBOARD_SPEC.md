# Local Dashboard & Gateway Specification (Raspberry Pi 4)

Satisfies requirement **R7**: results shown on a local dashboard hosted on the Pi.
The dashboard is the system's only clinical-facing surface: it is the component a nurse or
clinician operates, and the only place a result is presented for review. It is developed
after the models and the BLE path are verified correct, so that it presents results already
known to be sound.

---

## 0. Delivered features (2026-08-26, frozen)

| Feature | Behaviour |
|---|---|
| Class probability bars | Green, filled proportionally; winning class brighter. Numeric value always shown alongside, so the reading never depends on colour. |
| Case media | Source radiograph on pulmonary cases, source audio player on cardiac cases, served from `/api/media/{node}/{vector_id}`. |
| Spectrogram | Every cardiac heart-sound result. Live paths carry it inline; reference vectors are rendered and cached by `/api/spectrogram/1/{vector_id}`. |
| Upload chest X-ray | Any image Pillow can open, converted to 96x96 greyscale and pushed to the pulmonary node. |
| Upload heart sound | Any WAV, resampled to 8 kHz mono and pushed to the cardiac node. |
| Record heart sound | 5 s from the MEMS microphone. **Shows the spectrogram and no class** -- see below. |
| Navigation | The case counter and the case-detail panel follow whatever the card is displaying. |
| Caching | The dashboard document is served `no-store`; static assets are version-stamped. |

**Why a live microphone recording asserts no class.** The cardiac model was
trained on curated recordings. A MEMS microphone on a stethoscope is a different
transducer, coupling and noise floor, and the model has never been validated on
it. A softmax classifier returns a class regardless. Rather than present an
unvalidated class, the card shows the recorded signal alone. The discriminating
statistic -- energy above 1 kHz -- was measured across 53 training vectors and 50
field recordings: a threshold of 0.0015 flags 82 % of field recordings and 0 % of
the training set.

---

## 1. Service architecture

```
                 ┌──────────────────── Raspberry Pi 4 ────────────────────┐
  BLE            │                                                        │
Chiron-Cardio ──►│  chiron_gateway.py                                     │
Chiron-Pulmo  ──►│    ├─ BLE client (bleak) — one task per node           │
                 │    ├─ SQLite writer (events.db)                        │
                 │    └─ FastAPI app                                      │
                 │         ├─ GET  /                → dashboard (static)  │
                 │         ├─ GET  /api/nodes       → node info + status   │
                 │         ├─ GET  /api/events      → history (paged)      │
                 │         ├─ GET  /api/stats       → live accuracy tally  │
                 │         ├─ POST /api/command/{n} → BLE write            │
                 │         └─ WS   /ws              → live event stream    │
                 └────────────────────────────────────────────────────────┘
                                  http://chiron.local:8000
```

**Single process.** `bleak` and `uvicorn` are both asyncio, so the BLE clients run as asyncio
tasks inside the FastAPI event loop. No IPC, no message broker, no second service to fail.

---

## 2. Technology choices

| Layer | Choice | Why not the alternative |
|---|---|---|
| BLE | `bleak` | `gatttool` is deprecated; raw D-Bus is a time sink |
| Backend | FastAPI + uvicorn | WebSocket + REST + static files in ~200 lines; `uvicorn` already in our toolchain |
| Store | SQLite (stdlib) | Zero-config, file-backed, survives reboot; Postgres is absurd here |
| Frontend | **Static HTML + vanilla JS + Chart.js (vendored)** | React/Next.js needs Node, a build step and `node_modules` on the Pi. All of that is demo-day risk for zero visible benefit at this scale. |
| Charts | Chart.js, **vendored locally** | No CDN — the demo must work with no internet |

> The earlier project proposal specifies a React/Next.js UI. Deliberately overruled: a build
> toolchain on a Raspberry Pi is a liability during a live final, and the judging criterion is
> *user friendly*, not *modern framework*. If time remains after everything else is verified,
> the frontend can be upgraded — the API contract does not change.

**Everything is vendored. Zero external network requests at runtime.**

---

## 3. Data model

```sql
CREATE TABLE events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc            TEXT    NOT NULL,   -- Pi receive time, ISO-8601
    node_type         INTEGER NOT NULL,   -- 1 cardio, 2 pulmo
    node_address      TEXT    NOT NULL,
    class_id          INTEGER NOT NULL,   -- 255 = UNCERTAIN
    class_name        TEXT    NOT NULL,
    confidence        REAL    NOT NULL,
    top2_class_id     INTEGER,
    top2_confidence   REAL,
    vector_id         INTEGER,
    expected_class_id INTEGER,            -- NULL if unknown
    is_correct        INTEGER,            -- NULL if no ground truth
    dsp_time_ms       INTEGER,
    inference_time_ms INTEGER,
    hysteresis_ok     INTEGER,
    self_test         INTEGER,
    sequence          INTEGER
);
CREATE INDEX idx_events_ts   ON events(ts_utc);
CREATE INDEX idx_events_node ON events(node_type, ts_utc);

CREATE TABLE nodes (
    node_type        INTEGER PRIMARY KEY,
    address          TEXT,
    name             TEXT,
    class_names      TEXT,      -- newline separated, read from GATT
    model_version    TEXT,
    build_id         TEXT,
    arena_used_bytes INTEGER,
    model_size_bytes INTEGER,
    last_seen_utc    TEXT,
    connected        INTEGER
);
```

`is_correct` is computed on ingest from `expected_class_id`. That single column is what makes
the live-accuracy panel possible.

---

## 4. Dashboard layout

```
┌───────────────────────────────────────────────────────────────────────────┐
│  PROJECT CHIRON — Edge Cardiopulmonary Screening        ● Cardio  ● Pulmo  │
│  Local demonstration · not a medical device                    [Self-Test] │
├──────────────────────────────────┬────────────────────────────────────────┤
│  CARDIAC NODE          xG26 Dev  │  PULMONARY NODE       xG26 Explorer     │
│  ┌────────────────────────────┐  │  ┌───────────────────────────────────┐  │
│  │   AORTIC STENOSIS          │  │  │   ASTHMA                          │  │
│  │   confidence  97.3 %       │  │  │   confidence  95.8 %              │  │
│  │   matches expected       │  │  │   matches expected              │  │
│  └────────────────────────────┘  │  └───────────────────────────────────┘  │
│  class probability bars          │  class probability bars                 │
│  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇  AS  97 %    │  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇  Asthma      96 %    │
│  ▇                    MR   2 %    │  ▇                   COPD         3 %    │
│  ·                    N    1 %    │  ·                   Normal       1 %    │
│                                  │                                         │
│  DSP 24 ms · Infer 19 ms         │  DSP 43 ms · Infer 38 ms                │
│  Arena 58,112 B · build a3f19c   │  Arena 119,744 B · build a3f19c         │
│  [◀ prev]  vector 07/24  [next ▶]│  [◀ prev]  vector 12/24  [next ▶]       │
├──────────────────────────────────┴────────────────────────────────────────┤
│  LIVE SELF-TEST ACCURACY                                                   │
│  Cardio  24/24  100.0 %      Pulmo  23/24   95.8 %                         │
├───────────────────────────────────────────────────────────────────────────┤
│  CONFIDENCE HISTORY (last 50 inferences, both nodes)                       │
│  [line chart]                                                              │
├───────────────────────────────────────────────────────────────────────────┤
│  EVENT LOG                                                    [Export CSV] │
│  12:04:31  Pulmo   Asthma            95.8 %  vec 12  exp Asthma   │
│  12:04:29  Cardio  Aortic Stenosis   97.3 %  vec 07  exp AS       │
│  12:04:27  Pulmo   COPD              88.1 %  vec 11  exp COPD     │
└───────────────────────────────────────────────────────────────────────────┘
```

### Design rules
1. **Ground truth is always on screen.** Every row shows expected vs predicted and a /.
   This is the difference between a dashboard that *displays* results and one that *proves*
   them.
2. **The medical disclaimer is in the header**, on every screen, always.
3. **Offline nodes are visibly offline**, greyed with a "last seen" time — never silently
   stale data.
4. **`UNCERTAIN` renders as amber**, not as a class. Uncertainty is a real output.
5. **Latency and memory are shown**, not hidden. They are engineering evidence.
6. Legible from **2 m away on a projector** — this will be judged over a video call.

---

## 5. API contract

| Method | Path | Returns |
|---|---|---|
| `GET` | `/` | dashboard HTML |
| `GET` | `/api/nodes` | both nodes: identity, connected, class names, arena, build |
| `GET` | `/api/events?limit=100&node=1` | recent events, newest first |
| `GET` | `/api/stats` | per node: total runs, correct, accuracy, mean latency |
| `POST` | `/api/command/{node_type}` | `{"opcode": 2, "param": 0}` → GATT write |
| `GET` | `/api/export.csv` | full event log as CSV |
| `WS` | `/ws` | JSON event on every notification |

WebSocket message:
```json
{
  "type": "inference",
  "ts_utc": "2026-08-12T12:04:31.482Z",
  "node_type": 2, "node_name": "Chiron-Pulmo",
  "class_id": 1, "class_name": "Asthma", "confidence": 0.958,
  "top2_class_name": "COPD", "top2_confidence": 0.031,
  "vector_id": 12, "expected_class_name": "Asthma", "is_correct": true,
  "dsp_time_ms": 43, "inference_time_ms": 38, "hysteresis_ok": true
}
```

---

## 6. Deployment on the Pi

```bash
sudo apt install -y python3-venv bluez
python3 -m venv ~/chiron/.venv && ~/chiron/.venv/bin/pip install bleak fastapi uvicorn
```

`/etc/systemd/system/chiron-gateway.service`:
```ini
[Unit]
Description=Chiron BLE Gateway and Dashboard
After=bluetooth.target network.target
Requires=bluetooth.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/chiron
ExecStart=/home/pi/chiron/.venv/bin/python -m uvicorn chiron_gateway:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now chiron-gateway
```

**Acceptance:** power-cycle the Pi with no keyboard or monitor attached; within 60 s the
dashboard must be reachable and both nodes connected.

---

## 7. Demo-day hardening

| Risk | Mitigation |
|---|---|
| No venue Wi-Fi | Pi runs its **own AP** (`hostapd` + `dnsmasq`); reviewers join `chiron-demo` |
| Pi fails entirely | The **same `chiron_gateway.py` runs on a Windows host** (bleak is cross-platform). This path is exercised as part of pre-deployment checks. |
| BLE drops mid-demo | Auto-reconnect with backoff; dashboard shows "reconnecting…", not a blank screen |
| Board resets | Status characteristic reports `state=boot`; dashboard shows it and recovers |
| A reviewer asks whether results are pre-recorded | | Hand them the `[Self-Test]` button and let them trigger a live sweep |

That last row is the important one. The dashboard should make live interaction so easy that
the question answers itself.

---

## Clinical presentation revision (2026-08-22)

The dashboard was re-pitched from an engineering readout to something a nurse or
clinical officer could actually use.

### What changed

| Change | Reason |
|---|---|
| **Full disease names everywhere** — "Mitral valve prolapse", never "MVP" | Abbreviations are fine on a BLE wire and wrong in front of clinical staff |
| **"Not a medical device" banner removed** | Requested; the boundary is stated in the competition submission and remains in the firmware boot log and every document here |
| **Case detail view** | Select any case, open it, and see the full analysis rather than one line in a log |
| **Six languages**, English default | Hindi, Bengali, Telugu, Marathi, Tamil |
| **No emoji, no rating scales** | Numbers and plain words only, e.g. "Refer to cardiology within 14 days" |
| **Cross-validated accuracy shown beside batch accuracy** | See "the 100% problem" below |

### Class names come from the client, not the device

The board transmits short codes over GATT. The dashboard maps *class id* →
full name via `static/i18n.js`. The device remains the source of truth for the
id; only its presentation is local. This is also what makes translation possible
without touching firmware.

### Three tiers of information, deliberately separated

The case view keeps these visually distinct, because conflating them is how a
screening tool starts implying things it cannot support:

1. **Analysis result** — what the model produced (class, confidence, full distribution).
2. **Measured from this recording** — real numbers computed from the exact audio
   flashed on the device: heart rate, rhythm regularity, sound events, signal
   quality. Produced by `core/build_case_data.py`.
   For the pulmonary node this section says plainly that **an X-ray classifier
   measures no vital signs** — inventing numbers there would be the dishonest part.
3. **Suggested next steps** — generic reference material keyed by class, marked
   *"not derived from this recording"* and given a coloured left rule so it never
   reads as a measurement of this patient. Referral-and-confirmation guidance
   only: which test confirms, who to refer to, within how many days. No treatment
   or dosing.

A fourth section, **"What this screen cannot tell you"**, carries the per-class
limitations (severity cannot be graded from sound; the pulmonary model was
trained on ages 1–5).

### Heart rate is genuinely measured

Autocorrelation of a Shannon-energy envelope over a 45–140 bpm window. Across the
53 cardiac cases this yields 60–94 bpm with **no outliers**.

An explicit octave correction (prefer 2x lag when its autocorrelation is
comparable) was implemented and **made results worse** — it doubled periods that
were already correct, pushing 5 of 53 outside the physiological range where
previously only 1 was. It was removed in favour of the narrower search window.
The cap is a stated assumption: a genuine tachycardia above 140 bpm would be
clipped, which is acceptable for resting clinical recordings and would not be for
live monitoring.

### The 100% problem, and why it is not "fixed" in the UI

Cardiac self-test accuracy is **53/53 = 100.00%**, which looks implausible.

The sample was enlarged from 29 to 53 cases specifically to test whether the
figure would come down. **It did not** — the model genuinely classifies every
held-out cardiac case correctly, on hardware, and the offline model does too.

Making the number look more modest would mean fabricating it, which is a far
worse failure under technical review than an awkward number. Instead the
dashboard **discloses more**:

- **Cross-validated accuracy is shown next to batch accuracy** — 98.9% ± 1.1%,
  which is the statistically meaningful figure and is not 100%.
- **n is always visible** ("53 / 53"), so the reader can judge the sample size.
- The reference dataset is named in every case view, so anyone can ask why
  Yaseen is an easy corpus. The analysis is in report A1.

**100 cases per batch is not achievable for the cardiac node.** 100 raw-audio
vectors need 4.8 MB against a 3.19 MB flash region — 1.5x the entire space. The
ceiling is 59; 53 is what fits with margin. A full batch runs **53 cardiac + 120
pulmonary = 173 cases**, which is above 100 in total. Running one node's set
twice to reach a bigger number would double the denominator without adding a
single new case, so it is not done.

### Translation status

The non-English clinical strings have **not been reviewed by native speakers**.
Medical terminology is exactly where a plausible-but-wrong translation causes
harm. `static/i18n.js` carries this notice at the top, and English is the
reference text. This must be checked by a clinician who speaks each language
before any real-world use.

### Case navigation (2026-08-22)

**Stepping to a case analyses it.** Previously Next only moved a pointer, and a
second tap on "Analyse this case" was needed to get a result — redundant when
walking a batch. "Analyse this case" is retained for deliberately re-running the
current case.

**Debounced at 260 ms.** Holding Next would otherwise queue one inference per
tap. At 644 ms per pulmonary case, ten taps means the board is still grinding
through a backlog long after the project lead stopped. Verified: 10 rapid taps advance
the index by 10 and send **exactly one** command, for the case landed on.

**The transition had to be visible.** Consecutive cases frequently share a
finding — cases 11 and 12 of the pulmonary set both read "No pneumonia pattern" —
so the verdict text alone cannot signal that anything happened. Three cues:

1. The verdict is **replaced** with "Analysing…" plus a moving scan bar. It is
   not dimmed-but-readable: leaving the previous case's finding on screen while
   a new one computes is how someone reads the wrong result for the right case.
2. The arriving result rises and fades in (~340 ms).
3. The **case counter flashes** and is styled as the prominent element, since the
   number is the thing that actually changed.

A `prefers-reduced-motion` block keeps the state changes and drops the movement.

Stuck-state guard: if the board is mid-sweep the single-case request waits its
turn, so "Analysing…" clears itself after 8 s rather than hanging.
