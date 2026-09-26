# cardio/fusion — heart sound + patient context

Combines the heart-sound model with patient metadata and symptoms, the way a
clinician weighs an auscultation finding against the history.

Questions this workstream answers:

- **Late fusion** (combine the model's output with metadata) against **early fusion**
  (combine representations) — which helps, and by how much?
- Does metadata improve the decision, or does the model start leaning on it instead of
  the sound? Measured by ablation: signal only, metadata only, both.

Uses the `PatientMetadata` schema from `core.contracts`.
