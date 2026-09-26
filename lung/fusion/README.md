# lung/fusion — chest X-ray + patient context

Combines the X-ray model with patient metadata and symptoms — age, fever, cough,
breathing rate — the way a clinician reads a film alongside the history.

Questions this workstream answers:

- **Late fusion** against **early fusion** — which helps, and by how much?
- Does metadata improve the decision, or does the model start leaning on it instead of
  the image? Measured by ablation: image only, metadata only, both.

Uses the `PatientMetadata` schema from `core.contracts`.
