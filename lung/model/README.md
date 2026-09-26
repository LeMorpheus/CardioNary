# lung/model — chest X-ray classifier

Two classes: no pneumonia pattern, and a pattern consistent with pneumonia.

Input is the 96 x 96 greyscale image from `core.images`. Interpretability lives here
too: occlusion maps showing which region of the lung field drove each result.
Training and evaluation go through the shared harness in `core`.
