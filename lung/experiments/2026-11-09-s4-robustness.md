# S4 robustness and batch-normalisation adaptation

**Branch:** `lung-vedansh-rsna-evaluation`  
**Data:** Kermany fold 0; adaptation on 32 images, with adaptation patients excluded
from scoring.

## What was run

Ran `lung.eval.run_robustness` for blur, noise, contrast, and JPEG compression at
five severity levels, comparing plain inference with batch-normalisation adaptation.

## Results and conclusion

Adaptation was not uniformly beneficial. It improved accuracy under severe blur
(severity 3: **0.835 to 0.943**) and severe noise (severity 5: **0.588 to 0.715**),
but JPEG severity 5 accuracy fell (**0.954 to 0.901**). Record results by
corruption and severity rather than claiming an overall robustness gain.

## Source files

- `reports/xray_robustness.json`
- `reports/figures/xray_robustness.png`
