# S2 RSNA evaluation and fusion

**Branch:** `lung-vedansh-rsna-evaluation`  
**Data:** RSNA, evaluated with Kermany normalization; Kermany fold 0 held out.

## What was run

Ran the float and int8 X-ray models on RSNA, then compared CNN-only, feature fusion,
and metadata fusion. The bootstrap used 1,000 resamples (seed 0). RSNA DICOM age
was unavailable for all 14,863 patients, so the metadata vote used sex only.

## Results

- RSNA AUROC: float **0.9071** (95% CI **0.9019–0.9122**); int8 **0.8970**
  (95% CI **0.8917–0.9019**).
- CNN plus image features AUROC was **0.9010**, versus **0.9071** for CNN alone:
  gain **−0.0061** (95% CI **−0.00735 to −0.00474**). The target of at least
  **+0.03** was not met.
- Adding the sex-only metadata vote changed AUROC from **0.8955** to **0.9003**:
  gain **+0.0048** (95% CI **+0.00419 to +0.00542**). This does not establish an
  age-and-sex metadata result because all ages were missing.
- Kermany calibration ECE changed from **0.0269** to **0.0247**, meeting the
  configured target of **<= 0.05**.

## Conclusion

The S2 evaluation ran successfully, but feature fusion reduced AUROC and missed
its target. Report the sex-only limitation; do not describe the metadata result
as an age-and-sex comparison.

## Source files

- `reports/xray_rsna.json`
- `reports/xray_fusion.json`
- `reports/xray_cnn_rsna.csv`
- `reports/xray_features_rsna.csv`
