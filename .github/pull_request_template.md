## What this changes

<!-- One or two sentences. What does this pull request do, and why? -->

## Lane

- [ ] `core` — shared pipeline (both leads must review)
- [ ] `cardio` — model · fusion · tta
- [ ] `lung` — model · fusion · tta
- [ ] `app` — dashboard
- [ ] `docs` / `infra`

## Checklist

- [ ] The branch name follows `<lane>/<workstream>/<name>-<topic>` (see `docs/BRANCHING.md`)
- [ ] `python -m pytest tests/` passes locally, including the anti-leakage tests
- [ ] Nothing from the test fold reached training: normalisation, class weights and quantisation calibration were all fitted on the training fold only
- [ ] Every new number has a results file behind it, in `reports/` or `<lane>/experiments/`
- [ ] Anything that did not work is written up in `<lane>/experiments/`, not deleted
- [ ] No datasets or trained weights are committed
- [ ] Dashboard changes use no red anywhere in the design

## Evidence

<!-- Metrics, a plot, or a screenshot. Each number should name the file it came from. -->

## AI assistance

<!-- Required by the course. Which tool, what it was used for, and how you checked
     its output. Write "None" if no AI tool was used. -->
