# cardio/tta — test-time adaptation

A model trained on one set of stethoscopes and recording conditions will meet
different ones at a new site. This workstream adapts the model at test time, from
that site's own unlabelled recordings, so it becomes instrument-agnostic.

Evaluated three ways: accuracy on shifted data before adaptation, after adaptation,
and on the original data after adaptation — to show nothing was forgotten.
