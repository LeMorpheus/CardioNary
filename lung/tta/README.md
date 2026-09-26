# lung/tta — test-time adaptation

X-ray machines, exposure settings and hospitals differ. A model trained on one
source will meet others. This workstream adapts the model at test time, from a new
site's own unlabelled images, so it holds up across equipment.

Evaluated three ways: accuracy on shifted data before adaptation, after adaptation,
and on the original data after adaptation — to show nothing was forgotten.
