# cardio/model — heart-sound classifier

Five classes: Normal, Aortic Stenosis, Mitral Stenosis, Mitral Regurgitation,
Mitral Valve Prolapse.

Input is the log-mel spectrogram from `core.features` — 3 s of audio at 8 kHz,
40 mel bands by 192 frames. Training and evaluation go through the shared harness in
`core`, so results are directly comparable with the lung team's.
