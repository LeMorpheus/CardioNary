"""Tests for cardio/predict.py, the heart result the dashboard shows.

Every recording here is made up, so the tests need no data. The ones that run the
model skip themselves when the model files are absent (models/ is not tracked on
the cardio branch).

Run:
    python -m pytest tests/test_cardio_predict.py -v
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from scipy.io import wavfile

from cardio import predict
from cardio.model.final_model import CLASSES, MODEL_DIR, REPO_ROOT, STEM
from core.contracts import FusedOutput, PatientMetadata

MODEL_FILES = (MODEL_DIR / f"{STEM}_float.keras", MODEL_DIR / f"{STEM}_int8.tflite",
               REPO_ROOT / "reports" / f"quantization_{STEM}.json")
needs_model = pytest.mark.skipif(not all(p.exists() for p in MODEL_FILES),
                                 reason="model files not in this checkout (models/ is not tracked on cardio)")


def heartbeat(seconds=6.0, fs=4000, bpm=72):
    """A clean synthetic heartbeat: low-pitched S1 and S2 bursts, nothing above 1 kHz."""
    x = np.zeros(int(seconds * fs))
    period = 60.0 / bpm

    def burst(start, dur, freq, amp):
        n0, n = int(start * fs), int(dur * fs)
        n = min(n, len(x) - n0)
        if n > 0:
            x[n0:n0 + n] += amp * np.hanning(n) * np.sin(2 * np.pi * freq * np.arange(n) / fs)

    for k in range(int(seconds / period) + 1):
        burst(k * period + 0.1, 0.10, 50.0, 0.6)          # S1
        burst(k * period + 0.4, 0.08, 70.0, 0.4)          # S2
    return x, fs


def wav_bytes(x, fs) -> bytes:
    buf = io.BytesIO()
    wavfile.write(buf, fs, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    return buf.getvalue()


@pytest.fixture
def no_fusion_files(tmp_path, monkeypatch):
    """Run as if neither calibration nor fusion parameters exist yet, whatever reports/ holds."""
    monkeypatch.setattr(predict, "FUSION_PARAMS", tmp_path / "heart_fusion_params.json")
    monkeypatch.setattr(predict, "CALIBRATION", tmp_path / "heart_calibration.json")
    return tmp_path


# ---------------------------------------------------------------------------
# Without the model
# ---------------------------------------------------------------------------
def test_systole_or_diastole_follows_the_measured_energy():
    assert predict.systole_or_diastole({"segmentation_ok": 0}) is None
    assert predict.systole_or_diastole({"segmentation_ok": 1, "murmur_detected": 0}) == "neither"
    loud_systole = {"segmentation_ok": 1, "murmur_detected": 1, "systolic_ratio": 0.8, "diastolic_ratio": 0.1}
    loud_diastole = {"segmentation_ok": 1, "murmur_detected": 1, "systolic_ratio": 0.05, "diastolic_ratio": 0.12}
    assert predict.systole_or_diastole(loud_systole) == "systole"
    assert predict.systole_or_diastole(loud_diastole) == "diastole"


def test_temperature_sharpens_or_softens_but_never_changes_the_winner():
    p = np.array([0.1, 0.6, 0.1, 0.1, 0.1])
    assert predict.calibrate(p, None) == pytest.approx(p)
    sharp, soft = predict.calibrate(p, 0.5), predict.calibrate(p, 2.0)
    for q in (sharp, soft):
        assert q.sum() == pytest.approx(1.0) and q.argmax() == 1
    assert sharp[1] > p[1] > soft[1]


def test_temperature_is_read_from_fusion_parameters_first(no_fusion_files):
    assert predict.temperature() is None
    predict.CALIBRATION.write_text('{"T": 0.5}')
    assert predict.temperature() == 0.5
    predict.FUSION_PARAMS.write_text('{"T": 0.48, "weights": {}}')
    assert predict.temperature() == 0.48
    predict.FUSION_PARAMS.write_text('{"T": -1}')
    with pytest.raises(ValueError):
        predict.temperature()


def test_uploads_are_read_from_memory(tmp_path):
    data = wav_bytes(*heartbeat(2.0))
    path = tmp_path / "beat.wav"
    path.write_bytes(data)
    assert predict._wav_source(path) == str(path)
    for upload in (data, io.BytesIO(data)):
        assert predict._wav_source(upload).getvalue() == data
    with path.open("rb") as fh:                # a handle that was already read to the end
        fh.read()
        assert predict._wav_source(fh).getvalue() == data
    with pytest.raises(TypeError):
        predict._wav_source(12345)


def test_unreadable_and_too_short_are_withheld_before_the_model():
    assert predict.analyse_heart(b"this is not a wav file").withheld == "unreadable"
    res = predict.analyse_heart(wav_bytes(*heartbeat(1.0)))        # 1 s: no 3-second window
    assert res.withheld == "too_short" and res.model_output is None and res.seconds_of_audio == pytest.approx(1.0)


def test_characteristics_from_audio_matches_the_file_version(tmp_path):
    from cardio.model.features import characteristics, characteristics_from_audio
    from core.features import CARDIO_CONFIG, load_wav

    path = tmp_path / "beat.wav"
    path.write_bytes(wav_bytes(*heartbeat()))
    from_file = characteristics(path)
    from_audio = characteristics_from_audio(load_wav(str(path), CARDIO_CONFIG))
    assert from_file.keys() == from_audio.keys() and len(from_file) == 17
    assert from_file == from_audio


# ---------------------------------------------------------------------------
# With the shipped model
# ---------------------------------------------------------------------------
@needs_model
def test_clean_heartbeat_gives_valid_contracts(no_fusion_files):
    res = predict.analyse_heart(wav_bytes(*heartbeat()), PatientMetadata(sex="female"))
    assert res.withheld is None and res.windows >= 1
    out = res.model_output
    assert out.modality == "heart" and set(out.probabilities) == set(CLASSES) and out.quality_ok
    assert sum(out.probabilities.values()) == pytest.approx(1.0)
    assert res.predicted_class in CLASSES
    # no fitted fusion yet: the CNN vote alone, with weight 1 (the missing-vote rule)
    assert res.fusion == "cnn_only" and res.temperature is None
    assert res.fused.weights == {"cnn": 1.0} and res.p_h == pytest.approx(out.p_abnormal)
    assert set(res.timings_ms) == set(predict.STAGES) and res.total_ms > 0
    assert len(res.characteristics) == 17 and len(res.pcm16()) == 2 * len(res.audio_8k)


@needs_model
def test_broadband_noise_is_withheld(no_fusion_files):
    noise = np.random.default_rng(0).normal(0, 0.3, 6 * 48000)
    res = predict.analyse_heart(wav_bytes(noise, 48000))
    assert res.withheld == "quality_check" and res.quality["in_distribution"] is False
    assert res.model_output is None and res.fused is None
    # the edge budget runs a whole case anyway; the result stays withheld
    full = predict.analyse_heart(wav_bytes(noise, 48000), complete=True)
    assert full.withheld == "quality_check" and not full.model_output.quality_ok and full.fused is not None


@needs_model
def test_path_bytes_and_upload_give_the_same_answer(no_fusion_files, tmp_path):
    data = wav_bytes(*heartbeat())
    path = tmp_path / "beat.wav"
    path.write_bytes(data)
    answers = [predict.analyse_heart(r).model_output.probabilities for r in (path, data, io.BytesIO(data))]
    assert answers[0] == answers[1] == answers[2]


@needs_model
def test_cnn_is_the_shipped_8bit_model(no_fusion_files, tmp_path):
    """Same windows and same 8-bit probabilities as the agreement and edge-budget scripts."""
    from core.features import CARDIO_CONFIG, features_for_file

    path = tmp_path / "beat.wav"
    path.write_bytes(wav_bytes(*heartbeat()))
    model = predict.load_model()
    _, p_int8 = model.predict_windows(features_for_file(str(path), CARDIO_CONFIG))
    expected = p_int8.mean(0) / p_int8.mean(0).sum()
    got = predict.analyse_heart(path).model_output.probabilities
    assert [got[c] for c in CLASSES] == pytest.approx(expected, abs=1e-6)


@needs_model
def test_temperature_and_heart_fusion_are_used_once_they_exist(no_fusion_files, monkeypatch):
    data = wav_bytes(*heartbeat())
    plain = predict.analyse_heart(data).model_output.probabilities
    predict.CALIBRATION.write_text('{"T": 0.5}')
    seen = {}

    def fake_heart_fusion(p_cnn, features, metadata):
        seen.update(p_cnn=p_cnn, features=features, metadata=metadata)
        return FusedOutput("heart", p_abnormal=0.5 * p_cnn + 0.5 * 0.2,
                           votes={"cnn": p_cnn, "features": 0.2}, weights={"cnn": 0.5, "features": 0.5})

    monkeypatch.setattr(predict, "fusion_function", lambda: fake_heart_fusion)
    meta = PatientMetadata(age_years=40.0, sex="male")
    res = predict.analyse_heart(data, meta)
    calibrated = res.model_output.probabilities
    winner = max(plain, key=plain.get)
    assert res.temperature == 0.5 and max(calibrated, key=calibrated.get) == winner
    assert calibrated[winner] >= plain[winner]
    assert seen["p_cnn"] == pytest.approx(1 - calibrated["Normal"])   # calibrated, as the roadmap says
    assert len(seen["features"]) == 17 and seen["metadata"] is meta
    assert res.fusion == "heart_fusion" and res.fused.weights == {"cnn": 0.5, "features": 0.5}
