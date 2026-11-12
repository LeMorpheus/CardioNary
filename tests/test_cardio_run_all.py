"""Checks on the plan of cardio/run_all.py. Nothing here runs a step.

Run:
    python -m pytest tests/test_cardio_run_all.py -v
"""

from __future__ import annotations

import pytest

from cardio import run_all

# Steps whose code a teammate has still to merge into this branch; any other module
# name that does not exist is a typo in run_all.STEPS.
PENDING = {"cardio.fusion.votes", "cardio.fusion.calibrate", "cardio.fusion.late_fusion",
           "cardio.eval.run_robustness"}


def test_steps_are_unique_and_write_result_files():
    names = [s.name for s in run_all.STEPS]
    assert len(names) == len(set(names))
    for s in run_all.STEPS:
        assert s.writes and all(w.startswith("reports/") for w in s.writes)


def test_every_module_exists_or_is_known_to_be_pending():
    for s in run_all.STEPS:
        assert s.code.exists() or s.module in PENDING, f"{s.module}: no such file; check the name"


def test_clinic_steps_need_the_clinic_folder(tmp_path):
    without = {s.name: why for s, why in run_all.plan()}
    assert without["clinic"].startswith("clinic step") and without["requantise_clinic"].startswith("clinic step")
    with_dir = {s.name: why for s, why in run_all.plan(clinic_dir=tmp_path)}
    assert not (with_dir["clinic"] or "").startswith("clinic step")


def test_a_file_written_earlier_in_the_run_counts_as_present():
    step = run_all.Step("x", "cardio.run_all", needs=("reports/not_written_yet.csv",), model=False)
    assert run_all.skip_reason(step, None).startswith("data not on this laptop")
    assert run_all.skip_reason(step, None, produced={"reports/not_written_yet.csv"}) is None


def test_only_and_from_pick_steps_in_order():
    assert [s.name for s, _ in run_all.plan(only=["edge_budget"])] == ["edge_budget"]
    names = [s.name for s, _ in run_all.plan(start="agreement")]
    assert names[0] == "agreement" and "cv_intervals" not in names
    with pytest.raises(ValueError):
        run_all.plan(only=["no_such_step"])


def test_final_refuses_to_start_when_code_is_missing(monkeypatch, capsys):
    fake = run_all.Step("future_step", "cardio.no_such_module", writes=("reports/future.json",), model=False)
    monkeypatch.setattr(run_all, "STEPS", (fake,))
    assert run_all.skip_reason(fake, None) == run_all.NOT_YET
    assert run_all.main(["--final"]) == 2
    assert "future_step" in capsys.readouterr().out
