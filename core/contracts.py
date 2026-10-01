"""What each model takes in and gives back, defined once for both teams.

The heart-sound model, the chest X-ray model, late fusion and the visit
decision pass these objects to each other. Agreeing on them now lets the two
teams work in parallel and still plug together at the end.

These are data only. No model, fusion rule or decision rule lives here: those
are the teams' work, in cardio/ and lung/. What does live here are the checks,
so a bad value fails where it is made rather than three steps later:

  * probabilities lie in [0, 1], and a model's class probabilities sum to 1
  * fusion weights sum to 1, and the metadata vote never exceeds its cap
  * patient metadata is either a plausible value or None, never a guess

Usage:
    from core.contracts import ModelOutput, PatientMetadata, FusedOutput

    out = ModelOutput("heart", {"Normal": 0.7, "AS": 0.1, "MS": 0.1,
                                "MR": 0.05, "MVP": 0.05},
                      secondary={"heart_rate_bpm": 78.0})
    out.p_abnormal          # 0.3
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

Modality = Literal["heart", "xray"]

CLASSES: dict[str, tuple[str, ...]] = {
    "heart": ("Normal", "AS", "MS", "MR", "MVP"),
    "xray": ("Normal", "Pneumonia"),
}

# Symptoms the health worker can tick. Extend this list on a shared- branch,
# so both teams' metadata classifiers see the same vocabulary.
SYMPTOMS = frozenset({"breathlessness", "chest_pain", "fever", "cough"})

# The three late-fusion votes, and the most weight the metadata vote may get.
VOTES = ("cnn", "features", "metadata")
METADATA_WEIGHT_CAP = 0.3

_TOL = 1e-6


def _check_probability(name: str, p: float) -> None:
    if not (isinstance(p, (int, float)) and 0.0 <= p <= 1.0):
        raise ValueError(f"{name} must be a probability in [0, 1], got {p!r}")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PatientMetadata:
    """Patient details and symptoms, as entered by the health worker.

    Every field is optional; a value that was not recorded is None. Datasets
    that record an age group instead of an age convert it when loading.

    Only facts about the patient belong here. Facts about how the input was
    acquired (an X-ray's view position, the stethoscope used) are deliberately
    absent: they describe the equipment, not the patient, and a classifier
    would learn them as a shortcut.
    """

    age_years: Optional[float] = None
    sex: Optional[Literal["female", "male"]] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    pregnant: Optional[bool] = None
    symptoms: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.age_years is not None and not 0 <= self.age_years <= 120:
            raise ValueError(f"age_years out of range: {self.age_years}")
        if self.sex is not None and self.sex not in ("female", "male"):
            raise ValueError(f"sex must be 'female', 'male' or None, got {self.sex!r}")
        if self.height_cm is not None and not 20 <= self.height_cm <= 250:
            raise ValueError(f"height_cm out of range: {self.height_cm}")
        if self.weight_kg is not None and not 0.5 <= self.weight_kg <= 300:
            raise ValueError(f"weight_kg out of range: {self.weight_kg}")
        unknown = set(self.symptoms) - SYMPTOMS
        if unknown:
            raise ValueError(f"unknown symptoms {sorted(unknown)}; known: {sorted(SYMPTOMS)}")
        object.__setattr__(self, "symptoms", frozenset(self.symptoms))


# ---------------------------------------------------------------------------
# One model's answer
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelOutput:
    """What one screening model gives back for one input.

    probabilities -- one entry per class of that modality, summing to 1
    secondary     -- measured secondary characteristics, name -> value; a
                     characteristic that could not be measured is left out,
                     never filled with a guess
    quality_ok    -- False when the input failed the quality check, so the
                     visit decision can ignore this test
    """

    modality: Modality
    probabilities: dict[str, float]
    secondary: dict[str, float] = field(default_factory=dict)
    quality_ok: bool = True

    def __post_init__(self) -> None:
        if self.modality not in CLASSES:
            raise ValueError(f"modality must be one of {sorted(CLASSES)}, got {self.modality!r}")
        expected = set(CLASSES[self.modality])
        if set(self.probabilities) != expected:
            raise ValueError(f"{self.modality} probabilities need exactly the classes "
                             f"{sorted(expected)}, got {sorted(self.probabilities)}")
        for cls, p in self.probabilities.items():
            _check_probability(f"probabilities[{cls!r}]", p)
        total = sum(self.probabilities.values())
        if abs(total - 1.0) > _TOL:
            raise ValueError(f"class probabilities must sum to 1, got {total:.6f}")
        for name, v in self.secondary.items():
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                raise ValueError(f"secondary[{name!r}] must be a finite number, got {v!r}")

    @property
    def p_abnormal(self) -> float:
        """Probability that the input is not normal."""
        return 1.0 - self.probabilities["Normal"]


# ---------------------------------------------------------------------------
# One branch after late fusion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FusedOutput:
    """One branch (heart or X-ray) after late fusion.

    votes   -- each vote's probability that the input is abnormal, keyed by
               "cnn", "features" and "metadata"; a vote whose input is missing
               is left out
    weights -- the same keys, summing to 1 (renormalised when a vote is left
               out); the metadata weight never exceeds METADATA_WEIGHT_CAP
    """

    modality: Modality
    p_abnormal: float
    votes: dict[str, float]
    weights: dict[str, float]

    def __post_init__(self) -> None:
        if self.modality not in CLASSES:
            raise ValueError(f"modality must be one of {sorted(CLASSES)}, got {self.modality!r}")
        _check_probability("p_abnormal", self.p_abnormal)
        if "cnn" not in self.votes:
            raise ValueError("the CNN vote is always present")
        unknown = set(self.votes) - set(VOTES)
        if unknown:
            raise ValueError(f"unknown votes {sorted(unknown)}; known: {list(VOTES)}")
        if set(self.weights) != set(self.votes):
            raise ValueError(f"weights {sorted(self.weights)} must match votes {sorted(self.votes)}")
        for name, p in self.votes.items():
            _check_probability(f"votes[{name!r}]", p)
        for name, w in self.weights.items():
            _check_probability(f"weights[{name!r}]", w)
        total = sum(self.weights.values())
        if abs(total - 1.0) > _TOL:
            raise ValueError(f"fusion weights must sum to 1, got {total:.6f}")
        if self.weights.get("metadata", 0.0) > METADATA_WEIGHT_CAP + _TOL:
            raise ValueError(f"metadata weight {self.weights['metadata']} exceeds the cap "
                             f"of {METADATA_WEIGHT_CAP}")


# ---------------------------------------------------------------------------
# One decision per visit
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VisitDecision:
    """The single refer-or-not answer for one patient visit.

    p_heart, p_xray -- each branch's fused probability; None if that test was
                       not done or failed the quality check
    p_any           -- probability that at least one finding is present
    cost_ratio      -- C_miss / C_refer, set by the clinic
    """

    refer: bool
    p_any: float
    cost_ratio: float
    p_heart: Optional[float] = None
    p_xray: Optional[float] = None

    def __post_init__(self) -> None:
        if self.p_heart is None and self.p_xray is None:
            raise ValueError("a visit needs at least one usable test")
        for name in ("p_heart", "p_xray"):
            p = getattr(self, name)
            if p is not None:
                _check_probability(name, p)
        _check_probability("p_any", self.p_any)
        if not self.cost_ratio > 0:
            raise ValueError(f"cost_ratio must be positive, got {self.cost_ratio}")
