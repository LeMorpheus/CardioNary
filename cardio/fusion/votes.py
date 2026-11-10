"""Feature vote and metadata vote logistic regression models.

Task 2: Feature vote from secondary characteristics (Yaseen).
Task 3: Metadata vote from patient demographic details (PhysioNet).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cardio.eval.intervals import metric_intervals

AGE_MAP = {
    "Neonate": 0,
    "Infant": 1,
    "Child": 2,
    "Adolescent": 3,
    "Young adult": 4,
    "Young Adult": 4,
}


def _num(val):
    return float(val) if val not in (None, "", "nan") else np.nan


def new_vote(random_state=0):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state),
    )


def run_feature_vote():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    features_csv = reports_dir / "heart_features_yaseen.csv"

    if not features_csv.exists():
        raise FileNotFoundError(f"Missing {features_csv}. Run cardio.model.features --dataset yaseen first.")

    df = pd.read_csv(features_csv)
    feature_cols = [c for c in df.columns if c not in ["patient_id", "fold", "label"]]
    X = df[feature_cols].to_numpy(dtype=float)
    y = (df.label != 0).astype(int).to_numpy()

    p_lr = np.zeros(len(df))
    for k in range(5):
        train = (df.fold != k).to_numpy()
        test = (df.fold == k).to_numpy()
        vote = new_vote().fit(X[train], y[train])
        p_lr[test] = vote.predict_proba(X[test])[:, 1]

    intervals = metric_intervals(y, p_lr, df.patient_id.to_numpy(), ["auroc"])
    print(f"Feature vote OOF AUROC: {intervals['auroc']}")

    out_df = pd.DataFrame({
        "patient_id": df.patient_id,
        "fold": df.fold,
        "label": df.label,
        "p_lr": p_lr,
    })
    out_path = reports_dir / "heart_plr_yaseen.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved out-of-fold feature vote probabilities to {out_path}")

    # Fit once more on all Yaseen clips for use on unseen datasets and in tool
    full_model = new_vote().fit(X, y)
    return full_model, feature_cols


def run_metadata_vote():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path("data/manifests/circor_manifest.csv")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}.")

    c = pd.read_csv(manifest_path)
    c["murmur"] = c["labels"].map(lambda s: json.loads(s)["murmur"])
    meta = c["metadata"].map(json.loads)
    c["age"] = meta.map(lambda d: AGE_MAP.get(d.get("Age"), np.nan))
    c["sex"] = meta.map(lambda d: {"Female": 0, "Male": 1}.get(d.get("Sex"), np.nan))
    c["height"] = meta.map(lambda d: _num(d.get("Height")))
    c["weight"] = meta.map(lambda d: _num(d.get("Weight")))
    c["pregnant"] = meta.map(lambda d: {"False": 0, "True": 1}.get(d.get("Pregnancy status"), np.nan))

    def patient_label(visits):
        known = set(visits) - {"Unknown"}
        return np.nan if not known else int("Present" in known)

    patients = c.groupby("patient_id").agg(
        label=("murmur", patient_label),
        age=("age", "first"),
        sex=("sex", "first"),
        height=("height", "first"),
        weight=("weight", "first"),
        pregnant=("pregnant", "first"),
    ).dropna(subset=["label"])

    patients["label"] = patients["label"].astype(int)

    # 5 stratified folds on patient label
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    folds = np.zeros(len(patients), dtype=int)
    for fold_idx, (_, test_idx) in enumerate(skf.split(patients, patients["label"])):
        folds[test_idx] = fold_idx
    patients["fold"] = folds

    # Save cardio/data/circor_folds.csv (patient_id, fold)
    folds_path = Path("cardio/data/circor_folds.csv")
    folds_df = pd.DataFrame({
        "patient_id": patients.index,
        "fold": patients["fold"].values,
    })
    folds_df.to_csv(folds_path, index=False)
    print(f"Saved 5 stratified patient folds to {folds_path}")

    # Compute out-of-fold pM
    meta_cols = ["age", "sex", "height", "weight", "pregnant"]
    X = patients[meta_cols].to_numpy(dtype=float)
    y = patients["label"].to_numpy()
    p_m = np.zeros(len(patients))

    for k in range(5):
        train = (patients["fold"] != k).to_numpy()
        test = (patients["fold"] == k).to_numpy()
        model = new_vote().fit(X[train], y[train])
        p_m[test] = model.predict_proba(X[test])[:, 1]

    patients["p_m"] = p_m
    intervals = metric_intervals(y, p_m, patients.index.to_numpy(), ["auroc"])
    print(f"Metadata vote OOF AUROC: {intervals['auroc']}")

    out_df = pd.DataFrame({
        "patient_id": patients.index,
        "fold": patients["fold"].values,
        "label": patients["label"].values,
        "p_m": p_m,
    })
    out_path = reports_dir / "heart_pm_physionet.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved out-of-fold metadata vote probabilities to {out_path}")

    full_model = new_vote().fit(X, y)
    return full_model, meta_cols


def main():
    print("=== Task 2: Feature vote ===")
    run_feature_vote()
    print("\n=== Task 3: Metadata vote ===")
    run_metadata_vote()


if __name__ == "__main__":
    main()
