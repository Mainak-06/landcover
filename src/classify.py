"""Random Forest / SVM pixel classifier."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
from .config import CLASS_NAMES, NODATA_CLASS
from .features import feature_stack, weak_labels


def build_model(kind: str = "Random Forest", seed: int = 42):
    if kind == "SVM":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5, gamma="scale"))
    return RandomForestClassifier(n_estimators=120, max_depth=18, n_jobs=-1,
                                  random_state=seed, class_weight="balanced")


def sample_training(bands, labels=None, per_class: int = 3000, seed: int = 42):
    """Draw a class-balanced sample. Falls back to weak labels."""
    rng = np.random.default_rng(seed)
    X = feature_stack(bands)
    y = weak_labels(bands) if labels is None else labels
    Xs, ys = [], []
    for c in CLASS_NAMES:
        idx = np.flatnonzero(y.ravel() == c)
        if idx.size == 0:
            continue
        pick = rng.choice(idx, min(per_class, idx.size), replace=False)
        Xs.append(X.reshape(-1, X.shape[-1])[pick])
        ys.append(np.full(pick.size, c))
    if not Xs:
        raise ValueError("No training pixels found - check band order / data.")
    return np.vstack(Xs), np.concatenate(ys)


def train(bands, kind="Random Forest", labels=None, per_class=3000, seed=42):
    X, y = sample_training(bands, labels, per_class, seed)
    strat = y if np.bincount(y).min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=strat)
    model = build_model(kind, seed).fit(Xtr, ytr)
    pred = model.predict(Xte)
    present = sorted(set(yte) | set(pred))
    metrics = {
        "accuracy": float(accuracy_score(yte, pred)),
        "kappa": float(cohen_kappa_score(yte, pred)),
        "labels": present,
        "confusion": confusion_matrix(yte, pred, labels=present),
        "report": classification_report(yte, pred, labels=present,
                                        target_names=[CLASS_NAMES[i] for i in present],
                                        output_dict=True, zero_division=0),
    }
    return model, metrics


def predict_map(model, bands, valid=None, chunk: int = 200_000) -> np.ndarray:
    X = feature_stack(bands)
    h, w, f = X.shape
    flat = np.nan_to_num(X.reshape(-1, f), nan=0.0, posinf=0.0, neginf=0.0)
    out = np.empty(flat.shape[0], np.uint8)
    for s in range(0, flat.shape[0], chunk):
        out[s:s + chunk] = model.predict(flat[s:s + chunk])
    out = out.reshape(h, w)
    if valid is not None:
        out[~valid] = NODATA_CLASS
    return out


def feature_importance(model):
    from .features import indices
    names = ["Blue", "Green", "Red", "NIR", "SWIR1", "NDVI", "NDWI", "NDBI", "MNDWI"]
    rf = model if hasattr(model, "feature_importances_") else None
    return dict(zip(names, rf.feature_importances_)) if rf is not None else None
