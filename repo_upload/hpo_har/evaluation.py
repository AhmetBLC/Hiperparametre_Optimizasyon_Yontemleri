"""Subject-grouped cross-validation (the HPO objective) and test-set metrics."""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
                             log_loss, matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold

from .models import fit_predict


class EncodedData:
    """Integer-encoded view of a Dataset used by all optimisers."""

    def __init__(self, ds, n_folds: int):
        self.name = ds.name
        self.classes = list(ds.classes)
        cidx = {c: i for i, c in enumerate(self.classes)}
        self.X = ds.X_train
        self.y = np.array([cidx[c] for c in ds.y_train])
        self.g = ds.g_train
        self.X_test = ds.X_test
        self.y_test = np.array([cidx[c] for c in ds.y_test])
        self.g_test = ds.g_test
        self.n_classes = len(self.classes)
        # Deterministic subject-grouped folds, identical for every method and seed.
        self.folds = list(GroupKFold(n_splits=n_folds).split(self.X, self.y, self.g))


def macro_f1_present(y_true, y_pred) -> float:
    """Macro-F1 over the classes present in y_true (rare classes can be absent from a fold)."""
    labels = np.unique(y_true)
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def _subsample(idx: np.ndarray, y: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    if frac >= 1.0:
        return idx
    keep = []
    for c in np.unique(y[idx]):
        m = idx[y[idx] == c]
        k = max(1, int(round(frac * len(m))))
        keep.append(rng.choice(m, size=k, replace=False))
    return np.sort(np.concatenate(keep))


# ----------------------------------------------------------------------------- evaluation checkpoint
# Every finished evaluation of the running job is appended to a small JSONL file. If the session
# dies, the job is restarted from scratch with this cache: all optimisers are deterministic for a
# given seed, so they propose exactly the same configurations again and the stored results are
# returned instantly (with their original timings) until the point where the job was interrupted.
_CKPT = {"path": None, "cache": {}}


def set_checkpoint(path) -> int:
    """Activate (path) or deactivate (None) the evaluation checkpoint; returns #cached results."""
    _CKPT["path"], _CKPT["cache"] = (str(path) if path else None), {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    _CKPT["cache"][d["k"]] = d["r"]
                except (ValueError, KeyError):          # a line cut off by the crash
                    pass
    return len(_CKPT["cache"])


def _ckpt_key(model, params, seed, frac, frac_seed) -> str:
    return json.dumps([model, params, int(seed), round(float(frac), 9), int(frac_seed)], sort_keys=True)


def cv_evaluate(data: EncodedData, model: str, params: dict, seed: int,
                frac: float = 1.0, frac_seed: int = 0) -> dict:
    """Mean macro-F1 over the GroupKFold folds. frac < 1 trains on a stratified fraction of
    each training fold (low-fidelity evaluation used by BOHB); validation folds are complete."""
    key = _ckpt_key(model, params, seed, frac, frac_seed) if _CKPT["path"] else None
    if key is not None and key in _CKPT["cache"]:
        return dict(_CKPT["cache"][key])
    res = _cv_evaluate(data, model, params, seed, frac, frac_seed)
    if key is not None:
        _CKPT["cache"][key] = res
        with open(_CKPT["path"], "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": key, "r": res}) + "\n")
            f.flush()
            os.fsync(f.fileno())
    return res


def _cv_evaluate(data, model, params, seed, frac, frac_seed) -> dict:
    t_wall, t_cpu = time.perf_counter(), time.process_time()
    scores = []
    for k, (tr, va) in enumerate(data.folds):
        tr_used = _subsample(tr, data.y, frac, np.random.default_rng([frac_seed, k]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred, _, _ = fit_predict(model, params, seed, data.X[tr_used], data.y[tr_used],
                                     data.X[va], data.n_classes)
        scores.append(macro_f1_present(data.y[va], pred))
    return {"score": float(np.mean(scores)), "fold_scores": [round(s, 6) for s in scores],
            "cpu_s": time.process_time() - t_cpu, "wall_s": time.perf_counter() - t_wall}


def test_evaluate(data: EncodedData, model: str, params: dict | None, seed: int) -> dict:
    """Refit on the full training set and evaluate once on the held-out test subjects."""
    t_cpu = time.process_time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred, scores, is_proba = fit_predict(model, params, seed, data.X, data.y, data.X_test,
                                             data.n_classes, want_scores=True)
    y = data.y_test
    labels = np.arange(data.n_classes)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1_macro": macro_f1_present(y, pred),
        "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y, pred, labels=np.unique(y), average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, labels=np.unique(y), average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
    }
    present = np.unique(y)
    yb = (y[:, None] == present[None, :]).astype(int)      # one-vs-rest indicators (also for 2 classes)
    try:   # one-vs-rest AUC from probabilities (RF, XGB) or decision values (SVM, uncalibrated)
        out["roc_auc_ovr"] = roc_auc_score(yb, scores[:, present], average="macro")
    except ValueError:
        out["roc_auc_ovr"] = float("nan")
    if is_proba:
        p = np.clip(scores, 1e-15, 1.0)
        out["log_loss"] = log_loss(y, p / p.sum(axis=1, keepdims=True), labels=labels)
    else:
        out["log_loss"] = float("nan")       # SVM without Platt scaling produces no probabilities
    out = {k: float(v) for k, v in out.items()}
    out["refit_cpu_s"] = time.process_time() - t_cpu
    return {"metrics": out, "y_pred": pred.astype(np.int16)}
