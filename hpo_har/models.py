"""Classifier factories. Every model is single-threaded so that timings are comparable
and many evaluations can run in parallel, one per CPU core."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .spaces import DEFAULTS


def make_model(name: str, params: dict | None, seed: int):
    p = dict(DEFAULTS[name] if params is None else params)
    if name == "SVM":
        # No probability calibration: ROC-AUC uses decision_function scores (one-vs-rest).
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(kernel="rbf", cache_size=500, decision_function_shape="ovr", **p))])
    if name == "RandomForest":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **p)
    if name == "XGBoost":
        import xgboost as xgb
        # No early stopping: n_estimators is a searched hyperparameter.
        return xgb.XGBClassifier(objective="multi:softprob", tree_method="hist", n_jobs=1,
                                 random_state=seed, verbosity=0, eval_metric="mlogloss", **p)
    raise ValueError(name)


def fit_predict(name: str, params: dict | None, seed: int, X_tr, y_tr, X_te, n_classes: int,
                want_scores: bool = False):
    """Fit and predict with integer labels 0..n_classes-1.

    Training folds may miss rare classes; labels are re-encoded for the fit and mapped
    back so that every model sees contiguous labels (XGBoost requires this). Scores
    (probabilities or decision values) are returned as an (n, n_classes) matrix.
    """
    present = np.unique(y_tr)
    y_fit = np.searchsorted(present, y_tr)
    model = make_model(name, params, seed)
    model.fit(X_tr, y_fit)
    pred = present[np.asarray(model.predict(X_te)).astype(int)]
    if not want_scores:
        return pred, None, False
    scores = np.zeros((len(X_te), n_classes))
    if name == "SVM":
        d = model.decision_function(X_te)
        if d.ndim == 1:                                     # two classes present
            d = np.column_stack([-d, d])
        scores[:] = d.min() - 1.0                           # classes unseen in training rank last
        scores[:, present] = d
        is_proba = False
    else:
        scores[:, present] = model.predict_proba(X_te)
        is_proba = True
    return pred, scores, is_proba
