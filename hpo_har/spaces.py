"""Search spaces, library defaults, grid construction and encoding helpers.

All six optimisers share exactly the same bounds. Continuous/integer optimisers sample
inside the bounds; Grid Search uses k points per dimension placed at the centres of k
equal-width cells spanning the same bounds (on the log scale where the space is log),
so that every grid covers the full space with equal resolution in all dimensions.

Bounds were widened relative to v1 so that library defaults lie strictly inside the
space (e.g. RandomForest max_features='sqrt' = 0.042 for 561 features, 0.08-0.11
for the 83-498 features of the raw-signal datasets).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Param:
    name: str
    kind: str          # "float" | "int"
    low: float
    high: float
    log: bool = False

    def decode(self, u: float):
        """Map u in [0, 1] to a parameter value."""
        u = float(np.clip(u, 0.0, 1.0))
        if self.log:
            v = math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        else:
            v = self.low + u * (self.high - self.low)
        if self.kind == "int":
            v = int(min(self.high, max(self.low, math.floor(v + 0.5))))
        return v

    def encode(self, v) -> float:
        """Inverse of decode (position of a value in [0, 1])."""
        if self.log:
            return (math.log(v) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        return (v - self.low) / (self.high - self.low)


SPACES: dict[str, list[Param]] = {
    "SVM": [
        Param("C", "float", 1e-2, 1e3, log=True),
        Param("gamma", "float", 1e-5, 1e0, log=True),
    ],
    "RandomForest": [
        Param("n_estimators", "int", 50, 500, log=True),
        Param("max_depth", "int", 2, 50, log=True),
        Param("max_features", "float", 0.01, 1.0, log=True),
        Param("min_samples_leaf", "int", 1, 30, log=True),
    ],
    "XGBoost": [
        Param("n_estimators", "int", 50, 500, log=True),
        Param("max_depth", "int", 1, 10),
        Param("learning_rate", "float", 5e-3, 0.5, log=True),
        Param("subsample", "float", 0.5, 1.0),
        Param("colsample_bytree", "float", 0.3, 1.0),
    ],
}

# Library defaults (scikit-learn >= 1.3, XGBoost >= 2.0). "gamma='scale'" equals
# 1 / n_features on standardised inputs, which lies inside [1e-5, 1].
DEFAULTS = {
    "SVM": {"C": 1.0, "gamma": "scale"},
    "RandomForest": {"n_estimators": 100, "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 1},
    "XGBoost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.3, "subsample": 1.0,
                "colsample_bytree": 1.0},
}


def decode_vector(model: str, u) -> dict:
    return {p.name: p.decode(x) for p, x in zip(SPACES[model], u)}


def encode_params(model: str, params: dict) -> np.ndarray:
    return np.array([SPACES[model][i].encode(params[p.name]) for i, p in enumerate(SPACES[model])])


def grid_levels(n_dims: int, target: int) -> list[int]:
    """Balanced number of grid levels per dimension whose product is closest to target.

    Levels differ by at most one between dimensions; extra levels go to the first
    dimensions in the declared parameter order (no problem-specific prior is used).
    """
    best, best_err = None, None
    for base in range(1, target + 1):
        for extra in range(0, n_dims + 1):
            levels = [base + 1] * extra + [base] * (n_dims - extra)
            prod = math.prod(levels)
            err = abs(prod - target)
            if best_err is None or err < best_err or (err == best_err and prod < math.prod(best)):
                best, best_err = levels, err
        if base ** n_dims > 2 * target:
            break
    return best


def build_grid(model: str, target: int) -> list[dict]:
    space = SPACES[model]
    levels = grid_levels(len(space), target)
    axes = [[(i + 0.5) / k for i in range(k)] for k in levels]   # cell centres in [0, 1]
    grid, seen = [], set()
    for u in itertools.product(*axes):
        p = decode_vector(model, u)
        key = tuple(sorted(p.items()))
        if key not in seen:                                      # integer rounding can collide
            seen.add(key)
            grid.append(p)
    return grid


def distributions(model: str):
    """Optuna distributions for the space (imported lazily)."""
    import optuna
    out = {}
    for p in SPACES[model]:
        if p.kind == "int":
            out[p.name] = optuna.distributions.IntDistribution(int(p.low), int(p.high), log=p.log)
        else:
            out[p.name] = optuna.distributions.FloatDistribution(p.low, p.high, log=p.log)
    return out


def boundary_distance(model: str, params: dict) -> dict:
    """Normalised distance of each selected value to the nearest bound (0 = on a bound)."""
    out = {}
    for p in SPACES[model]:
        u = p.encode(params[p.name])
        out[p.name] = float(min(u, 1.0 - u))
    return out
