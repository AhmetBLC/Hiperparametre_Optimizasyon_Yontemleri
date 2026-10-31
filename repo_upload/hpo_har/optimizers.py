"""The six hyperparameter optimisation methods.

Every method spends at most `budget` full-evaluation equivalents, where one unit is a
complete 3-fold GroupKFold evaluation. All methods are sequential, so the first B units
of a run with budget 100 are exactly what a run with budget B would have done; this
gives exact any-time (budget-performance) curves from a single run.

  Grid    cell-centred grid of ~target points (spaces.build_grid)
  Random  Optuna RandomSampler
  TPE     Optuna TPESampler (multivariate, 10 random start-up trials)   [Bayesian optimisation]
  CMAES   Optuna CmaEsSampler (with margin for integer parameters)
  GA      real-coded genetic algorithm (this module)
  BOHB    TPE + Hyperband pruning over training-set fidelity 1/9, 1/3, 1  (Falkner et al., 2018)
"""
from __future__ import annotations

import numpy as np

from .evaluation import EncodedData, cv_evaluate
from .spaces import SPACES, build_grid, decode_vector, distributions


class Log:
    def __init__(self):
        self.records = []
        self.cum_cost = 0.0

    def add(self, params, res, cost, fidelity=1.0, **extra):
        self.cum_cost += cost
        rec = {"i": len(self.records) + 1, "params": params, "score": res["score"],
               "fold_scores": res["fold_scores"], "cost": cost, "cum_cost": round(self.cum_cost, 6),
               "fidelity": fidelity, "cpu_s": round(res["cpu_s"], 4), "wall_s": round(res["wall_s"], 4)}
        rec.update(extra)
        self.records.append(rec)
        return rec


def best_at_budget(records, budget: float):
    """Best full-fidelity configuration found within the first `budget` units."""
    cand = [r for r in records if r["fidelity"] >= 1.0 and r["cum_cost"] <= budget + 1e-9]
    if not cand:
        return None
    return max(cand, key=lambda r: (r["score"], -r["i"]))      # ties -> earliest


# ----------------------------------------------------------------------------- Grid
def run_grid(data: EncodedData, model: str, seed: int, target: int, **_):
    log = Log()
    for p in build_grid(model, target):
        log.add(p, cv_evaluate(data, model, p, seed), 1.0)
    return log.records, {"grid_size": len(log.records)}


# ----------------------------------------------------------------------------- Optuna-based
def _optuna():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def _sampler(name: str, seed: int):
    optuna = _optuna()
    if name == "Random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name in ("TPE", "BOHB"):
        return optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=10,
                                          warn_independent_sampling=False)
    if name == "CMAES":
        return optuna.samplers.CmaEsSampler(seed=seed, with_margin=True, warn_independent_sampling=False)
    raise ValueError(name)


def run_optuna(data: EncodedData, model: str, seed: int, budget: int, method: str, **_):
    optuna = _optuna()
    dists = distributions(model)
    study = optuna.create_study(direction="maximize", sampler=_sampler(method, seed))
    log = Log()
    while log.cum_cost + 1.0 <= budget + 1e-9:
        trial = study.ask(dists)
        p = dict(trial.params)
        res = cv_evaluate(data, model, p, seed)
        study.tell(trial, res["score"])
        log.add(p, res, 1.0, trial=trial.number)
    return log.records, {}


def run_bohb(data: EncodedData, model: str, seed: int, budget: int, eta: int = 3, **_):
    optuna = _optuna()
    dists = distributions(model)
    max_r = eta ** 2
    rungs = [1, eta, max_r]                                   # resources -> fractions 1/9, 1/3, 1
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=max_r, reduction_factor=eta)
    study = optuna.create_study(direction="maximize", sampler=_sampler("BOHB", seed), pruner=pruner)
    log = Log()
    while log.cum_cost + 1.0 / max_r <= budget + 1e-9:
        trial = study.ask(dists)
        p = dict(trial.params)
        state, value, reported = "stop", None, False
        for r in rungs:
            frac = r / max_r
            if log.cum_cost + frac > budget + 1e-9:
                break                                         # budget exhausted inside this trial
            res = cv_evaluate(data, model, p, seed, frac=frac, frac_seed=seed * 100003 + trial.number)
            log.add(p, res, frac, fidelity=frac, trial=trial.number, rung=r)
            trial.report(res["score"], r)
            reported = True
            if r == max_r:
                state, value = "complete", res["score"]
                break
            if trial.should_prune():
                state = "pruned"
                break
        if state == "complete":
            study.tell(trial, value)
        elif state == "pruned":
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        else:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED if reported else optuna.trial.TrialState.FAIL)
            break
    n_full = sum(1 for r in log.records if r["fidelity"] >= 1.0)
    return log.records, {"n_trials": len(study.trials), "n_full_evaluations": n_full}


# ----------------------------------------------------------------------------- Genetic algorithm
def run_ga(data: EncodedData, model: str, seed: int, budget: int, pop_size: int = 10,
           tournament_k: int = 3, p_crossover: float = 0.9, blx_alpha: float = 0.5,
           mutation_sigma: float = 0.1, elitism: int = 1, **_):
    """Real-coded GA on the unit hypercube (genes decoded with spaces.Param.decode).

    Generational replacement with elitism, tournament selection, BLX-alpha crossover
    (Eshelman & Schaffer, 1993) and Gaussian mutation applied to each gene with
    probability 1/D. The initial population costs pop_size evaluations and every later
    generation pop_size - elitism; the last generation is truncated to fit the budget.
    """
    rng = np.random.default_rng(seed)
    D = len(SPACES[model])
    log = Log()
    gen_stats = []

    def evaluate(pop, g):
        fit = np.empty(len(pop))
        for j, u in enumerate(pop):
            p = decode_vector(model, u)
            res = cv_evaluate(data, model, p, seed)
            log.add(p, res, 1.0, generation=g)
            fit[j] = res["score"]
        return fit

    def stats(g, pop, fit):
        gen_stats.append({"generation": g, "best": float(fit.max()), "mean": float(fit.mean()),
                          "diversity": float(pop.std(axis=0).mean()), "n_evals": len(log.records)})

    pop = rng.random((min(pop_size, budget), D))
    fit = evaluate(pop, 0)
    stats(0, pop, fit)
    g = 0
    while len(log.records) < budget:
        g += 1
        order = np.argsort(-fit)
        elite = order[:elitism]

        def tournament():
            c = rng.choice(len(pop), size=min(tournament_k, len(pop)), replace=False)
            return pop[c[np.argmax(fit[c])]]

        n_new = min(pop_size - elitism, budget - len(log.records))
        children = []
        while len(children) < n_new:
            a, b = tournament(), tournament()
            if rng.random() < p_crossover:
                lo, hi = np.minimum(a, b), np.maximum(a, b)
                span = hi - lo
                child = rng.uniform(lo - blx_alpha * span, hi + blx_alpha * span)
            else:
                child = a.copy()
            mut = rng.random(D) < 1.0 / D
            child = child + mut * rng.normal(0.0, mutation_sigma, D)
            children.append(np.clip(child, 0.0, 1.0))
        children = np.array(children)
        # Elites keep their fitness (evaluation is deterministic for a fixed seed), so
        # each generation spends pop_size - elitism evaluations.
        child_fit = evaluate(children, g)
        pop = np.vstack([pop[elite], children])
        fit = np.concatenate([fit[elite], child_fit])
        stats(g, pop, fit)
    return log.records, {"generations": gen_stats, "pop_size": pop_size, "n_gen": len(gen_stats)}


METHODS = {"Grid": run_grid, "Random": run_optuna, "TPE": run_optuna, "CMAES": run_optuna,
           "GA": run_ga, "BOHB": run_bohb}
