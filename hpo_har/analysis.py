"""Collect job results and produce the tables and statistics of the paper.

Statistical design
------------------
* Blocks are (dataset x model) combinations; the unit within a block is a seed.
* Across blocks: Friedman test with Iman-Davenport correction, Kendall's W, Nemenyi
  critical difference, and Holm-corrected Wilcoxon signed-rank tests on the seed-averaged
  scores (Demsar, 2006; Garcia & Herrera, 2008).
* Within blocks: two-sided Mann-Whitney U tests over seeds against Random Search
  (Holm-corrected within each block) -> win / tie / loss counts.
* Test-set uncertainty: subject-level cluster bootstrap (windows of the same subject are
  resampled together), giving 95% CIs for macro-F1 and for differences to Random Search.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config
from .datasets import build_dataset
from .runner import (DETERMINISTIC_MODELS, dataset_fingerprints, enumerate_jobs, job_path, read_json_gz,
                     reconcile_results)
from .spaces import SPACES, boundary_distance, build_grid, grid_levels

HPO_METHODS = ["Grid", "Random", "TPE", "CMAES", "GA", "BOHB"]
METHOD_LABEL = {"Grid": "Grid Search", "Random": "Random Search", "TPE": "Bayesian opt. (TPE)",
                "CMAES": "CMA-ES", "GA": "Genetic algorithm", "BOHB": "BOHB", "Default": "Default"}
DATASET_LABEL = {"uci_har": "UCI HAR", "wisdm": "WISDM", "pamap2": "PAMAP2",
                 "motionsense": "MotionSense", "harth": "HARTH"}
MODEL_LABEL = {"SVM": "SVM (RBF)", "RandomForest": "Random Forest", "XGBoost": "XGBoost"}


# ----------------------------------------------------------------------------- loading
class Results:
    def __init__(self, cfg: Config, extra_dirs: list | None = None, validate: bool = True):
        """validate=False reads every job file without checking it against the local dataset cache
        (for re-analysing downloaded result files on a machine without the data)."""
        self.cfg = cfg
        dirs = [cfg.results_dir / "jobs"] + [Path(d) for d in (extra_dirs or [])]
        if validate:
            fps = dataset_fingerprints(cfg)
            reconcile_results(cfg, fps)
            valid = {job_path(cfg, j["id"], fps[j["dataset"]]).name for j in enumerate_jobs(cfg)}
        else:
            valid = {p.name for d in dirs if Path(d).exists() for p in Path(d).glob("*.json.gz")}
        files = sorted({p.name: p for d in dirs if Path(d).exists() for p in Path(d).glob("*.json.gz")
                        if p.name in valid}.values())
        fin, rec, gens = [], [], []
        self.preds = {}
        for f in files:
            j = read_json_gz(f)
            if j.get("status") != "ok":
                continue
            base = {k: j[k] for k in ["dataset", "model", "method", "tag", "seed"]}
            base["processor"] = (j.get("env") or {}).get("processor") or "unknown"
            for r in j["records"]:
                rec.append({**base, **{k: r.get(k) for k in ["i", "score", "cost", "cum_cost", "fidelity",
                                                              "cpu_s", "generation"]},
                            "params": json.dumps(r["params"]) if r["params"] is not None else None})
            for fz in j["finals"]:
                nominal = j.get("target") if j["method"] == "Grid" else (0 if j["method"] == "Default" else fz["budget"])
                row = {**base, "budget": fz["budget"], "budget_nominal": nominal, "cv_score": fz["cv_score"],
                       "best_index": fz["best_index"], "params": json.dumps(fz["params"]),
                       "search_cpu_s": fz["search_cpu_s"], "search_wall_s": fz["search_wall_s"],
                       **{f"test_{k}": v for k, v in fz["test"].items()}}
                fin.append(row)
                self.preds[(j["dataset"], j["model"], j["method"], nominal, j["seed"])] = np.array(fz["y_pred"])
            for g in j.get("extra", {}).get("generations", []):
                gens.append({**base, **g})
        self.finals = self._broadcast(pd.DataFrame(fin))
        self.records = pd.DataFrame(rec)
        self.hw_factors = hardware_factors(self.records)
        self._normalise_cpu()
        self.generations = pd.DataFrame(gens)
        self.datasets = sorted(self.finals.dataset.unique(), key=lambda d: list(DATASET_LABEL).index(d)) \
            if len(self.finals) else []
        self.models = [m for m in ["SVM", "RandomForest", "XGBoost"] if len(self.finals) and m in set(self.finals.model)]
        self.methods = [m for m in HPO_METHODS if len(self.finals) and m in set(self.finals.method)]
        print(f"loaded {len(files)} of {len(valid)} job files: {len(self.finals)} final rows, "
              f"{len(self.records)} evaluations")
        if len(files) < len(valid):
            print(f"  note: {len(valid) - len(files)} jobs are not finished yet - results are partial")

    def _normalise_cpu(self):
        """CPU seconds rescaled to the reference processor (see hardware_factors)."""
        f = self.hw_factors
        key = lambda d: list(zip(d.dataset, d.model, d.processor))  # noqa: E731
        if len(self.records):
            self.records["cpu_s_norm"] = self.records.cpu_s * [f.get(k, 1.0) for k in key(self.records)]
        if len(self.finals):
            self.finals["search_cpu_norm"] = self.finals.search_cpu_s * [f.get(k, 1.0) for k in key(self.finals)]

    def _broadcast(self, df: pd.DataFrame) -> pd.DataFrame:
        """SVM Grid/Default runs are deterministic and were run with one seed only."""
        if df.empty:
            return df
        seeds = sorted(set(self.cfg.seeds) | set(df.seed))
        extra = []
        m = df.model.isin(DETERMINISTIC_MODELS) & df.method.isin(["Grid", "Default"])
        for _, r in df[m].iterrows():
            for s in seeds:
                if s != r.seed:
                    extra.append(dict(r, seed=s))
                    self.preds[(r.dataset, r.model, r.method, r.budget_nominal, s)] = \
                        self.preds[(r.dataset, r.model, r.method, r.budget_nominal, r.seed)]
        return pd.concat([df, pd.DataFrame(extra)], ignore_index=True) if extra else df

    def at_budget(self, budget: int, include_default: bool = False) -> pd.DataFrame:
        """Final rows at a nominal budget (Grid uses the grid whose target equals budget)."""
        d = self.finals[self.finals.budget_nominal == budget]
        if include_default:
            d = pd.concat([d, self.finals[self.finals.method == "Default"]])
        return d

    def budgets(self):
        """Checkpoint budgets that have results (Grid targets coincide with checkpoints)."""
        have = set(self.finals.budget_nominal.unique()) if len(self.finals) else set()
        return [b for b in self.cfg.checkpoints if b in have]


# ----------------------------------------------------------------------------- helpers
def hardware_factors(records: pd.DataFrame, min_pairs: int = 10) -> dict:
    """Factors that rescale CPU seconds measured on each processor to the most common processor.

    Jobs may run on different machines (e.g. several Colab sessions). Because all optimisers
    with the same seed share start-up configurations, many identical evaluations (same dataset,
    model, seed, configuration and fidelity) exist on more than one processor. The factor for a
    (dataset, model, processor) is the median CPU-time ratio over those matched evaluations;
    with fewer than min_pairs matches the model-level geometric mean is used, else 1.
    """
    if records.empty or "processor" not in records or records.processor.nunique() < 2:
        return {}
    ref = records.processor.value_counts().idxmax()
    r = records.dropna(subset=["params"])
    r = r.assign(key=r.dataset + "|" + r.model + "|" + r.seed.astype(str) + "|" + r.params + "|" +
                 r.fidelity.round(6).astype(str))
    med = r.groupby(["key", "processor"]).cpu_s.median().unstack()
    out, by_model = {}, {}
    rows = []
    for proc in med.columns:
        if proc == ref:
            continue
        pair = med[[proc, ref]].dropna()
        pair = pair[(pair[proc] > 0) & (pair[ref] > 0)]
        ratio = pair[ref] / pair[proc]                       # multiply proc seconds by this
        ds = pair.index.str.split("|").str[0]
        mod = pair.index.str.split("|").str[1]
        for m in set(mod):
            sel = ratio[mod == m]
            if len(sel):
                by_model[(m, proc)] = float(np.exp(np.log(sel).mean()))
        for d, m in set(zip(r.dataset, r.model)):
            sel = ratio[(ds == d) & (mod == m)]
            if len(sel) >= min_pairs:
                out[(d, m, proc)] = float(sel.median())
            elif (m, proc) in by_model:
                out[(d, m, proc)] = by_model[(m, proc)]
            rows.append({"dataset": d, "model": m, "processor": proc, "matched_evaluations": int(len(sel)),
                         "factor": out.get((d, m, proc), 1.0)})
    hardware_factors.table = pd.DataFrame(rows).assign(reference=ref) if rows else pd.DataFrame()
    return out


def cv_test_agreement(res, budget: int, n_perm: int = 10000) -> dict:
    """Does a method's advantage on the CV objective carry over to the held-out subjects?"""
    cv = block_matrix(res, budget, "cv_score")
    te = block_matrix(res, budget, "test_f1_macro")
    cols = [c for c in cv.columns if c in te.columns]
    cv, te = cv[cols], te[cols]
    dcv = cv.sub(cv.mean(axis=1), axis=0).stack()
    dte = te.sub(te.mean(axis=1), axis=0).stack()
    rho, p = stats.spearmanr(dcv.values, dte.values)
    # The 6 x blocks points are not independent (deviations sum to zero within a block), so the
    # reference distribution is obtained by permuting the method labels of the test deviations
    # within each block (two-sided, add-one estimate).
    rng = np.random.default_rng(2026)
    Ccv = cv.sub(cv.mean(axis=1), axis=0).values
    T = te.sub(te.mean(axis=1), axis=0).values
    ok = ~(np.isnan(Ccv).any(axis=1) | np.isnan(T).any(axis=1))
    Ccv, T = Ccv[ok], T[ok]
    rcv = stats.rankdata(Ccv.ravel())
    rho_chk = np.corrcoef(rcv, stats.rankdata(T.ravel()))[0, 1]
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = np.array([row[rng.permutation(len(row))] for row in T]).ravel()
        null[b] = np.corrcoef(rcv, stats.rankdata(perm))[0, 1]
    assert abs(rho_chk - rho) < 1e-9, (rho_chk, rho)
    p_perm = (1 + np.sum(np.abs(null) >= abs(rho) - 1e-12)) / (n_perm + 1)
    d = res.finals[res.finals.method == "Default"].groupby(["dataset", "model"])[["cv_score", "test_f1_macro"]].mean()
    gain_cv = cv.mean(axis=1) - d.cv_score.reindex(cv.index)
    gain_te = te.mean(axis=1) - d.test_f1_macro.reindex(te.index)
    best_te = te.max(axis=1)
    return {"spearman_rho": float(rho), "p": float(p), "p_perm": float(p_perm), "n_perm": int(n_perm), "n_points": int(len(dcv)),
            "deviations": pd.DataFrame({"cv": dcv, "test": dte}),
            "gain": pd.DataFrame({"cv_gain": gain_cv, "test_gain": gain_te}),
            "default_ge_best_hpo": int((d.test_f1_macro.reindex(te.index) >= best_te).sum()),
            "blocks": int(len(te))}


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def fmt_p(p):
    if np.isnan(p):
        return "n/a"
    return "> 0.99" if p > 0.99 else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def block_matrix(res: Results, budget: int, metric: str = "test_f1_macro", methods=None) -> pd.DataFrame:
    """Seed-averaged score per block (rows) and method (columns)."""
    methods = methods or res.methods
    d = res.at_budget(budget)
    d = d[d.method.isin(methods)]
    piv = d.groupby(["dataset", "model", "method"])[metric].mean().unstack("method")
    return piv.reindex(columns=[m for m in methods if m in piv.columns]).dropna()


# ----------------------------------------------------------------------------- tables
def table_datasets(cfg: Config) -> pd.DataFrame:
    rows = []
    for name in cfg.datasets:
        meta = json.loads((cfg.cache_dir / f"{name}.json").read_text())
        rows.append({"Dataset": DATASET_LABEL[name], "Sensors": meta["description"].split(": ", 1)[1],
                     "Subjects (train/test)": f"{meta['n_subjects_train']}/{meta['n_subjects_test']}",
                     "Classes": len(meta["classes"]), "Windows (train/test)": f"{meta['n_train']}/{meta['n_test']}",
                     "Features": meta["n_features"], "Split": meta["split"]})
    return pd.DataFrame(rows)


def table_spaces(targets=(50, 100)) -> pd.DataFrame:
    rows = []
    for model, space in SPACES.items():
        lv = {t: grid_levels(len(space), t) for t in targets}
        for i, p in enumerate(space):
            rows.append({"Model": MODEL_LABEL[model], "Hyperparameter": p.name, "Type": p.kind,
                         "Lower": p.low, "Upper": p.high, "Scale": "log" if p.log else "linear",
                         **{f"Grid levels (~{t})": lv[t][i] for t in targets}})
        for t in targets:
            rows.append({"Model": MODEL_LABEL[model], "Hyperparameter": f"grid size (~{t})",
                         f"Grid levels (~{t})": len(build_grid(model, t))})
    return pd.DataFrame(rows)


def table_main(res: Results, budget: int, metric: str = "test_f1_macro") -> pd.DataFrame:
    d = res.at_budget(budget, include_default=True)
    g = d.groupby(["dataset", "model", "method"])[metric].agg(["mean", "std"])
    out = []
    for (ds, m), sub in g.groupby(level=[0, 1]):
        row = {"Dataset": DATASET_LABEL[ds], "Model": MODEL_LABEL[m]}
        best = sub.loc[(ds, m)].drop("Default", errors="ignore")["mean"].max()
        for meth in ["Default"] + res.methods:
            if (ds, m, meth) in sub.index:
                mu, sd = sub.loc[(ds, m, meth)]
                star = "*" if meth != "Default" and np.isclose(mu, best) else ""
                row[METHOD_LABEL[meth]] = f"{mu:.4f} ± {0 if np.isnan(sd) else sd:.4f}{star}"
        out.append(row)
    return pd.DataFrame(out)


def friedman_tests(res: Results, budget: int, metric: str = "test_f1_macro") -> dict:
    M = block_matrix(res, budget, metric)
    N, k = M.shape
    if N < 2 or k < 3:
        return {"N": N, "k": k}
    ranks = M.rank(axis=1, ascending=False)
    chi2, p = stats.friedmanchisquare(*[M[c].values for c in M.columns])
    ff = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) != chi2 else np.inf
    p_id = stats.f.sf(ff, k - 1, (k - 1) * (N - 1))
    q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    pairs = []
    for a, b in itertools.combinations(M.columns, 2):
        diff = M[a] - M[b]
        if np.allclose(diff, 0):
            w, pw = np.nan, 1.0
        else:
            w, pw = stats.wilcoxon(M[a], M[b], zero_method="pratt")
        nz = diff[diff != 0]
        r = stats.rankdata(np.abs(nz))
        rb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum() if len(nz) else 0.0     # matched-pairs rank-biserial
        pairs.append({"Comparison": f"{METHOD_LABEL[a]} vs {METHOD_LABEL[b]}", "Mean diff": diff.mean(),
                      "Wins/ties/losses": f"{(diff > 0).sum()}/{(diff == 0).sum()}/{(diff < 0).sum()}",
                      "W": w, "p": pw, "Rank-biserial r": rb})
    pairs = pd.DataFrame(pairs)
    pairs["p (Holm)"] = holm(pairs.p.values)
    return {"N": N, "k": k, "chi2": chi2, "p_friedman": p, "F_iman_davenport": ff, "p_iman_davenport": p_id,
            "kendall_w": chi2 / (N * (k - 1)), "cd": cd, "avg_rank": ranks.mean().sort_values(),
            "pairwise": pairs, "matrix": M}


def table_ranks(res: Results, metrics=("test_f1_macro", "cv_score")) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        for b in res.budgets():
            meths = [m for m in res.methods if m != "Grid" or b in res.cfg.grid_targets]
            M = block_matrix(res, b, metric, meths)
            if M.empty:
                continue
            r = M.rank(axis=1, ascending=False).mean()
            rows.append({"Metric": "test macro-F1" if metric.startswith("test") else "CV macro-F1 (objective)",
                         "Budget": b, "Blocks": len(M), **{METHOD_LABEL[m]: round(r[m], 2) for m in r.index}})
    return pd.DataFrame(rows)


def within_block_vs_random(res: Results, budget: int, metric: str = "test_f1_macro", alpha=0.05) -> pd.DataFrame:
    d = res.at_budget(budget)
    rows, tally = [], {m: [0, 0, 0] for m in res.methods if m != "Random"}
    for (ds, mod), sub in d.groupby(["dataset", "model"]):
        ref = sub[sub.method == "Random"][metric].values
        if len(ref) < 3:
            continue
        ps, comp = [], []
        for meth in tally:
            x = sub[sub.method == meth][metric].values
            if len(x) < 3:
                continue
            if np.ptp(np.concatenate([x, ref])) == 0:
                p = 1.0
            else:
                p = stats.mannwhitneyu(x, ref, alternative="two-sided").pvalue
            ps.append(p)
            comp.append((meth, x.mean() - ref.mean()))
        for (meth, delta), pa in zip(comp, holm(ps)):
            outcome = "win" if (pa < alpha and delta > 0) else ("loss" if (pa < alpha and delta < 0) else "tie")
            tally[meth][["win", "tie", "loss"].index(outcome)] += 1
            rows.append({"Dataset": DATASET_LABEL[ds], "Model": MODEL_LABEL[mod], "Method": METHOD_LABEL[meth],
                         "Mean diff vs Random": delta, "p (Holm, within block)": pa, "Outcome": outcome})
    summary = pd.DataFrame([{"Method": METHOD_LABEL[m], "Wins": w, "Ties": t, "Losses": l}
                            for m, (w, t, l) in tally.items()])
    return pd.DataFrame(rows), summary


def _subject_confusions(y, pred, g, K):
    subj = np.unique(g)
    C = np.zeros((len(subj), K, K))
    for i, s in enumerate(subj):
        m = g == s
        np.add.at(C[i], (y[m], pred[m]), 1)
    return C


def _macro_f1_from_conf(T):
    """T: (..., K, K) confusion counts -> macro-F1 over classes with support > 0."""
    tp = np.diagonal(T, axis1=-2, axis2=-1)
    support = T.sum(axis=-1)
    predicted = T.sum(axis=-2)
    den = support + predicted
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    present = support > 0
    return (f1 * present).sum(axis=-1) / np.maximum(present.sum(axis=-1), 1)


def cluster_bootstrap(res: Results, budget: int, reps: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Subject-level bootstrap of seed-averaged test macro-F1, and paired differences to Random."""
    reps = reps or res.cfg.bootstrap_reps
    rng = np.random.default_rng(seed)
    rows = []
    for ds in res.datasets:
        dsobj = build_dataset(ds, res.cfg, verbose=False)
        cidx = {c: i for i, c in enumerate(dsobj.classes)}
        y = np.array([cidx[c] for c in dsobj.y_test])
        g = dsobj.g_test
        K = len(dsobj.classes)
        n_subj = len(np.unique(g))
        W = rng.multinomial(n_subj, np.full(n_subj, 1.0 / n_subj), size=reps).astype(float)   # (reps, subjects)
        for mod in res.models:
            boot = {}
            for meth in res.methods:
                seeds = sorted(s for (d, m, me, b, s) in res.preds if d == ds and m == mod and me == meth and b == budget)
                if not seeds:
                    continue
                seeds = [s for s in seeds if len(res.preds[(ds, mod, meth, budget, s)]) == len(y)]
                if not seeds:
                    continue
                C = np.stack([_subject_confusions(y, res.preds[(ds, mod, meth, budget, s)], g, K) for s in seeds], 1)
                T = np.einsum("rs,sjkl->rjkl", W, C)                          # (reps, seeds, K, K)
                point = _macro_f1_from_conf(C.sum(0)).mean()
                boot[meth] = (_macro_f1_from_conf(T).mean(axis=1), point)
            if "Random" not in boot:
                continue
            for meth, (b, point) in boot.items():
                lo, hi = np.percentile(b, [2.5, 97.5])
                row = {"Dataset": DATASET_LABEL[ds], "Model": MODEL_LABEL[mod], "Method": METHOD_LABEL[meth],
                       "macro-F1": point, "CI low": lo, "CI high": hi}
                if meth != "Random":
                    dlt = b - boot["Random"][0]
                    dlo, dhi = np.percentile(dlt, [2.5, 97.5])
                    row.update({"Diff vs Random": point - boot["Random"][1], "Diff CI low": dlo, "Diff CI high": dhi,
                                "CI excludes 0": bool(dlo > 0 or dhi < 0)})
                rows.append(row)
    return pd.DataFrame(rows)


def table_cost(res: Results, budget: int) -> pd.DataFrame:
    d = res.at_budget(budget)
    col = "search_cpu_norm" if "search_cpu_norm" in d else "search_cpu_s"
    g = d.groupby(["dataset", "model", "method"])[col].mean().unstack("method") / 60.0
    g = g.reindex(columns=[m for m in res.methods if m in g.columns])
    g.index = [f"{DATASET_LABEL[a]} | {MODEL_LABEL[b]}" for a, b in g.index]
    g.columns = [METHOD_LABEL[c] for c in g.columns]
    rel = g.div(g[METHOD_LABEL["Random"]], axis=0) if METHOD_LABEL["Random"] in g.columns else g
    out = g.round(2).astype(str) + " (" + rel.round(2).astype(str) + "x)"
    return out.reset_index().rename(columns={"index": "Block (CPU-minutes, ratio to Random)"})


def anytime_curves(res: Results) -> pd.DataFrame:
    """Best full-fidelity CV score after b budget units, for b = 1..budget."""
    r = res.records[(res.records.fidelity >= 1.0) & (res.records.method != "Default")].copy()
    rows = []
    B = np.arange(1, res.cfg.budget + 1)
    for key, sub in r.groupby(["dataset", "model", "method", "tag", "seed"]):
        cc = sub.cum_cost.values
        best = np.maximum.accumulate(sub.score.values)
        idx = np.searchsorted(cc, B + 1e-9, side="right") - 1
        val = np.where(idx >= 0, best[np.clip(idx, 0, None)], np.nan)
        if key[2] == "Grid":                       # grid is only meaningful once complete
            val = np.where(B >= cc[-1] - 1e-9, best[-1], np.nan)
        rows.append(pd.DataFrame({"dataset": key[0], "model": key[1], "method": key[2], "tag": key[3],
                                  "seed": key[4], "budget": B, "best_cv": val}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def table_boundaries(res: Results, budget: int, tol: float = 0.02) -> pd.DataFrame:
    d = res.at_budget(budget)
    rows = []
    for (mod, meth), sub in d.groupby(["model", "method"]):
        dist = pd.DataFrame([boundary_distance(mod, json.loads(p)) for p in sub.params])
        for p in dist.columns:
            rows.append({"Model": MODEL_LABEL[mod], "Hyperparameter": p, "Method": METHOD_LABEL[meth],
                         "% within 2% of a bound": 100 * (dist[p] < tol).mean()})
    out = pd.DataFrame(rows)
    return out.pivot_table(index=["Model", "Hyperparameter"], columns="Method",
                           values="% within 2% of a bound").round(1).reset_index() if len(out) else out


def hyperparameter_importance(res: Results, method: str = "Random") -> pd.DataFrame:
    """fANOVA importance (Hutter et al., 2014) on the pooled Random Search trials of each block.
    Random Search samples are i.i.d. over the space, which is what fANOVA assumes."""
    import optuna
    from optuna.importance import FanovaImportanceEvaluator
    from .spaces import distributions
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    r = res.records[(res.records.method == method) & (res.records.fidelity >= 1.0)]
    rows = []
    for (ds, mod), sub in r.groupby(["dataset", "model"]):
        dists = distributions(mod)
        st = optuna.create_study(direction="maximize")
        for _, row in sub.iterrows():
            st.add_trial(optuna.trial.create_trial(params=json.loads(row.params), distributions=dists, value=row.score))
        imp = optuna.importance.get_param_importances(st, evaluator=FanovaImportanceEvaluator(seed=0))
        for k, v in imp.items():
            rows.append({"Dataset": DATASET_LABEL[ds], "Model": MODEL_LABEL[mod], "Hyperparameter": k,
                         "Importance": v, "n_trials": len(sub)})
    return pd.DataFrame(rows)


def table_secondary_metrics(res: Results, budget: int) -> pd.DataFrame:
    d = res.at_budget(budget, include_default=True)
    cols = ["test_accuracy", "test_balanced_accuracy", "test_cohen_kappa", "test_mcc", "test_roc_auc_ovr", "test_log_loss"]
    g = d.groupby(["model", "method"])[cols].mean().reset_index()
    g["model"] = g.model.map(MODEL_LABEL)
    g["method"] = g.method.map(METHOD_LABEL)
    return g.rename(columns={c: c.replace("test_", "") for c in cols}).round(4)


# ----------------------------------------------------------------------------- export
def save_table(df: pd.DataFrame, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    try:
        df.to_latex(out / f"{name}.tex", index=False, float_format="%.4f")
    except Exception:  # noqa: BLE001  (jinja2 missing)
        pass


def run_analysis(cfg: Config, extra_dirs=None, out_dir: Path | None = None, figures: bool = True,
                 validate: bool = True) -> dict:
    res = Results(cfg, extra_dirs, validate=validate)
    out = Path(out_dir or cfg.results_dir / "analysis")
    tdir = out / "tables"
    summary = {"config": json.loads(cfg.to_json())}
    save_table(table_datasets(cfg), tdir, "T01_datasets")
    save_table(table_spaces(tuple(cfg.grid_targets)), tdir, "T02_search_spaces")
    main_b = cfg.budget
    for b in res.budgets():
        save_table(table_main(res, b), tdir, f"T03_test_macroF1_budget{b}")
        save_table(table_main(res, b, "cv_score"), tdir, f"T03b_cv_macroF1_budget{b}")
    save_table(table_ranks(res), tdir, "T04_average_ranks_by_budget")
    for metric in ["test_f1_macro", "cv_score"]:
        fr = friedman_tests(res, main_b, metric)
        if "chi2" in fr:
            tag = "test" if metric.startswith("test") else "cv"
            save_table(fr["pairwise"].round(5), tdir, f"T05_pairwise_wilcoxon_{tag}_budget{main_b}")
            summary[f"friedman_{tag}"] = {k: (float(v) if np.isscalar(v) else v.round(3).to_dict())
                                          for k, v in fr.items() if k not in ("pairwise", "matrix")}
    detail, tally = within_block_vs_random(res, main_b)
    save_table(detail.round(5), tdir, f"T06a_within_block_vs_random_budget{main_b}")
    save_table(tally, tdir, f"T06b_win_tie_loss_vs_random_budget{main_b}")
    try:
        boot = cluster_bootstrap(res, main_b)
    except Exception as e:  # noqa: BLE001  (needs the local dataset cache)
        boot = pd.DataFrame()
        print("subject bootstrap skipped:", e)
    save_table(boot.round(4), tdir, f"T07_subject_bootstrap_budget{main_b}")
    save_table(table_cost(res, main_b), tdir, f"T08_search_cost_budget{main_b}")
    save_table(table_boundaries(res, main_b), tdir, f"T09_boundary_hits_budget{main_b}")
    save_table(table_secondary_metrics(res, main_b), tdir, f"T10_secondary_metrics_budget{main_b}")
    if getattr(hardware_factors, "table", None) is not None and len(hardware_factors.table):
        save_table(hardware_factors.table.round(3), tdir, "T12_cpu_time_hardware_factors")
    agr = cv_test_agreement(res, main_b)
    save_table(agr["gain"].reset_index().round(5), tdir, f"T13_gain_over_default_cv_vs_test_budget{main_b}")
    summary["cv_test_agreement"] = {k: agr[k] for k in ["spearman_rho", "p", "p_perm", "n_perm", "n_points", "default_ge_best_hpo", "blocks"]}
    try:
        imp = hyperparameter_importance(res)
        save_table(imp.round(4), tdir, "T11_fanova_importance")
    except Exception as e:  # noqa: BLE001
        imp = pd.DataFrame()
        print("fANOVA skipped:", e)
    curves = anytime_curves(res)
    if figures:
        from . import figures as F
        F.make_all(res, curves, boot, imp, out / "figures", main_b)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"analysis written to {out}")
    return {"results": res, "curves": curves, "bootstrap": boot, "importance": imp, "summary": summary}
