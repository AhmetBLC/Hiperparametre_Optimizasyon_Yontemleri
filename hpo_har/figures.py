"""Publication figures (English labels, 300 dpi PNG + vector PDF)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import (DATASET_LABEL, METHOD_LABEL, MODEL_LABEL, block_matrix,  # noqa: E402
                       friedman_tests)
from .spaces import SPACES  # noqa: E402

# Fixed categorical order (validated reference palette); markers give a second encoding.
COLOR = {"Grid": "#2a78d6", "Random": "#eb6834", "TPE": "#1baf7a", "CMAES": "#eda100",
         "GA": "#e87ba4", "BOHB": "#008300", "Default": "#52514e"}
MARKER = {"Grid": "s", "Random": "o", "TPE": "^", "CMAES": "D", "GA": "v", "BOHB": "P", "Default": "x"}
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRIDC, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
                     "legend.frameon": False})


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def _plain_log_x(ax, ticks):
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())


def _legend(fig, methods, ncol=None, y=1.02):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLOR[m], marker=MARKER[m], lw=2, ms=5, label=METHOD_LABEL[m]) for m in methods]
    fig.legend(handles=h, loc="upper center", ncol=ncol or len(methods), bbox_to_anchor=(0.5, y), fontsize=8)


def fig_anytime(res, curves: pd.DataFrame, out: Path):
    """Best CV macro-F1 vs budget (mean and 95% CI over seeds), one panel per block."""
    if curves.empty:
        return
    D, M = res.datasets, res.models
    fig, axes = plt.subplots(len(D), len(M), figsize=(3.0 * len(M), 2.0 * len(D)), squeeze=False)
    for i, ds in enumerate(D):
        for j, mod in enumerate(M):
            ax = axes[i, j]
            sub = curves[(curves.dataset == ds) & (curves.model == mod)]
            lo_all = []
            for meth in res.methods:
                s = sub[sub.method == meth]
                if s.empty:
                    continue
                if meth == "Grid":
                    pts = s.dropna().groupby("tag").agg(b=("budget", "min"))
                    for tag, b in pts.b.items():
                        v = s[(s.tag == tag) & (s.budget == b)].best_cv
                        ax.errorbar(b, v.mean(), yerr=1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0,
                                    color=COLOR[meth], marker=MARKER[meth], ms=4, lw=1, zorder=4)
                    continue
                g = s.groupby("budget").best_cv
                mu, se, n = g.mean(), g.std(ddof=1), g.count()
                ci = 1.96 * se / np.sqrt(n)
                ax.plot(mu.index, mu.values, color=COLOR[meth], lw=1.5)
                ax.fill_between(mu.index, (mu - ci).values, (mu + ci).values, color=COLOR[meth], alpha=0.15, lw=0)
                if len(mu) > 10:
                    lo_all.append(mu.iloc[min(9, len(mu) - 1)])
            if lo_all:
                top = sub.best_cv.max()
                ax.set_ylim(min(lo_all) - 0.2 * (top - min(lo_all) + 1e-3), top + 0.1 * (top - min(lo_all) + 1e-3))
            _plain_log_x(ax, [b for b in [1, 3, 10, 25, 50, 100, 200] if b <= res.cfg.budget])
            if i == 0:
                ax.set_title(MODEL_LABEL[mod], fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{DATASET_LABEL[ds]}\nbest CV macro-F1")
            if i == len(D) - 1:
                ax.set_xlabel("budget (full evaluations)")
    _legend(fig, res.methods, y=1.01)
    fig.tight_layout()
    _save(fig, out, "fig01_anytime_cv")


def fig_rank_vs_budget(res, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0), sharey=True)
    for ax, metric, title in [(axes[0], "cv_score", "CV macro-F1 (search objective)"),
                              (axes[1], "test_f1_macro", "Test macro-F1 (held-out subjects)")]:
        rows = []
        for b in res.budgets():
            meths = [m for m in res.methods if m != "Grid" or b in res.cfg.grid_targets]
            M = block_matrix(res, b, metric, meths)
            if M.empty:
                continue
            r = M.rank(axis=1, ascending=False)
            for m in r.columns:
                rows.append({"budget": b, "method": m, "rank": r[m].mean(), "se": r[m].std(ddof=1) / np.sqrt(len(r))})
        d = pd.DataFrame(rows)
        for m in res.methods:
            s = d[d.method == m]
            if s.empty:
                continue
            ax.errorbar(s.budget, s["rank"], yerr=1.96 * s.se.fillna(0), color=COLOR[m], marker=MARKER[m],
                        ms=5, lw=1.5, capsize=2)
        _plain_log_x(ax, res.budgets())
        ax.set_xlabel("budget (full evaluations)")
        ax.set_title(title, fontsize=9, color=INK)
        ax.invert_yaxis()
    axes[0].set_ylabel("average rank across blocks (1 = best)")
    _legend(fig, res.methods, y=1.08)
    fig.tight_layout()
    _save(fig, out, "fig02_rank_vs_budget")


def fig_test_distributions(res, out: Path, budget: int):
    d = res.at_budget(budget, include_default=True)
    D, M = res.datasets, res.models
    meths = ["Default"] + res.methods
    fig, axes = plt.subplots(len(D), len(M), figsize=(3.0 * len(M), 1.9 * len(D)), squeeze=False)
    rng = np.random.default_rng(0)
    for i, ds in enumerate(D):
        for j, mod in enumerate(M):
            ax = axes[i, j]
            sub = d[(d.dataset == ds) & (d.model == mod)]
            for k, m in enumerate(meths):
                v = sub[sub.method == m].test_f1_macro.values
                if len(v) == 0:
                    continue
                ax.scatter(k + rng.uniform(-0.15, 0.15, len(v)), v, s=9, color=COLOR[m], alpha=0.8, lw=0)
                ax.hlines(v.mean(), k - 0.3, k + 0.3, color=INK, lw=1.2)
            ax.set_xticks(range(len(meths)))
            ax.set_xticklabels([] if i < len(D) - 1 else [METHOD_LABEL[m].replace(" (TPE)", "") for m in meths],
                               rotation=45, ha="right", fontsize=7)
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(MODEL_LABEL[mod], fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{DATASET_LABEL[ds]}\ntest macro-F1")
    fig.tight_layout()
    _save(fig, out, f"fig03_test_f1_by_seed_budget{budget}")


def fig_cd(res, out: Path, budget: int, metric="test_f1_macro"):
    fr = friedman_tests(res, budget, metric)
    if "avg_rank" not in fr:
        return
    ranks, cd, k = fr["avg_rank"], fr["cd"], fr["k"]
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * k + 0.9))
    ax.grid(False)
    for s in ["left", "right", "top"]:
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    ax.set_xlim(0.8, k + 0.2)                     # rank 1 (best) on the left
    ax.set_ylim(-0.6, k / 2 + 0.6)
    ax.set_xlabel("average rank (1 = best)")
    srt = ranks.sort_values()
    half = (k + 1) // 2
    for n, (m, r) in enumerate(srt.items()):
        left = n < half
        y = 0.4 + 0.45 * (n if left else (k - 1 - n))
        xe = 0.8 if left else k + 0.2
        ax.plot([r, r], [0, y], color=COLOR[m], lw=1.2)
        ax.plot([r, xe], [y, y], color=COLOR[m], lw=1.2)
        ax.plot(r, 0, MARKER[m], color=COLOR[m], ms=5)
        ax.text(xe - 0.05 if left else xe + 0.05, y, f"{METHOD_LABEL[m]} ({r:.2f})", va="center",
                ha="right" if left else "left", fontsize=8, clip_on=False)
    # cliques: groups whose rank range is below CD
    vals, names = srt.values, list(srt.index)
    cl, lvl = [], 0
    for a in range(k):
        b = max(i for i in range(k) if vals[i] - vals[a] <= cd)
        if b > a and not any(a >= c0 and b <= c1 for c0, c1 in cl):
            cl.append((a, b))
            ax.plot([vals[a] - 0.03, vals[b] + 0.03], [-0.15 - 0.12 * lvl] * 2, color=INK, lw=2.5)
            lvl += 1
    ax.plot([1, 1 + cd], [k / 2 + 0.4] * 2, color=INK, lw=1.5)
    ax.text(1 + cd / 2, k / 2 + 0.47, f"CD = {cd:.2f}", ha="center", fontsize=8)
    ax.set_title(f"Friedman p = {fr['p_friedman']:.3g}, Iman-Davenport p = {fr['p_iman_davenport']:.3g}, "
                 f"N = {fr['N']} blocks", fontsize=8, color=INK2)
    _save(fig, out, f"fig04_cd_diagram_{'test' if metric.startswith('test') else 'cv'}_budget{budget}")


def fig_cost(res, out: Path, budget: int):
    d = res.at_budget(budget)
    M = block_matrix(res, budget, "test_f1_macro")
    if M.empty:
        return
    ranks = M.rank(axis=1, ascending=False)
    col = "search_cpu_norm" if "search_cpu_norm" in d else "search_cpu_s"
    cost = d.groupby(["dataset", "model", "method"])[col].mean().unstack("method").reindex(M.index)
    rel = cost.div(cost["Random"], axis=0) if "Random" in cost else cost
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for m in M.columns:
        x, y = rel[m].values, ranks[m].values
        gm = np.exp(np.mean(np.log(x)))
        ax.errorbar(gm, y.mean(), yerr=1.96 * y.std(ddof=1) / np.sqrt(len(y)),
                    xerr=[[gm - np.exp(np.percentile(np.log(x), 25))], [np.exp(np.percentile(np.log(x), 75)) - gm]],
                    color=COLOR[m], marker=MARKER[m], ms=7, capsize=2, lw=1.2, label=METHOD_LABEL[m])
    from matplotlib.ticker import FuncFormatter, NullFormatter
    from matplotlib.ticker import FixedLocator
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.invert_yaxis()
    ax.set_xlabel("search CPU time relative to Random Search (geometric mean, IQR)")
    ax.set_ylabel("average rank, test macro-F1")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    _save(fig, out, f"fig05_cost_vs_rank_budget{budget}")


def fig_selected_locations(res, out: Path, budget: int):
    """Where in the space the selected configurations lie (0 = lower bound, 1 = upper)."""
    d = res.at_budget(budget)
    fig, axes = plt.subplots(1, len(res.models), figsize=(3.4 * len(res.models), 2.6), squeeze=False)
    rng = np.random.default_rng(1)
    for j, mod in enumerate(res.models):
        ax = axes[0, j]
        names = [p.name for p in SPACES[mod]]
        sub = d[d.model == mod]
        for k, meth in enumerate(res.methods):
            s = sub[sub.method == meth]
            for i, p in enumerate(SPACES[mod]):
                u = np.array([p.encode(json.loads(x)[p.name]) for x in s.params])
                ax.scatter(u, i + (k - len(res.methods) / 2) * 0.09 + rng.uniform(-0.02, 0.02, len(u)), s=6,
                           color=COLOR[meth], alpha=0.7, lw=0, zorder=3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xlim(-0.03, 1.03)
        ax.axvspan(-0.03, 0.02, color=GRIDC, lw=0, alpha=0.6, zorder=0)   # behind the points
        ax.axvspan(0.98, 1.03, color=GRIDC, lw=0, alpha=0.6, zorder=0)
        ax.set_xlabel("normalised position in the search space")
        ax.set_title(MODEL_LABEL[mod], fontsize=9, color=INK)
        ax.grid(axis="y", visible=False)
    _legend(fig, res.methods, y=1.1)
    fig.tight_layout()
    _save(fig, out, f"fig06_selected_configurations_budget{budget}")


def fig_importance(imp: pd.DataFrame, out: Path):
    if imp is None or imp.empty:
        return
    models = [m for m in MODEL_LABEL.values() if m in set(imp.Model)]
    fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models), 2.6), squeeze=False)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#2a78d6", "#0d3a73"])
    for j, mod in enumerate(models):
        ax = axes[0, j]
        piv = imp[imp.Model == mod].pivot_table(index="Hyperparameter", columns="Dataset", values="Importance")
        piv = piv.reindex(columns=[c for c in DATASET_LABEL.values() if c in piv.columns])
        ax.imshow(piv.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        for (a, b), v in np.ndenumerate(piv.values):
            ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.55 else INK)
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(piv.columns, rotation=40, ha="right", fontsize=7.5)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels(piv.index, fontsize=7.5)
        ax.set_title(mod, fontsize=9, color=INK)
    fig.tight_layout()
    _save(fig, out, "fig07_fanova_importance")


def fig_bootstrap(boot: pd.DataFrame, out: Path, budget: int):
    if boot is None or boot.empty or "Diff vs Random" not in boot:
        return
    b = boot.dropna(subset=["Diff vs Random"])
    blocks = list(dict.fromkeys(zip(b.Dataset, b.Model)))
    meths = [m for m in METHOD_LABEL if METHOD_LABEL[m] in set(b.Method)]
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(blocks) + 1.0))
    for i, (ds, mod) in enumerate(blocks):
        for k, m in enumerate(meths):
            r = b[(b.Dataset == ds) & (b.Model == mod) & (b.Method == METHOD_LABEL[m])]
            if r.empty:
                continue
            y = i + (k - len(meths) / 2) * 0.13
            r = r.iloc[0]
            ax.plot([r["Diff CI low"] * 100, r["Diff CI high"] * 100], [y, y], color=COLOR[m], lw=1.3)
            ax.plot(r["Diff vs Random"] * 100, y, MARKER[m], color=COLOR[m], ms=3.5)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels([f"{a} | {c}" for a, c in blocks], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("difference in test macro-F1 vs Random Search (percentage points, 95% subject-bootstrap CI)")
    ax.grid(axis="y", visible=False)
    _legend(fig, meths, y=1.0 + 0.6 / (0.32 * len(blocks) + 1.0))
    fig.tight_layout()
    _save(fig, out, f"fig08_bootstrap_vs_random_budget{budget}")


def fig_ga_dynamics(res, out: Path):
    g = res.generations
    if g is None or g.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.8))
    for mod, sub in g.groupby("model"):
        a = sub.groupby("generation")[["best", "mean", "diversity"]].mean()
        axes[0].plot(a.index, a["best"] - a["best"].iloc[-1], marker="o", ms=3, lw=1.3, label=MODEL_LABEL[mod])
        axes[1].plot(a.index, a["diversity"], marker="o", ms=3, lw=1.3, label=MODEL_LABEL[mod])
    axes[0].set_ylabel("best fitness of generation\n(relative to final generation)")
    axes[1].set_ylabel("population diversity\n(mean gene std, unit cube)")
    for ax in axes:
        ax.set_xlabel("generation")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out, "fig09_ga_dynamics")


def fig_cv_vs_test(res, out: Path, budget: int):
    """(a) method advantage on the CV objective vs on the test subjects; (b) gain over defaults."""
    from .analysis import cv_test_agreement
    a = cv_test_agreement(res, budget)
    dev, gain = a["deviations"], a["gain"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for m in dev.index.get_level_values(-1).unique():
        s = dev.xs(m, level=-1)
        ax.scatter(s.cv * 100, s.test * 100, s=22, color=COLOR[m], marker=MARKER[m], lw=0, alpha=0.85,
                   label=METHOD_LABEL[m], zorder=3)
    lim = max(dev.abs().max()) * 100 * 1.08
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel("CV macro-F1 relative to block mean (pp)")
    ax.set_ylabel("test macro-F1 relative to block mean (pp)")
    ax.set_title(f"(a) Spearman \u03c1 = {a['spearman_rho']:.2f} (permutation p = {a['p_perm']:.2f}, n = {a['n_points']})",
                 fontsize=9, color=INK)
    ax.set_ylim(-lim, lim)
    ax = axes[1]
    g = gain.groupby(level=1).mean() * 100
    g = g.reindex([m for m in ["SVM", "RandomForest", "XGBoost"] if m in g.index])
    x = np.arange(len(g))
    ax.bar(x - 0.19, g.cv_gain, 0.36, color="#2a78d6", label="CV (search objective)")
    ax.bar(x + 0.19, g.test_gain, 0.36, color="#eb6834", label="test (held-out subjects)")
    for xi, (c, t) in enumerate(zip(g.cv_gain, g.test_gain)):
        ax.text(xi - 0.19, c + (0.08 if c >= 0 else -0.08), f"{c:+.1f}", ha="center",
                va="bottom" if c >= 0 else "top", fontsize=7.5, color=INK2)
        ax.text(xi + 0.19, t + (0.08 if t >= 0 else -0.08), f"{t:+.1f}", ha="center",
                va="bottom" if t >= 0 else "top", fontsize=7.5, color=INK2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in g.index])
    ax.set_ylabel("tuned minus default macro-F1 (pp)")
    ax.set_title("(b) gain over library defaults, mean of all HPO methods", fontsize=9, color=INK)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7.5, loc="upper left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.07), fontsize=7.5)
    fig.tight_layout()
    _save(fig, out, f"fig10_cv_vs_test_budget{budget}")


def make_all(res, curves, boot, imp, out: Path, budget: int):
    for f in [lambda: fig_anytime(res, curves, out), lambda: fig_rank_vs_budget(res, out),
              lambda: fig_test_distributions(res, out, budget), lambda: fig_cd(res, out, budget),
              lambda: fig_cd(res, out, budget, "cv_score"), lambda: fig_cost(res, out, budget),
              lambda: fig_selected_locations(res, out, budget), lambda: fig_importance(imp, out),
              lambda: fig_bootstrap(boot, out, budget), lambda: fig_ga_dynamics(res, out),
              lambda: fig_cv_vs_test(res, out, budget)]:
        try:
            f()
        except Exception as e:  # noqa: BLE001
            import traceback
            print("figure failed:", e)
            traceback.print_exc()
