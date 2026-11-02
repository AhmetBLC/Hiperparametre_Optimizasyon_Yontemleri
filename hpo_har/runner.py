"""Job enumeration, sharding, parallel execution with resume, and runtime calibration.

A job is one (dataset, model, method, seed) search. Each finished job is written as one
gzipped JSON file, so a crashed or disconnected session loses at most the jobs that
were running, and several machines can share the work (shard_index / num_shards) and
write into the same results folder (e.g. a shared Google Drive folder).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import os
import platform
import time
import threading
import traceback
import uuid
from pathlib import Path

import numpy as np

from .config import Config
from .datasets import build_dataset, subsample_train
from .evaluation import EncodedData, cv_evaluate, set_checkpoint, test_evaluate
from .optimizers import METHODS, best_at_budget
from .spaces import SPACES, build_grid, decode_vector

DETERMINISTIC_MODELS = {"SVM"}      # SVC has no randomness: one seed is enough for Grid/Default


def environment_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "processor": platform.processor()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["processor"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    for lib in ["numpy", "pandas", "scipy", "sklearn", "xgboost", "optuna", "cmaes"]:
        try:
            info[lib] = __import__(lib).__version__
        except Exception:  # noqa: BLE001
            info[lib] = None
    return info


# ----------------------------------------------------------------------------- jobs
def enumerate_jobs(cfg: Config) -> list[dict]:
    jobs = []
    for ds in cfg.datasets:
        for model in cfg.models:
            seeds_det = cfg.seeds[:1] if model in DETERMINISTIC_MODELS else cfg.seeds
            for s in seeds_det:
                jobs.append({"dataset": ds, "model": model, "method": "Default", "tag": "Default", "seed": s})
            for method in cfg.methods:
                if method == "Grid":
                    for t in cfg.grid_targets:
                        for s in seeds_det:
                            jobs.append({"dataset": ds, "model": model, "method": "Grid",
                                         "tag": f"Grid{t}", "target": t, "seed": s})
                else:
                    for s in cfg.seeds:
                        jobs.append({"dataset": ds, "model": model, "method": method, "tag": method, "seed": s})
    for j in jobs:
        j["id"] = f"{j['dataset']}__{j['model']}__{j['tag']}__s{j['seed']}"
    return jobs


def _relative_cost(job: dict, cfg: Config, sizes: dict) -> float:
    n, d = sizes.get(job["dataset"], (5000, 200))
    w = {"SVM": (n / 5000) ** 2 * 1.0, "RandomForest": n / 5000 * d / 200 * 3.0,
         "XGBoost": n / 5000 * d / 200 * 4.0}[job["model"]]
    evals = {"Default": 1, "Grid": job.get("target", 50)}.get(job["method"], cfg.budget)
    return w * evals


def shard_jobs(jobs, cfg: Config, sizes: dict) -> list[dict]:
    """Largest jobs first (or smallest first), dealt round-robin over shards."""
    order = sorted(jobs, key=lambda j: (-_relative_cost(j, cfg, sizes), j["id"]))
    if getattr(cfg, "job_order", "largest_first") == "smallest_first":
        order = order[::-1]
    mine = [j for k, j in enumerate(order) if k % cfg.num_shards == cfg.shard_index]
    return mine


def job_path(cfg: Config, job_id: str, fp: str) -> Path:
    """Result file of a job; fp is the fingerprint of the dataset version it was computed on."""
    return cfg.results_dir / "jobs" / f"{job_id}__d{fp}.json.gz"


_FP_RE = re.compile(r"^(?P<id>.+?)(?:__d(?P<fp>[0-9a-f]{8}))?\.json\.gz$")


def _write_json_gz(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def read_json_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- one job
_DATA_CACHE: dict = {}


def _data(cfg: Config, name: str) -> EncodedData:
    key = (name, cfg.max_train_rows, cfg.cv_folds)
    if key not in _DATA_CACHE:
        ds = build_dataset(name, cfg, verbose=False)
        if cfg.max_train_rows:
            ds = subsample_train(ds, cfg.max_train_rows)
        _DATA_CACHE[key] = EncodedData(ds, cfg.cv_folds)
    return _DATA_CACHE[key]


def dataset_fingerprint(cfg: Config, name: str) -> str:
    """Short hash of the exact training/test split a job sees (changes if the data change)."""
    d = _data(cfg, name)
    h = hashlib.sha1()
    for a in (d.y, d.g, d.y_test, d.g_test):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(json.dumps([d.classes, list(d.X.shape), list(d.X_test.shape), cfg.cv_folds]).encode())
    return h.hexdigest()[:8]


def dataset_fingerprints(cfg: Config) -> dict:
    return {n: dataset_fingerprint(cfg, n) for n in cfg.datasets}


def reconcile_results(cfg: Config, fps: dict, verbose: bool = True) -> dict:
    """Keep only result files computed on the current version of each dataset.

    Files from v2.0.0 (no fingerprint in the name) are accepted when their train/test sizes and
    classes match the current data and renamed; anything computed on other data versions
    (e.g. before the PAMAP2 split change) is moved to results/<preset>/stale/ - never deleted.
    """
    jdir = cfg.results_dir / "jobs"
    counts = {"kept": 0, "migrated": 0, "stale": 0}
    if not jdir.exists():
        return counts
    stale_dir = cfg.results_dir / "stale"
    for f in sorted(jdir.glob("*.json.gz")):
        m = _FP_RE.match(f.name)
        if not m:
            continue
        job_id, fp = m.group("id"), m.group("fp")
        ds = job_id.split("__")[0]
        if ds not in fps:
            continue
        if fp == fps[ds]:
            counts["kept"] += 1
            continue
        ok = False
        if fp is None:
            try:
                j = read_json_gz(f)
                d = _data(cfg, ds)
                ok = (j.get("n_test") == len(d.y_test) and j.get("n_train") == len(d.y)
                      and j.get("classes") == d.classes)
            except Exception:  # noqa: BLE001
                ok = False
        if ok:
            os.replace(f, job_path(cfg, job_id, fps[ds]))
            counts["migrated"] += 1
        else:
            stale_dir.mkdir(parents=True, exist_ok=True)
            os.replace(f, stale_dir / f.name)
            counts["stale"] += 1
    if verbose and (counts["migrated"] or counts["stale"]):
        print(f"results check: {counts['kept']} current, {counts['migrated']} older files accepted, "
              f"{counts['stale']} computed on an older data version moved to {stale_dir} (will be re-run)")
    return counts


CLAIM_STALE_S = 45 * 60          # a claim not refreshed for 45 min belongs to a dead session
HEARTBEAT_S = 5 * 60


def _side_path(cfg: Config, kind: str, job: dict) -> Path:
    ext = {"partial": "jsonl", "claims": "json"}[kind]
    return cfg.results_dir / kind / f"{job['id']}__d{job['fp']}.{ext}"


def _read_claim(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def claimed_elsewhere(cfg: Config, job: dict) -> bool:
    c = _read_claim(_side_path(cfg, "claims", job))
    return bool(c) and c.get("session") != job.get("session") and time.time() - c.get("t", 0) < CLAIM_STALE_S


def _write_claim(path: Path, session: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps({"session": session, "host": platform.node(), "t": time.time()}))
    os.replace(tmp, path)


def run_job(job: dict, cfg: Config) -> dict:
    from threadpoolctl import threadpool_limits
    out_path = job_path(cfg, job["id"], job["fp"])
    if out_path.exists():
        return {"id": job["id"], "status": "exists"}
    if claimed_elsewhere(cfg, job):              # another live session is running this job
        return {"id": job["id"], "status": "claimed"}
    claim, partial = _side_path(cfg, "claims", job), _side_path(cfg, "partial", job)
    session = job.get("session", "local")
    _write_claim(claim, session)
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(HEARTBEAT_S):
            try:
                _write_claim(claim, session)
            except OSError:
                pass

    threading.Thread(target=heartbeat, daemon=True).start()
    partial.parent.mkdir(parents=True, exist_ok=True)
    n_resumed = set_checkpoint(partial)
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        with threadpool_limits(1):
            data = _data(cfg, job["dataset"])
            model, method, seed = job["model"], job["method"], job["seed"]
            extra = {}
            if method == "Default":
                res = cv_evaluate(data, model, None, seed)
                records = [{"i": 1, "params": None, "score": res["score"], "fold_scores": res["fold_scores"],
                            "cost": 1.0, "cum_cost": 1.0, "fidelity": 1.0,
                            "cpu_s": res["cpu_s"], "wall_s": res["wall_s"]}]
                budgets = [1]
            else:
                fn = METHODS[method]
                kw = dict(data=data, model=model, seed=seed, budget=cfg.budget, method=method,
                          target=job.get("target"), eta=cfg.bohb_eta, pop_size=cfg.ga_pop_size)
                records, extra = fn(**kw)
                budgets = [len(records)] if method == "Grid" else [b for b in cfg.checkpoints if b <= cfg.budget]
            finals, cache = [], {}
            for b in budgets:
                best = records[0] if method == "Default" else best_at_budget(records, b)
                if best is None:
                    continue
                key = json.dumps(best["params"], sort_keys=True)
                if key not in cache:
                    cache[key] = test_evaluate(data, model, best["params"], seed)
                te = cache[key]
                used = [r for r in records if r["cum_cost"] <= b + 1e-9]
                finals.append({"budget": b, "cv_score": best["score"], "best_index": best["i"],
                               "params": best["params"], "test": te["metrics"],
                               "y_pred": te["y_pred"].tolist(),
                               "search_cpu_s": float(sum(r["cpu_s"] for r in used)),
                               "search_wall_s": float(sum(r["wall_s"] for r in used))})
        result = dict(job, status="ok", records=records, finals=finals, extra=extra,
                      n_train=int(len(data.y)), n_test=int(len(data.y_test)), classes=data.classes,
                      job_cpu_s=time.process_time() - c0, job_wall_s=time.perf_counter() - t0,
                      env=environment_info(), config={k: getattr(cfg, k) for k in
                      ["preset", "budget", "checkpoints", "cv_folds", "ga_pop_size", "bohb_eta",
                       "window_sec", "window_overlap", "max_windows_per_subject_class", "max_train_rows"]})
        result["resumed_evaluations"] = n_resumed
        _write_json_gz(out_path, result)
        stale = out_path.with_name(job["id"] + ".error.txt")
        for p in (stale, partial):
            if p.exists():
                p.unlink()
        return {"id": job["id"], "status": "ok", "minutes": round((time.perf_counter() - t0) / 60, 2),
                "resumed": n_resumed}
    except Exception as e:  # noqa: BLE001
        err = out_path.with_name(job["id"] + ".error.txt")
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(traceback.format_exc())
        return {"id": job["id"], "status": "error", "error": repr(e)}
    finally:
        stop.set()
        set_checkpoint(None)
        try:
            c = _read_claim(claim)
            if c and c.get("session") == session:
                claim.unlink()
        except OSError:
            pass


# ----------------------------------------------------------------------------- orchestration
def prepare_datasets(cfg: Config) -> dict:
    """Download, window and featurise every dataset once (cached). Returns sizes."""
    sizes = {}
    for name in cfg.datasets:
        print(f"[data] {name}")
        ds = build_dataset(name, cfg)
        sizes[name] = (len(ds.y_train), ds.X_train.shape[1])
    return sizes


def run_all(cfg: Config, n_workers: int | None = None, verbose: int = 5) -> list[dict]:
    from joblib import Parallel, delayed
    for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[v] = "1"
    sizes = {n: (json.loads((cfg.cache_dir / f"{n}.json").read_text())["n_train"],
                 json.loads((cfg.cache_dir / f"{n}.json").read_text())["n_features"])
             for n in cfg.datasets if (cfg.cache_dir / f"{n}.json").exists()}
    fps = dataset_fingerprints(cfg)
    reconcile_results(cfg, fps)
    session = uuid.uuid4().hex[:8]
    jobs = shard_jobs(enumerate_jobs(cfg), cfg, sizes)
    for j in jobs:
        j["fp"], j["session"] = fps[j["dataset"]], session
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "config.json").write_text(cfg.to_json())
    n = n_workers or cfg.n_workers
    all_out, rnd = [], 0
    while True:
        todo = [j for j in jobs if not job_path(cfg, j["id"], j["fp"]).exists()]
        if not todo:
            print("all jobs of this shard are finished")
            break
        free = [j for j in todo if not claimed_elsewhere(cfg, j)]
        rnd += 1
        print(f"[round {rnd}] shard {cfg.shard_index + 1}/{cfg.num_shards}: {len(jobs)} jobs, "
              f"{len(jobs) - len(todo)} done, {len(todo) - len(free)} running in another session, "
              f"{len(free)} to run here (order: {getattr(cfg, 'job_order', 'largest_first')})")
        if not free:
            print(f"The remaining {len(todo)} jobs are being run by another session (or by a session that "
                  f"stopped less than {CLAIM_STALE_S // 60} min ago). Nothing to do here: this session stops.\n"
                  f"If no other session is running, run this cell again in {CLAIM_STALE_S // 60} minutes; "
                  f"those jobs will then continue from their checkpoints.")
            break
        out = Parallel(n_jobs=n, backend="loky", verbose=verbose)(delayed(run_job)(j, cfg) for j in free)
        all_out += out
        bad = [o for o in out if o["status"] == "error"]
        print(f"round {rnd}: {sum(o['status'] == 'ok' for o in out)} ok "
              f"({sum(o.get('resumed', 0) > 0 for o in out)} resumed from a checkpoint), "
              f"{sum(o['status'] == 'claimed' for o in out)} taken by another session, {len(bad)} errors")
        for o in bad[:10]:
            print("  ", o["id"], o["error"])
        if bad and len(bad) == len([o for o in out if o["status"] != "claimed"]):
            break               # nothing but errors: stop instead of retrying forever
    return all_out


def status(cfg: Config) -> dict:
    jobs = enumerate_jobs(cfg)
    fps = dataset_fingerprints(cfg)
    for j in jobs:
        j["fp"], j["session"] = fps[j["dataset"]], None
    done = sum(job_path(cfg, j["id"], j["fp"]).exists() for j in jobs)
    running = sum(claimed_elsewhere(cfg, j) for j in jobs if not job_path(cfg, j["id"], j["fp"]).exists())
    errors = list((cfg.results_dir / "jobs").glob("*.error.txt")) if (cfg.results_dir / "jobs").exists() else []
    return {"total": len(jobs), "done": done, "running_now": running, "remaining": len(jobs) - done,
            "error_files": len(errors)}


def calibrate(cfg: Config, paper_cfg: Config, n_workers: int | None = None, n_probe: int = 3) -> "pd.DataFrame":
    """Time full-data evaluations per (dataset, model) at the centre of the search space and at
    n_probe random configurations (single-threaded), and extrapolate to the paper preset."""
    import pandas as pd
    from threadpoolctl import threadpool_limits
    rows = []

    def units(j):   # one CV evaluation = 1 unit; a refit on the full training set ~ 0.5 unit
        if j["method"] == "Default":
            return 1.5
        if j["method"] == "Grid":
            return len(build_grid(j["model"], j["target"])) + 0.5
        return paper_cfg.budget + 0.5 * len(paper_cfg.checkpoints)

    n_jobs_by = {}
    for j in enumerate_jobs(paper_cfg):
        k = (j["dataset"], j["model"])
        n_jobs_by[k] = n_jobs_by.get(k, 0) + units(j)
    for ds_name in cfg.datasets:
        ds = build_dataset(ds_name, paper_cfg, verbose=False)
        data = EncodedData(ds, paper_cfg.cv_folds)
        for model in cfg.models:
            # centre of the space + random configurations (cost is very uneven across the space)
            us = [[0.5] * len(SPACES[model])] + np.random.default_rng(0).random((n_probe, len(SPACES[model]))).tolist()
            secs = []
            with threadpool_limits(1):
                for u in us:
                    t = time.perf_counter()
                    cv_evaluate(data, model, decode_vector(model, u), 0)
                    secs.append(time.perf_counter() - t)
            sec = float(np.mean(secs))
            units = n_jobs_by.get((ds_name, model), 0)
            rows.append({"dataset": ds_name, "model": model, "n_train": len(data.y),
                         "n_features": data.X.shape[1], "sec_per_eval": round(sec, 2),
                         "evaluations": int(units), "cpu_hours": round(sec * units / 3600, 1)})
    df = pd.DataFrame(rows)
    cores = n_workers or (os.cpu_count() if cfg.n_workers in (-1, None) else cfg.n_workers)
    total = df.cpu_hours.sum()
    print(df.to_string(index=False))
    print(f"\nEstimated total: {total:.0f} CPU-hours  ->  ~{total / cores:.0f} h wall-clock on {cores} cores"
          f" (rough: mean of {n_probe + 1} probe configurations per block)")
    return df
