"""Builds HPO_HAR_v2.ipynb (run once by the maintainer; the notebook is what users open)."""
import json

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


md("""
# Hyperparameter optimisation methods for human activity recognition — v2 benchmark

Six HPO methods (Grid Search, Random Search, Bayesian optimisation/TPE, CMA-ES, Genetic Algorithm, BOHB)
× three classifiers (SVM-RBF, Random Forest, XGBoost) × five HAR datasets (UCI HAR, WISDM, PAMAP2,
MotionSense, HARTH) × 10 seeds, with subject-grouped cross-validation, equal budgets, exact budget–performance
curves, subject-level bootstrap CIs and Friedman/Iman–Davenport/Wilcoxon–Holm statistics.

**How to use**
1. First run everything with `PRESET = "smoke"` (Runtime → Run all, ≈20–40 min). It downloads all datasets,
   checks every code path on a small subsample, estimates the runtime of the full study and finally downloads
   a small report file (`hpo_report_smoke.txt`).
2. Set `PRESET = "paper"` and run section 4. It is **resumable**: if the session disconnects, run the
   notebook again and finished jobs are skipped. Several sessions/machines can share the work with
   `NUM_SHARDS` / `SHARD_INDEX` while writing to the same Google Drive folder.
3. Section 5 produces all tables and figures (English) and a ZIP to download.
""")

md("""
## 1. Setup
The package code is embedded in this notebook (next cell), so nothing has to be uploaded.
When Colab asks for access to Google Drive, allow it: results are saved there so nothing is lost.
""")
code("""
import os, sys, subprocess, base64, io, zipfile
IN_COLAB = "google.colab" in sys.modules
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna>=3.6", "cmaes>=0.10", "xgboost>=2.0",
                "threadpoolctl", "kagglehub"], check=False)
CODE_B64 = "__CODE_B64__"
CODE_DIR = "/content" if IN_COLAB else os.path.abspath("hpo_code")
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_B64))).extractall(CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, "hpo_har_v2"))
for m in [k for k in sys.modules if k.startswith("hpo_har")]:
    del sys.modules[m]
import hpo_har
print("hpo_har", hpo_har.__version__, "ready")
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
""")

md("""
## 2. Configuration
`WORK_DIR` holds the downloaded data, the feature cache and all results — keep it on Google Drive so nothing
is lost when the Colab runtime is recycled.
""")
code("""
from hpo_har.config import get_config
from hpo_har.runner import prepare_datasets, run_all, status, calibrate, environment_info

PRESET      = "smoke"      # "smoke" first, then "paper"
WORK_DIR    = "/content/drive/MyDrive/hpo_work" if IN_COLAB else "./hpo_work"
RAW_DIR     = "/content/hpo_raw" if IN_COLAB else None   # raw archives on local disk; the feature cache stays on Drive
N_WORKERS   = -1           # parallel jobs (-1 = all cores); each evaluation is single-threaded
NUM_SHARDS  = 1            # e.g. 3 when three sessions/machines share the work
SHARD_INDEX = 0            # 0, 1, 2 ... (different in each session)
JOB_ORDER   = "largest_first"   # in a SECOND Colab session running at the same time use "smallest_first"

cfg = get_config(PRESET, work_dir=WORK_DIR, raw_root=RAW_DIR, n_workers=N_WORKERS,
                 num_shards=NUM_SHARDS, shard_index=SHARD_INDEX, job_order=JOB_ORDER)
print(cfg.to_json())
print(environment_info())
""")

md("## 3. Datasets (download, windowing, feature extraction — cached after the first run)")
code("""
import json, pandas as pd
prepare_datasets(cfg)
from hpo_har.analysis import table_datasets
display(table_datasets(cfg))
""")

md("""
## 4. Experiments
With `PRESET = "smoke"` this also calibrates the runtime of the full study. With `PRESET = "paper"` re-run this
cell after any disconnect; it continues where it stopped.
""")
code("""
run_all(cfg)
print(status(cfg))
if cfg.preset == "smoke":
    paper = get_config("paper", work_dir=WORK_DIR, raw_root=RAW_DIR)
    est = calibrate(cfg, paper)
""")

md("## 5. Analysis: tables, statistics and figures")
code("""
from hpo_har.analysis import run_analysis
out = run_analysis(cfg)
s = out["summary"]
for k in ["friedman_test", "friedman_cv"]:
    if k in s:
        print(k, {a: b for a, b in s[k].items() if a != "avg_rank"}, "\\navg ranks:", s[k]["avg_rank"])
""")
code("""
import glob
from IPython.display import Image, display
for f in sorted(glob.glob(str(cfg.results_dir / "analysis" / "figures" / "*.png"))):
    print(os.path.basename(f)); display(Image(f, width=900))
""")
code("""
import shutil
zip_path = shutil.make_archive(f"/content/hpo_v2_{cfg.preset}_analysis" if IN_COLAB else f"hpo_v2_{cfg.preset}_analysis",
                               "zip", str(cfg.results_dir / "analysis"))
print(zip_path)
if IN_COLAB:
    from google.colab import files
    files.download(zip_path)
""")

md("## 6. Report to send back (one small text file)")
code("""
import glob, json, platform
lines = ["HPO-HAR v2 report", f"preset: {cfg.preset}", f"environment: {json.dumps(environment_info())}",
         f"status: {status(cfg)}", ""]
for n in cfg.datasets:
    p = cfg.cache_dir / f"{n}.json"
    if p.exists():
        m = json.loads(p.read_text())
        lines.append(f"{n}: train {m['n_train']} test {m['n_test']} features {m['n_features']} classes {len(m['classes'])} "
                     f"subjects {m['n_subjects_train']}/{m['n_subjects_test']} dropped {m.get('dropped_classes')} "
                     f"build {m.get('build_seconds')} s")
    else:
        lines.append(f"{n}: NOT BUILT")
if "est" in globals():
    lines += ["", "runtime calibration:", est.to_string(index=False),
              f"total CPU-hours: {est.cpu_hours.sum():.0f}"]
errs = sorted(glob.glob(str(cfg.results_dir / "jobs" / "*.error.txt")))
lines += ["", f"error files: {len(errs)}"]
for f in errs[:20]:
    lines += ["----- " + os.path.basename(f), open(f).read()[-3000:]]
report = "\\n".join(lines)
rp = f"/content/hpo_report_{cfg.preset}.txt" if IN_COLAB else f"hpo_report_{cfg.preset}.txt"
open(rp, "w").write(report)
print(report[:4000])
if IN_COLAB:
    from google.colab import files
    files.download(rp)
""")

nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"colab": {"provenance": [], "toc_visible": True},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "cells": cells}
import base64, io, zipfile, pathlib
buf = io.BytesIO()
root = pathlib.Path(__file__).resolve().parent
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".txt", ".md") and p.name != "make_notebook.py":
            z.write(p, "hpo_har_v2/" + str(p.relative_to(root)))
b64 = base64.b64encode(buf.getvalue()).decode()
for c in nb["cells"]:
    c["source"] = [l.replace("__CODE_B64__", b64) for l in c["source"]]
json.dump(nb, open("HPO_HAR_v2.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote HPO_HAR_v2.ipynb with", len(cells), "cells")
