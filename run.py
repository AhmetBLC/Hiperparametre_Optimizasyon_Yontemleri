"""Command-line entry point (useful on a desktop/server with many cores).

Examples
--------
python run.py --preset smoke --work-dir D:/hpo_work                    # check everything + estimate runtime
python run.py --preset paper --work-dir D:/hpo_work --workers 8        # full study (resumable)
python run.py --preset paper --work-dir /drive/hpo_work --num-shards 3 --shard-index 0   # machine 1 of 3
python run.py --preset paper --work-dir D:/hpo_work --stage analyze    # tables + figures
"""
import argparse

from hpo_har.analysis import run_analysis
from hpo_har.config import get_config
from hpo_har.runner import calibrate, prepare_datasets, run_all, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="smoke", choices=["smoke", "paper"])
    ap.add_argument("--work-dir", default="./hpo_work")
    ap.add_argument("--raw-dir", default=None, help="where raw downloads go (default: <work-dir>/raw_downloads)")
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--stage", default="all", choices=["all", "prepare", "run", "analyze", "status"])
    a = ap.parse_args()
    over = dict(work_dir=a.work_dir, raw_root=a.raw_dir, n_workers=a.workers, num_shards=a.num_shards, shard_index=a.shard_index)
    if a.datasets:
        over["datasets"] = a.datasets
    cfg = get_config(a.preset, **over)
    if a.stage == "status":
        print(status(cfg))
        return
    if a.stage in ("all", "prepare"):
        prepare_datasets(cfg)
    if a.stage in ("all", "run"):
        run_all(cfg)
        print(status(cfg))
        if a.preset == "smoke":
            calibrate(cfg, get_config("paper", **over), n_workers=None if a.workers == -1 else a.workers)
    if a.stage in ("all", "analyze"):
        run_analysis(cfg)


if __name__ == "__main__":
    main()
