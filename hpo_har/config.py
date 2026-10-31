"""Experiment configuration and presets."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

ALL_DATASETS = ["uci_har", "wisdm", "pamap2", "motionsense", "harth"]
ALL_MODELS = ["SVM", "RandomForest", "XGBoost"]
# Order matters only for display; "Grid" is run at two grid sizes (see spaces.grid_targets).
ALL_METHODS = ["Grid", "Random", "TPE", "CMAES", "GA", "BOHB"]


@dataclass
class Config:
    preset: str = "paper"
    work_dir: str = "./hpo_work"            # cache + results live here (point to Google Drive on Colab)
    datasets: list = field(default_factory=lambda: list(ALL_DATASETS))
    models: list = field(default_factory=lambda: list(ALL_MODELS))
    methods: list = field(default_factory=lambda: list(ALL_METHODS))
    seeds: list = field(default_factory=lambda: list(range(10)))
    budget: int = 100                        # full-evaluation equivalents per search run
    checkpoints: list = field(default_factory=lambda: [10, 25, 50, 100])  # budgets evaluated on the test set
    grid_targets: list = field(default_factory=lambda: [50, 100])       # approximate grid sizes (= checkpoints)
    cv_folds: int = 3                        # GroupKFold on training subjects
    window_sec: float = 5.0                  # raw-signal datasets
    window_overlap: float = 0.0              # 0 = non-overlapping windows
    max_windows_per_subject_class: int = 100 # cap applied to raw-signal datasets (None = no cap)
    test_subject_fraction: float = 0.3       # for datasets without an official split
    min_test_subjects: int = 3               # at least 3 held-out subjects (needed for a subject bootstrap)
    split_seed: int = 2026                   # fixed subject split (independent of HPO seeds)
    ga_pop_size: int = 10
    bohb_eta: int = 3                        # Hyperband reduction factor; fidelities 1/9, 1/3, 1
    n_workers: int = -1                      # parallel jobs (-1 = all cores); every evaluation is single-threaded
    shard_index: int = 0                     # split the job list over several machines/sessions
    num_shards: int = 1
    job_order: str = "largest_first"         # "smallest_first" for a second session working in parallel
    bootstrap_reps: int = 2000
    max_train_rows: int | None = None        # smoke test only: subsample training windows for speed
    raw_root: str | None = None              # where raw downloads are extracted (default: work_dir/raw_downloads);
                                             # on Colab use local disk, e.g. "/content/hpo_raw" (Drive is slow for many files)

    @property
    def root(self) -> Path:
        return Path(self.work_dir)

    @property
    def cache_dir(self) -> Path:
        return self.root / "data_cache"

    @property
    def raw_dir(self) -> Path:
        return Path(self.raw_root) if self.raw_root else self.root / "raw_downloads"

    @property
    def results_dir(self) -> Path:
        return self.root / "results" / self.preset

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def get_config(preset: str = "paper", **overrides) -> Config:
    """Return a configuration preset.

    paper : full study (10 seeds, budget 100, 5 datasets, 3 models, 6 methods)
    smoke : quick end-to-end check of downloads, parsing and every code path on a small
            training subsample, plus a runtime calibration on the full data that estimates
            the cost of the paper preset. Builds the same dataset cache as 'paper'.
    """
    if preset == "paper":
        cfg = Config(preset="paper")
    elif preset == "smoke":
        cfg = Config(preset="smoke", seeds=[0], budget=6, checkpoints=[3, 6], grid_targets=[3, 6],
                     cv_folds=2, bootstrap_reps=200, ga_pop_size=3, max_train_rows=600)
    else:
        raise ValueError(f"unknown preset {preset!r}")
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"Config has no field {k!r}")
        setattr(cfg, k, v)
    return cfg
