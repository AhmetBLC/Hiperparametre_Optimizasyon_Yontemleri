"""Download, parse, window and cache the HAR datasets.

Every loader returns either ready-made features (UCI HAR, official 561 features and
official subject split) or a list of continuous single-label *segments*, which are
windowed and passed through the common feature extractor (features.py).

Datasets (all public):
  uci_har      Anguita et al. (2013)  UCI id 240, DOI 10.24432/C54S4K
  wisdm        Kwapisz et al. (2011)  WISDM Activity Prediction v1.1
  pamap2       Reiss & Stricker (2012) UCI id 231, DOI 10.24432/C5NW2H
  motionsense  Malekzadeh et al. (2019) github.com/mmalekzadeh/motion-sense
  harth        Logacjov et al. (2021)  UCI id 779, DOI 10.24432/C5NC90
"""
from __future__ import annotations

import json
import re
import shutil
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import extract_features

URLS = {
    "uci_har": ["https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"],
    "wisdm": ["https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz",
              "http://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"],
    "pamap2": ["https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"],
    "motionsense": ["https://github.com/mmalekzadeh/motion-sense/archive/refs/heads/master.zip",
                    "https://github.com/mmalekzadeh/motion-sense/archive/refs/heads/main.zip"],
    "harth": ["https://archive.ics.uci.edu/static/public/779/harth.zip"],
}
KAGGLE_FALLBACK = {
    "uci_har": "uciml/human-activity-recognition-with-smartphones",
    "motionsense": "malekzadeh/motionsense-dataset",
}
DESCRIPTIONS = {
    "uci_har": "UCI HAR: smartphone at the waist, acc+gyro 50 Hz, official 561 features",
    "wisdm": "WISDM v1.1: smartphone in trouser pocket, accelerometer 20 Hz",
    "pamap2": "PAMAP2: 3 IMUs (wrist, chest, ankle), acc+gyro 100 Hz, protocol activities",
    "motionsense": "MotionSense: iPhone in trouser pocket, device-motion 50 Hz",
    "harth": "HARTH: 2 accelerometers (lower back, thigh), 50 Hz, free-living",
}


@dataclass
class Dataset:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    g_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    g_test: np.ndarray
    classes: list
    feature_names: list
    meta: dict

    @property
    def n_classes(self) -> int:
        return len(self.classes)


# --------------------------------------------------------------------------- download
def _download(url: str, dest: Path, retries: int = 3) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            print(f"  downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (hpo-har benchmark)"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt} failed: {e}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"could not download {url}")


def _extract_recursive(archive: Path, out_dir: Path) -> None:
    """Extract zip/tar archives, including archives nested inside them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [archive]
    seen = set()
    while todo:
        a = todo.pop()
        if a in seen:
            continue
        seen.add(a)
        target = out_dir if a == archive else a.with_name(a.name + "_x")
        name = a.name.lower()
        if name.endswith(".zip"):
            with zipfile.ZipFile(a) as z:
                for m in z.infolist():
                    p = (target / m.filename).resolve()
                    if not str(p).startswith(str(target.resolve())):   # zip-slip guard
                        continue
                    if "__MACOSX" in m.filename:
                        continue
                    z.extract(m, target)
        elif name.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(a) as t:
                safe = [m for m in t.getmembers()
                        if not m.name.startswith(("/", "..")) and ".." not in Path(m.name).parts
                        and (m.isfile() or m.isdir())]
                t.extractall(target, members=safe)
        else:
            continue
        for p in target.rglob("*"):
            if p.is_file() and p.name.lower().endswith((".zip", ".tar.gz", ".tgz")) and p not in seen:
                todo.append(p)


def fetch(name: str, raw_dir: Path) -> Path:
    """Return a directory containing the extracted dataset (downloads once)."""
    d = raw_dir / name
    done = d / ".extracted"
    if done.exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    # A manually placed copy (any archive or extracted files) is used as is.
    reserved = {".extracted", "_archive", "extracted", "kaggle"}
    manual = [p for p in d.iterdir() if p.name not in reserved and not p.name.endswith("_x")]
    if manual:
        for p in manual:
            if p.is_file() and p.name.lower().endswith((".zip", ".tar.gz", ".tgz")):
                _extract_recursive(p, d / (p.name + "_x"))
        done.touch()
        return d
    errors = []
    for url in URLS[name]:
        try:
            fname = url.rstrip("/").split("/")[-1].replace("+", "_")
            arc = _download(url, d / "_archive" / fname)
            _extract_recursive(arc, d / "extracted")
            done.touch()
            return d
        except Exception as e:  # noqa: BLE001
            errors.append(f"{url}: {e}")
    if name in KAGGLE_FALLBACK:
        try:
            import kagglehub
            p = Path(kagglehub.dataset_download(KAGGLE_FALLBACK[name]))
            shutil.copytree(p, d / "kaggle", dirs_exist_ok=True)
            for a in (d / "kaggle").rglob("*.zip"):
                _extract_recursive(a, a.with_name(a.name + "_x"))
            done.touch()
            return d
        except Exception as e:  # noqa: BLE001
            errors.append(f"kagglehub {KAGGLE_FALLBACK[name]}: {e}")
    raise RuntimeError(
        f"Could not obtain dataset {name!r}.\n" + "\n".join(errors) +
        f"\nDownload it manually and put the archive (or extracted folder) into {d}")


# --------------------------------------------------------------------------- parsers
def _find(root: Path, pattern: str, must_contain: str | None = None) -> list[Path]:
    out = sorted(p for p in root.rglob(pattern) if p.is_file()
                 and (must_contain is None or must_contain.lower() in str(p).lower()))
    return out


def load_uci_har(root: Path) -> dict:
    xs = _find(root, "X_train.txt")
    if xs:
        base = xs[0].parent.parent
        names = pd.read_csv(base / "features.txt", sep=r"\s+", header=None)[1].tolist()
        labels = pd.read_csv(base / "activity_labels.txt", sep=r"\s+", header=None)
        lab = dict(zip(labels[0], labels[1]))
        out = {}
        for part in ["train", "test"]:
            X = pd.read_csv(base / part / f"X_{part}.txt", sep=r"\s+", header=None).values
            y = pd.read_csv(base / part / f"y_{part}.txt", header=None)[0].map(lab).values
            g = pd.read_csv(base / part / f"subject_{part}.txt", header=None)[0].values
            out[part] = (X.astype(np.float32), y.astype(str), g.astype(int))
        return {"features": out, "feature_names": names}
    tr, te = _find(root, "train.csv"), _find(root, "test.csv")
    if tr and te:
        out, names = {}, None
        for part, f in [("train", tr[0]), ("test", te[0])]:
            df = pd.read_csv(f)
            names = [c for c in df.columns if c not in ("subject", "Activity")]
            out[part] = (df[names].values.astype(np.float32), df["Activity"].astype(str).values,
                         df["subject"].astype(int).values)
        return {"features": out, "feature_names": names}
    raise FileNotFoundError(f"UCI HAR files not found under {root}")


def _split_runs(labels: np.ndarray, gaps: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Index ranges [start, end) of consecutive equal labels, also split where gaps is True."""
    n = len(labels)
    if n == 0:
        return []
    brk = np.zeros(n, dtype=bool)
    brk[1:] = labels[1:] != labels[:-1]
    if gaps is not None:
        brk[1:] |= gaps[1:]
    starts = np.flatnonzero(brk | (np.arange(n) == 0))
    ends = np.append(starts[1:], n)
    return list(zip(starts, ends))


def load_wisdm(root: Path) -> dict:
    files = [p for p in _find(root, "*.txt") if "raw" in p.name.lower() and "about" not in p.name.lower()]
    if not files:
        raise FileNotFoundError(f"WISDM raw file not found under {root}")
    text = files[0].read_text(errors="ignore")
    rows = []
    for rec in re.split(r"[;\n\r]+", text):
        parts = [t.strip() for t in rec.split(",")]
        if len(parts) < 6:
            continue
        try:
            rows.append((int(parts[0]), parts[1], int(float(parts[2])),
                         float(parts[3]), float(parts[4]), float(parts[5])))
        except ValueError:
            continue
    df = pd.DataFrame(rows, columns=["user", "activity", "ts", "x", "y", "z"])
    df = df[df.ts > 0].reset_index(drop=True)
    key = (df.user.astype(str) + "|" + df.activity).values
    dt = np.diff(df.ts.values, prepend=df.ts.values[0])
    gaps = (dt <= 0) | (dt > 1_000_000_000)          # timestamps are in nanoseconds; split at >1 s gaps
    segs = []
    for s, e in _split_runs(key, gaps):
        seg = df.iloc[s:e]
        segs.append({"subject": int(seg.user.iloc[0]), "label": seg.activity.iloc[0],
                     "data": seg[["x", "y", "z"]].values.astype(np.float32)})
    return {"segments": segs, "fs": 20.0, "channels": ["acc_x", "acc_y", "acc_z"], "triads": [(0, 1, 2)]}


PAMAP2_ACTIVITIES = {1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running", 6: "cycling",
                     7: "nordic_walking", 12: "ascending_stairs", 13: "descending_stairs",
                     16: "vacuum_cleaning", 17: "ironing", 24: "rope_jumping"}


def load_pamap2(root: Path) -> dict:
    files = [p for p in _find(root, "subject1*.dat") if "protocol" in str(p).lower()]
    if not files:
        raise FileNotFoundError(f"PAMAP2 Protocol/subject1xx.dat files not found under {root}")
    # 0-based column indices (timestamp, activityID, heart rate, then 3 IMU blocks of 17 values)
    blocks = {"hand": 3, "chest": 20, "ankle": 37}
    cols, names, triads = [], [], []
    for imu, b in blocks.items():
        for sensor, off in [("acc", 1), ("gyro", 7)]:           # acc = +-16 g accelerometer
            triads.append((len(names), len(names) + 1, len(names) + 2))
            for k, ax in enumerate("xyz"):
                cols.append(b + off + k)
                names.append(f"{imu}{sensor}_{ax}")
    segs = []
    for f in files:
        subj = int(re.findall(r"subject(\d+)", f.name)[0])
        if subj == 109:                                         # only a few minutes of protocol data
            continue
        df = pd.read_csv(f, sep=r"\s+", header=None, na_values=["NaN"])
        act = df[1].values.astype(int)
        ts = df[0].values
        gaps = np.diff(ts, prepend=ts[0]) > 0.05
        for s, e in _split_runs(act, gaps):
            a = act[s]
            if a not in PAMAP2_ACTIVITIES:
                continue
            seg = df.iloc[s:e, cols].interpolate(limit_direction="both").values
            if np.isnan(seg).any():
                continue
            segs.append({"subject": subj, "label": PAMAP2_ACTIVITIES[a], "data": seg.astype(np.float32)})
    return {"segments": segs, "fs": 100.0, "channels": names, "triads": triads}


MOTIONSENSE_ACTS = {"dws": "downstairs", "ups": "upstairs", "wlk": "walking", "jog": "jogging",
                    "sit": "sitting", "std": "standing"}


def load_motionsense(root: Path) -> dict:
    pat = re.compile(r"^(dws|ups|wlk|jog|sit|std)_(\d+)$")
    files = [p for p in _find(root, "sub_*.csv")
             if pat.match(p.parent.name) and "devicemotion" in str(p).lower()]
    if not files:
        raise FileNotFoundError(f"MotionSense A_DeviceMotion_data/<act>_<trial>/sub_<n>.csv not found under {root}")
    seen = set()
    groups = {"useracc": "userAcceleration", "gravity": "gravity", "gyro": "rotationRate"}
    names, triads = [], []
    for g in groups:
        triads.append((len(names), len(names) + 1, len(names) + 2))
        names += [f"{g}_{ax}" for ax in "xyz"]
    segs = []
    for f in files:
        act, trial = pat.match(f.parent.name).groups()
        subj = int(re.findall(r"sub_(\d+)", f.name)[0])
        if (act, trial, subj) in seen:          # the same file can exist in several mirrors
            continue
        seen.add((act, trial, subj))
        df = pd.read_csv(f)
        data = np.column_stack([df[f"{src}.{ax}"].values for src in groups.values() for ax in "xyz"])
        segs.append({"subject": subj, "label": MOTIONSENSE_ACTS[act], "data": data.astype(np.float32)})
    return {"segments": segs, "fs": 50.0, "channels": names, "triads": triads}


HARTH_ACTIVITIES = {1: "walking", 2: "running", 3: "shuffling", 4: "stairs_up", 5: "stairs_down",
                    6: "standing", 7: "sitting", 8: "lying", 13: "cycling_sit", 14: "cycling_stand"}


def load_harth(root: Path) -> dict:
    files = [p for p in _find(root, "S0*.csv") if re.match(r"^S\d{3}\.csv$", p.name)]
    if not files:
        raise FileNotFoundError(f"HARTH S0xx.csv files not found under {root}")
    seen = set()
    cols = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
    segs = []
    for f in files:
        subj = int(f.stem[1:])
        if subj in seen:
            continue
        seen.add(subj)
        df = pd.read_csv(f)
        lab = df["label"].replace({130: 13, 140: 14}).values.astype(int)   # inactive cycling -> cycling
        gaps = None
        try:
            t = pd.to_datetime(df["timestamp"]).values.astype("datetime64[ms]").astype(np.int64)
            gaps = np.diff(t, prepend=t[0]) > 1000
        except Exception:  # noqa: BLE001
            pass
        X = df[cols].values.astype(np.float32)
        for s, e in _split_runs(lab, gaps):
            if lab[s] in HARTH_ACTIVITIES:
                segs.append({"subject": subj, "label": HARTH_ACTIVITIES[lab[s]], "data": X[s:e]})
    return {"segments": segs, "fs": 50.0, "channels": cols,
            "triads": [(0, 1, 2), (3, 4, 5)]}


LOADERS = {"uci_har": load_uci_har, "wisdm": load_wisdm, "pamap2": load_pamap2,
           "motionsense": load_motionsense, "harth": load_harth}


# --------------------------------------------------------------------------- windowing / splitting
def make_windows(raw: dict, window_sec: float, overlap: float, cap: int | None, rng: np.random.Generator):
    fs = raw["fs"]
    L = int(round(window_sec * fs))
    step = max(1, int(round(L * (1.0 - overlap))))
    index = []                                   # (segment id, start, subject, label)
    for i, s in enumerate(raw["segments"]):
        n = len(s["data"])
        for st in range(0, n - L + 1, step):
            index.append((i, st, s["subject"], s["label"]))
    if not index:
        raise ValueError("no windows could be formed")
    idx = pd.DataFrame(index, columns=["seg", "start", "subject", "label"])
    if cap:
        keep = []
        for _, grp in idx.groupby(["subject", "label"], sort=True):
            keep.append(grp.sample(n=cap, random_state=int(rng.integers(2**31))) if len(grp) > cap else grp)
        idx = pd.concat(keep).sort_index()
    W = np.empty((len(idx), L, len(raw["channels"])), dtype=np.float32)
    for k, (seg, st) in enumerate(zip(idx.seg.values, idx.start.values)):
        W[k] = raw["segments"][seg]["data"][st:st + L]
    return W, idx.label.values.astype(str), idx.subject.values.astype(int), L


def drop_rare_classes(y, g, min_subjects=3, min_windows=20):
    ok = []
    for c in np.unique(y):
        m = y == c
        if len(np.unique(g[m])) >= min_subjects and m.sum() >= min_windows:
            ok.append(c)
    return np.isin(y, ok), sorted(set(np.unique(y)) - set(ok))


def subject_split(y, g, frac, seed, min_test=3):
    subjects = np.unique(g)
    k = min(len(subjects) - 3, max(min_test, int(round(frac * len(subjects)))))   # keep >= 3 for CV
    rng = np.random.default_rng(seed)
    classes = set(np.unique(y))
    best, best_cov = None, -1
    for _ in range(2000):
        test = set(rng.permutation(subjects)[:k].tolist())
        mt = np.isin(g, list(test))
        cov = len(set(y[mt])) + len(set(y[~mt]))
        if cov > best_cov:
            best, best_cov = test, cov
        if cov == 2 * len(classes):
            break
    return np.isin(g, list(best)), best_cov == 2 * len(classes)


# --------------------------------------------------------------------------- public API
def _cache_key(cfg) -> dict:
    return {"window_sec": cfg.window_sec, "window_overlap": cfg.window_overlap,
            "cap": cfg.max_windows_per_subject_class, "test_frac": cfg.test_subject_fraction,
            "split_seed": cfg.split_seed, "min_test_subjects": cfg.min_test_subjects, "version": 2}


def build_dataset(name: str, cfg, verbose: bool = True) -> Dataset:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    npz, js = cfg.cache_dir / f"{name}.npz", cfg.cache_dir / f"{name}.json"
    key = _cache_key(cfg)
    if npz.exists() and js.exists():
        meta = json.loads(js.read_text())
        if meta.get("cache_key") == key:
            z = np.load(npz, allow_pickle=False)
            return Dataset(name, z["X_train"], z["y_train"], z["g_train"], z["X_test"], z["y_test"],
                           z["g_test"], meta["classes"], meta["feature_names"], meta)
    t0 = time.time()
    root = fetch(name, cfg.raw_dir)
    raw = LOADERS[name](root)
    meta = {"name": name, "description": DESCRIPTIONS[name], "cache_key": key}
    if "features" in raw:                                 # UCI HAR: official features and split
        (Xtr, ytr, gtr), (Xte, yte, gte) = raw["features"]["train"], raw["features"]["test"]
        feat_names = raw["feature_names"]
        meta.update(fs=50.0, window_samples=128, features="official 561 features",
                    split=f"official ({len(np.unique(gtr))} train / {len(np.unique(gte))} test subjects)")
    else:
        rng = np.random.default_rng(cfg.split_seed)
        W, y, g, L = make_windows(raw, cfg.window_sec, cfg.window_overlap,
                                  cfg.max_windows_per_subject_class, rng)
        keep, dropped = drop_rare_classes(y, g)
        W, y, g = W[keep], y[keep], g[keep]
        X, feat_names = extract_features(W, raw["fs"], raw["channels"], raw["triads"])
        del W
        test_mask, full_cov = subject_split(y, g, cfg.test_subject_fraction, cfg.split_seed, cfg.min_test_subjects)
        Xtr, ytr, gtr = X[~test_mask], y[~test_mask], g[~test_mask]
        Xte, yte, gte = X[test_mask], y[test_mask], g[test_mask]
        meta.update(fs=raw["fs"], window_samples=L, channels=raw["channels"], dropped_classes=dropped,
                    split=f"subject hold-out ({len(np.unique(gtr))} train / {len(np.unique(gte))} test subjects)",
                    split_covers_all_classes=bool(full_cov), features="common extractor (features.py)")
    # plain numpy dtypes (pandas >= 3 may hand back string extension arrays)
    ytr, yte = np.asarray(ytr, dtype=object).astype("U"), np.asarray(yte, dtype=object).astype("U")
    gtr, gte = np.asarray(gtr).astype(np.int64), np.asarray(gte).astype(np.int64)
    Xtr, Xte = np.ascontiguousarray(Xtr, dtype=np.float32), np.ascontiguousarray(Xte, dtype=np.float32)
    classes = sorted(set(ytr.tolist()) | set(yte.tolist()))
    meta.update(classes=classes, feature_names=list(feat_names),
                n_train=int(len(ytr)), n_test=int(len(yte)), n_features=int(Xtr.shape[1]),
                n_subjects_train=int(len(np.unique(gtr))), n_subjects_test=int(len(np.unique(gte))),
                train_subjects=sorted(int(s) for s in np.unique(gtr)),
                test_subjects=sorted(int(s) for s in np.unique(gte)),
                class_counts_train={c: int((ytr == c).sum()) for c in classes},
                class_counts_test={c: int((yte == c).sum()) for c in classes},
                build_seconds=round(time.time() - t0, 1))
    np.savez_compressed(npz, X_train=Xtr, y_train=ytr, g_train=gtr, X_test=Xte, y_test=yte, g_test=gte)
    js.write_text(json.dumps(meta, indent=1))
    if verbose:
        print(f"  {name}: train {Xtr.shape}, test {Xte.shape}, {len(classes)} classes "
              f"({meta['build_seconds']} s)")
    return Dataset(name, Xtr, ytr, gtr, Xte, yte, gte, classes, list(feat_names), meta)


def subsample_train(ds: Dataset, n: int, seed: int = 0) -> Dataset:
    """Smoke-test helper: keep n training windows, stratified by class."""
    if n is None or len(ds.y_train) <= n:
        return ds
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(ds.y_train):
        m = np.flatnonzero(ds.y_train == c)
        idx.append(rng.choice(m, size=max(2, int(round(n * len(m) / len(ds.y_train)))), replace=False)
                   if len(m) > 2 else m)
    idx = np.sort(np.concatenate(idx))
    return Dataset(ds.name, ds.X_train[idx], ds.y_train[idx], ds.g_train[idx], ds.X_test, ds.y_test,
                   ds.g_test, ds.classes, ds.feature_names, dict(ds.meta, smoke_subsample=n))
