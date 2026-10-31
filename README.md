# HPO-HAR benchmark (v2)

Code for the study *"The effect of hyperparameter optimisation methods on classification performance in
human activity recognition"*. It compares six hyperparameter optimisation (HPO) methods on three classifiers
and five public HAR datasets under an equal evaluation budget, with subject-independent validation.

| | |
|---|---|
| **HPO methods** | Grid Search, Random Search, Bayesian optimisation (TPE), CMA-ES, real-coded Genetic Algorithm, BOHB (TPE + Hyperband) |
| **Classifiers** | SVM (RBF), Random Forest, XGBoost |
| **Datasets** | UCI HAR, WISDM v1.1, PAMAP2, MotionSense, HARTH |
| **Seeds** | 10 per stochastic method (SVM Grid/Default are deterministic and run once) |
| **Budget** | 100 full 3-fold GroupKFold evaluations; any-time results at 10, 25, 50, 100 |

## Design decisions

* **Same space for every method.** Continuous optimisers sample inside the bounds of `hpo_har/spaces.py`; Grid
  Search uses k points per dimension at the centres of k equal-width cells spanning the same bounds (log scale
  where applicable). Grid sizes ≈ 50 and ≈ 100 so that Grid is compared at both budgets.
* **Defaults inside the space.** Bounds were chosen so that the library defaults lie inside (e.g. Random Forest
  `max_features='sqrt'`), and the analysis reports how often selected values sit within 2 % of a bound.
* **Subject-independent evaluation.** Optimisation uses GroupKFold over training subjects; the test set consists of
  held-out subjects (official split for UCI HAR, a fixed 30 % subject hold-out otherwise).
* **Exact budget curves.** All methods are sequential, so the first B evaluations of a 100-evaluation run equal a
  run with budget B. BOHB uses training-set fractions 1/9, 1/3, 1 as fidelities and its cost is counted in
  full-evaluation equivalents (1/9 + 1/3 + 1 for a trial that reaches the top rung).
* **Fair timing.** Every evaluation runs on one CPU thread; many jobs run in parallel, one per core.
  CPU time (not wall time) is reported.
* **Statistics.** Blocks = dataset × classifier (15). Friedman + Iman–Davenport, Kendall's W, Nemenyi CD,
  Holm-corrected Wilcoxon signed-rank tests across blocks; Mann–Whitney tests over seeds against Random Search
  within each block; subject-level cluster bootstrap CIs on the test set (windows of one subject are resampled
  together, which McNemar's test cannot account for). fANOVA importance is computed on Random Search trials.
* **No early stopping, no probability calibration.** XGBoost `n_estimators` is searched; SVM ROC-AUC is computed
  from one-vs-rest decision values, and log-loss is reported only for RF and XGBoost.

## Running

```bash
pip install -r requirements.txt
python run.py --preset smoke --work-dir ./hpo_work           # ~15-30 min: downloads, checks, runtime estimate
python run.py --preset paper --work-dir ./hpo_work --workers 8   # full study, resumable
python run.py --preset paper --work-dir ./hpo_work --stage analyze
```

Several machines can share the work: give each the same `--work-dir` (e.g. a synced folder) and
`--num-shards N --shard-index i`. On Google Colab use `HPO_HAR_v2.ipynb`.

Datasets are downloaded from their original sources (UCI Machine Learning Repository, Fordham WISDM Lab,
GitHub). If a download fails, put the archive into `<work-dir>/raw_downloads/<dataset>/` and run again.

## Outputs

`<work-dir>/results/<preset>/jobs/` — one gzipped JSON per job (all evaluations, selected configurations, test
metrics and test predictions). `<work-dir>/results/<preset>/analysis/` — tables (CSV + LaTeX) and figures
(PNG + PDF).

---

## Türkçe hızlı başlangıç

1. `hpo_har_v2.zip` dosyasını Google Drive'ın ana klasörüne yükle, `HPO_HAR_v2.ipynb`'yi Colab'da aç.
2. `PRESET = "smoke"` ile tüm hücreleri çalıştır. Bu adım veri setlerini indirir, her kod yolunu küçük bir
   örneklemle dener ve tam çalışmanın kaç saat süreceğini tahmin eder. Çıktıyı (özellikle hata varsa) paylaş.
3. `PRESET = "paper"` yapıp 4. bölümü çalıştır. Bağlantı koparsa notebook'u yeniden çalıştırman yeterli; biten
   işler atlanır. Birden fazla Colab/Kaggle oturumu ya da bilgisayar aynı Drive klasörüne yazarak işi
   `NUM_SHARDS` / `SHARD_INDEX` ile paylaşabilir.
4. 5. bölüm tüm tabloları ve İngilizce şekilleri üretir ve ZIP olarak indirir.

## Results in this repository

- `results/jobs_paper.zip` – raw result files of all 1,065 searches (one `.json.gz` per search: every evaluation, the
  selected configurations at budgets 10/25/50/100, test metrics and environment). `hpo_har.analysis.Results(cfg, validate=False)`
  reads them without the datasets.
- `results/*.csv` – all tables of the paper (T01–T13); `results/figures/` – all figures (PNG + PDF).

## License and citation

MIT License (see `LICENSE`). Please cite the archived version (Zenodo DOI) – see `CITATION.cff`.
