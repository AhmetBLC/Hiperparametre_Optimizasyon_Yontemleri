"""HPO comparison benchmark for human activity recognition (v2).

Modules
-------
config      experiment presets
datasets    download, parsing, windowing and caching of the HAR datasets
features    vectorised time/frequency-domain feature extraction
spaces      hyperparameter search spaces, defaults and grid construction
models      classifier factories (SVM, RandomForest, XGBoost)
evaluation  subject-grouped cross-validation and test-set metrics
optimizers  Grid, Random, TPE, CMA-ES, Genetic Algorithm, BOHB
runner      job enumeration, sharding, parallel execution with resume
analysis    tables and statistical tests
figures     publication figures
"""
__version__ = "2.0.5"
