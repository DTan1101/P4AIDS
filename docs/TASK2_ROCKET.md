# Task 2 - ROCKET

## Objective

Evaluate MiniRocket and MultiRocket for Rain/Snow
classification under simulated sensor missingness.

Input: 24 hours, 5 meteorological variables.
Target: Rain (0) or Snow (1) at t+1.

## Shared dataset

- Szeged Weather 2006-2016.
- Prepared data: outputs/data/windows.npz.
- Train: 2006-2013.
- Validation: 2014.
- Test: 2015-2016.
- Missing seed: 123.
- Model seed: 42.
- Shared observation masks: outputs/masks/.

Channel order:

1. Temperature
2. Humidity
3. Wind Speed
4. Pressure
5. Visibility

True in a missingness mask means observed.

Do not regenerate shared data or masks.

## Pipeline

Shared mask -> interpolation -> ROCKET transform
-> optional mask/delta features -> feature scaling
-> weighted RidgeClassifier.

Entirely missing channels use the training median.

The transformer and scaler are fitted only on train.
Alpha and threshold are selected on validation Macro F1.
Each missing scenario is fitted independently.

## Resource configuration

MiniRocket kernel budget: 1008.
MultiRocket kernel budget: 336.
Transformer fit subset: 2500.
Batch size: 512.
CPU jobs: 2.

These are reduced kernel budgets, not aeon defaults.
The feature matrix limit does not cap total process RAM.

## Tests

    python -m pytest tests/test_rocket.py -v

## Quick benchmark

    python tools/run_rocket.py --variant mini --full-only --tag quick --kernels 168

## Main benchmark

    python tools/run_rocket.py --variant mini
    python tools/run_rocket.py --variant multi

20 main runs in total: 10 for each variant.

## Ablation examples

    python tools/run_rocket.py --variant mini --missing-kind block --missing-rate 0.5 --tag no_weight --no-weight

    python tools/run_rocket.py --variant mini --missing-kind block --missing-rate 0.5 --tag no_mask --no-mask-delta

    python tools/run_rocket.py --variant mini --full-only --tag temperature_only --channels temperature

    python tools/run_rocket.py --variant mini --missing-kind block --missing-rate 0.5 --tag ffill --imputer ffill

## Reporting

    python tools/plot_rocket_results.py

Primary metric: Macro F1.
PR-AUC: Average Precision for Snow, per shared metric.

Also report balanced accuracy, class-specific metrics,
confusion matrices, runtime and subgroup performance.

Prediction files contain sample indices, timestamps,
labels, predicted classes and uncalibrated decision scores.

The temperature subgroup uses the original last historical
temperature for diagnostics only.

Main results: outputs/results/rocket/.
Extra experiments: outputs/results/rocket_extra/.
Analysis: outputs/results/rocket_analysis/.

## Limitations

- Single weather dataset.
- Simulated sensor missingness.
- Reduced kernel budgets.
- Dependence between overlapping windows.
- Runtime may include first-run Numba compilation.
- Training runtime includes validation tuning.
- Linear parameter count excludes fixed kernels.
- One-class subgroups do not report Macro F1 or PR-AUC.

Use validation scores for model/feature selection.
Never report synthetic smoke tests as real results.

## Git

Commit only Task 2 code, tests, configuration,
documentation and approved report contributions.

Do not commit virtual environments, wheel installers,
temporary outputs or Python caches.
