# ZIMSEC O-Level Predictive Analytics — Updated Bundle

**Author:** Cesario Machinga
**Date:** April 2026

This bundle contains the redesigned modelling pipeline, retrained models, and an
upgraded frontend stylesheet for the ZIMSEC O-Level pass-rate prediction project.

---

## What changed and why

The original pipeline reported R² = 0.975 from a model that included the prior-year
pass rate as a feature, evaluated with random K-fold cross-validation. Two
issues motivated the redesign:

1. **Feature dominance.** `Previous_Year_Pass_Rate_Pct` correlates 0.96 with the
   target on its own. A single-feature linear regression on it alone reaches
   R² = 0.92, meaning everything else in the dataset added only ~0.05 of R²
   on top of pure persistence.

2. **Data leakage in CV.** 78.8% of pass-rate variance lies *between* districts
   and is persistent over time. Random K-fold places the same district in both
   training and test folds across different years, so the model partly memorises
   district baselines rather than learning generalisable patterns.

### The fix — two model families, honest evaluation

| Family | Includes Prior-Year? | Use case |
|---|---|---|
| **Forecast model** | Yes | "What will this district score next year?" |
| **Driver model** | No | "What underlying factors drive pass rates?" |

Both are evaluated with:
- **Temporal hold-out:** train on 2015–2021, test on 2022–2024.
- **GroupKFold by District:** 5-fold cross-validation where each held-out fold
  contains districts the model has never seen.

### Final results

**Forecast family (best: XGBoost)**

| Metric | Value |
|---|---|
| Hold-out R² (2022–2024) | **0.9754** |
| Hold-out MAE | **2.25 pp** |
| GroupKFold-by-district CV R² | **0.9404 ± 0.030** |

**Driver family (best: Random Forest)**

| Metric | Value |
|---|---|
| Hold-out R² (2022–2024) | **0.9195** |
| Hold-out MAE | **4.04 pp** |
| GroupKFold-by-district CV R² | **0.520 ± 0.316** |

The wide CV variance on the driver model is itself the main policy finding:
predicting an unseen district from socio-economic features alone has substantial
uncertainty. The forecast model holds up under district-out CV (0.94),
confirming it generalises *when prior-year data is available*.

### Top driver of pass rates (excluding autoregressive features)

`Proportion_Schools_With_Science_Labs` carries **69%** of the driver model's
feature importance — by far the dominant non-autoregressive lever. This is a
concrete, capital-budget-sized intervention that policymakers can act on.

---

## Bundle contents

```
zimsec_project/
├── README.md                                  ← this file
├── notebooks/
│   └── zimsec_olevel_predictive_model.ipynb   ← main deliverable (49 cells)
├── data/
│   └── cleaned_data.csv                       ← preprocessed data
├── output/
│   ├── models/                                ← .pkl serialised models
│   │   ├── manifest.json
│   │   ├── xgboost.pkl                        ← BEST forecast model
│   │   ├── random_forest.pkl
│   │   ├── linear_regression.pkl
│   │   ├── ridge_regression.pkl
│   │   ├── lasso_regression.pkl
│   │   ├── model_scaler.pkl                   ← StandardScaler for forecast
│   │   ├── feature_columns.pkl                ← feature order for forecast
│   │   ├── driver_random_forest.pkl           ← BEST driver model
│   │   ├── driver_scaler.pkl
│   │   └── driver_feature_columns.pkl
│   ├── tables/
│   │   ├── model_comparison.csv               ← forecast family results
│   │   ├── model_comparison_driver.csv        ← driver family results
│   │   ├── feature_importance.csv             ← forecast model importance
│   │   └── feature_importance_driver.csv      ← driver model importance
│   └── figures/
│       ├── pass_rate_trends.png
│       ├── urban_rural_comparison.png
│       ├── correlation_heatmap.png
│       ├── province_chart.png
│       ├── feature_importance.png
│       ├── shap_summary.png
│       ├── shap_bar.png
│       └── model_comparison.png
└── webapp/
    ├── main.py                                ← patched (loads .pkl, not .joblib)
    └── static/css/
        └── style.css                          ← upgraded editorial-academic UI
```

---

## How to integrate

### 1. Drop-in replacement

Replace these files in your existing project:

| Source | Destination |
|---|---|
| `notebooks/zimsec_olevel_predictive_model.ipynb` | `notebooks/` (new) |
| `data/cleaned_data.csv` | `data/cleaned_data.csv` |
| `output/**` | `output/**` |
| `webapp/main.py` | `webapp/main.py` |
| `webapp/static/css/style.css` | `webapp/static/css/style.css` |

The HTML templates do **not** need to change. The new stylesheet is a drop-in
upgrade — it overrides every custom class your templates already use.

### 2. Re-running the pipeline yourself

If you want to retrain from scratch (recommended at least once to verify):

```bash
cd zimsec_project
jupyter lab notebooks/zimsec_olevel_predictive_model.ipynb
```

Run all cells. The notebook regenerates everything in `output/` and overwrites
the `.pkl` files. Typical execution time: 60–90 seconds (depends on
GridSearchCV).

### 3. Launching the webapp

```bash
python webapp/main.py
# or
uvicorn webapp.main:app --reload
```

Visit `http://127.0.0.1:8000`. Login: `admin@zimsec.ac.zw` / `admin123`.

The dashboard will pick up the new metrics, the new charts, the new SHAP plots,
and render them under the upgraded editorial UI.

---

## Notebook structure

The notebook is mapped one-to-one to your dissertation's five project objectives:

| Section | Objective | Contents |
|---|---|---|
| 1 | Obj. 1 — Data collection & preprocessing | Load, audit, feature engineer, save `cleaned_data.csv` |
| 2 | Obj. 2 — EDA | Trends, urban/rural, province bar, correlation matrix |
| 3 | (methodological) | Variance decomposition, justification of CV strategy |
| 4 | Obj. 3 — Model development | Train both families × five algorithms with GridSearchCV |
| 5 | Obj. 4 — Evaluation | Comparison tables, residual diagnostics |
| 6 | Obj. 4 — Interpretation | Feature importance + SHAP for both families |
| 7 | Obj. 5 — Insights & recommendations | Policy interpretation, honest caveats |
| 8 | (deployment) | Save all `.pkl` artefacts + manifest |

Every code cell is preceded by a markdown cell that explains *why* the step is
needed and what to look for in the output.

---

## Frontend — design rationale

The previous stylesheet was a generic Bootstrap admin theme with bright
stat-card colour blocks. The replacement commits to a single coherent direction:

- **Editorial academic.** Inspired by FT-style data visualisation and
  high-end research publications. Restrained colour, typography-led, generous
  whitespace.
- **Typography pair.** *Newsreader* (variable serif, gives the dashboard a
  publication feel) for display headings, paired with *DM Sans* for UI. Numbers
  use *JetBrains Mono* via `font-variant-numeric: tabular-nums` so columns
  align.
- **Palette.** Deep ink (`#0a1628`) for the sidebar and primary text, warm
  parchment (`#faf7f2`) for the page background, a single brass accent
  (`#b45309`) used sparingly. Status colours (success/danger/warning) are
  desaturated rather than fully saturated.
- **Stat cards.** Replaced the saturated colour blocks with a typography-led
  card that uses a 3px accent rule on the left edge. Quieter, more legible.
- **Animations.** Single staggered fade-in on page load, no scattered
  micro-interactions. Hovering a card lifts it 1px with a softer shadow.

---

## Honest limitations

A few caveats that should appear in the dissertation:

1. **The driver model has wide CV uncertainty (±0.32).** Use it for ranking
   and prioritisation, not for predicting an exact pass rate for a brand-new
   district.
2. **The forecast model assumes prior-year data is available.** This is true
   for the 71 districts in the panel, but a brand-new district has no
   history — for those, the driver model is the only option and the CV variance
   applies.
3. **Typical error is ~2.3 pp (forecast) and ~4 pp (driver).** Do not use the
   model to rank adjacent districts that differ by less than 5 pp on the actual
   metric.
4. **78.8% of variance is structural (between districts).** The model is most
   useful for identifying systemically under-performing districts, not for
   catching short-term shocks (such as the 2020 COVID disruption).

These caveats are also written up in Section 7 of the notebook.
