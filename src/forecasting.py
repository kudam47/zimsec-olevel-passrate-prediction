"""
Forecasting module — replaces the original forecasting.py.

Powers main.py's /forecast page. Loads the trained .pkl forecast model
and projects pass rates for a chosen district over a 5-year horizon, with
optional scenario adjustments to simulate "what-if" policy levers.

The result dict returned by run_full_forecast() matches exactly what
templates/forecast.html expects to render.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from data_preparation import run_data_preparation

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "output" / "models"
TABLES_DIR  = BASE_DIR / "output" / "tables"


# ─────────────────────────────────────────────────────────────────────────────
#  Public API used by main.py
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset():
    """Return the latest combined dataset (master + uploads), cleaned."""
    cleaned = DATA_DIR / "cleaned_data.csv"
    if cleaned.exists():
        return pd.read_csv(cleaned)
    # Fall back to a fresh prepare run if cleaned_data.csv hasn't been written
    try:
        return run_data_preparation()
    except Exception:
        master = DATA_DIR / "zimsec_olevel_district_data.csv"
        return pd.read_csv(master) if master.exists() else None


def get_provinces_and_districts(df: pd.DataFrame) -> dict:
    """
    Build {province: [district, ...]} mapping for the dropdowns on
    forecast.html. Districts within each province are sorted alphabetically.
    """
    if df is None or "Province" not in df.columns or "District" not in df.columns:
        return {}
    out = {}
    for prov, sub in df.groupby("Province"):
        out[prov] = sorted(sub["District"].unique().tolist())
    return out


def run_full_forecast(province: str,
                      district: str,
                      start_year: int = 2025,
                      horizon: int = 5,
                      scenario_adjustments: dict = None) -> dict:
    """
    Generate a multi-year forecast for a single district.

    Parameters
    ----------
    province, district    : target location (must exist in the data)
    start_year            : first year to predict
    horizon               : number of years to project (default 5)
    scenario_adjustments  : {feature_name: percent_change}
                            e.g. {'Pupil_Teacher_Ratio': -10} means
                            "what if PTR drops by 10%". Used for the
                            scenario_forecast comparison.

    Returns
    -------
    Dict with keys: province, district, start_year, end_year,
                    last_observed_rate, historical, baseline_forecast,
                    scenario_forecast, delta_summary, narrative,
                    metrics, feature_importances
    """
    scenario_adjustments = scenario_adjustments or {}

    # 1. Load the data and the trained model
    df = load_dataset()
    if df is None:
        return {"error": "No dataset available. Upload data first."}

    sub = df[(df["Province"] == province) & (df["District"] == district)]
    if sub.empty:
        return {"error": f"No historical data for {district}, {province}."}

    try:
        model    = joblib.load(MODELS_DIR / "xgboost.pkl")
        scaler   = joblib.load(MODELS_DIR / "model_scaler.pkl")
        features = joblib.load(MODELS_DIR / "feature_columns.pkl")
    except FileNotFoundError:
        # Fall back to whichever forecast model is available
        try:
            comp = pd.read_csv(TABLES_DIR / "model_comparison.csv")
            best_name = comp.iloc[0]["Model"].lower().replace(" ", "_")
            model    = joblib.load(MODELS_DIR / f"{best_name}.pkl")
            scaler   = joblib.load(MODELS_DIR / "model_scaler.pkl")
            features = joblib.load(MODELS_DIR / "feature_columns.pkl")
        except Exception as e:
            return {"error": f"No trained model found. Run the notebook first. ({e})"}

    # 2. Anchor row = most recent observation for this district
    sub = sub.sort_values(["Year", "Setting"]).copy()
    last_row = sub.iloc[-1].copy()
    last_observed_rate = float(last_row["Pass_Rate_Pct"])

    # 3. Historical series for the chart (last 5 yrs by mean over sittings)
    hist = (sub.groupby("Year")["Pass_Rate_Pct"].mean()
                .reset_index()
                .tail(5)
                .to_dict(orient="records"))
    historical = [{"Year": int(r["Year"]),
                   "Pass_Rate_Pct": round(float(r["Pass_Rate_Pct"]), 2)}
                  for r in hist]

    # 4. Run the forecast — both baseline and scenario
    # Pull RMSE from saved metrics for the confidence band
    metrics = _load_metrics()
    pred_err = metrics.get("rmse", 2.92) if metrics else 2.92

    baseline_forecast = _project(last_row, features, scaler, model,
                                 start_year, horizon, adjustments=None,
                                 prediction_error=pred_err)
    scenario_forecast = None
    if any(v != 0 for v in scenario_adjustments.values()):
        scenario_forecast = _project(last_row, features, scaler, model,
                                     start_year, horizon,
                                     adjustments=scenario_adjustments,
                                     baseline_for_comparison=baseline_forecast,
                                     prediction_error=pred_err)

    # 5. Delta summary (only when scenario is run)
    delta_summary = None
    if scenario_forecast:
        b_end = baseline_forecast[-1]["Predicted_Pass_Rate"]
        s_end = scenario_forecast[-1]["Predicted_Pass_Rate"]
        delta_summary = {
            "baseline_end": round(b_end, 1),
            "scenario_end": round(s_end, 1),
            "improvement": round(s_end - b_end, 2),
        }

    # 6. Narrative
    end_year = start_year + horizon - 1
    end_pred = baseline_forecast[-1]["Predicted_Pass_Rate"]
    direction = ("an increase" if end_pred > last_observed_rate
                 else "a decline" if end_pred < last_observed_rate
                 else "no significant change")
    change_pp = abs(end_pred - last_observed_rate)
    narrative = (
        f"Based on historical performance, {district} is projected to reach "
        f"{end_pred:.1f}% by {end_year}, representing {direction} of "
        f"{change_pp:.1f} percentage points from the most recently observed "
        f"rate of {last_observed_rate:.1f}%. "
    )
    if scenario_forecast and delta_summary:
        narrative += (
            f"Under the chosen scenario adjustments, the {end_year} "
            f"projection shifts to {delta_summary['scenario_end']:.1f}% "
            f"({delta_summary['improvement']:+.1f} pp vs baseline)."
        )

    # 7. Metrics and feature importance (metrics already loaded above)
    feature_importances = _load_feature_importances(top_n=8)

    return {
        "province": province,
        "district": district,
        "start_year": start_year,
        "end_year": end_year,
        "last_observed_rate": round(last_observed_rate, 1),
        "historical": historical,
        "baseline_forecast": baseline_forecast,
        "scenario_forecast": scenario_forecast,
        "delta_summary": delta_summary,
        "narrative": narrative,
        "metrics": metrics,
        "feature_importances": feature_importances,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Internals
# ─────────────────────────────────────────────────────────────────────────────

def _project(anchor_row, features, scaler, model,
             start_year, horizon, adjustments=None,
             baseline_for_comparison=None,
             prediction_error=2.92):
    """
    Iterative forecast: each year's prediction feeds the next year's input.

    Returns rows with all fields the forecast template expects:
        Year, Predicted_Pass_Rate, Lower_CI, Upper_CI,
        Change_Pct (vs previous year),
        Change_From_Baseline (vs the baseline_for_comparison list, scenario only)

    Parameters
    ----------
    prediction_error : float
        Half-width of the 95% confidence interval (≈ RMSE * 1.96).
        Defaults to 2.92 from the saved comparison table.
    """
    adjustments = adjustments or {}
    out = []
    current = anchor_row.copy()

    # Apply scenario adjustments once (percent change applied to baseline value)
    for feat, pct in adjustments.items():
        if feat in current.index and pct:
            current[feat] = float(current[feat]) * (1 + pct / 100.0)

    # Anchor for year-over-year delta
    prior_pred = float(anchor_row.get("Pass_Rate_Pct", 0))

    for offset in range(horizon):
        year = start_year + offset
        x = pd.DataFrame([{f: current.get(f, 0) for f in features}])
        x["Year"] = year
        if "Previous_Year_Pass_Rate_Pct" in features and offset > 0:
            x["Previous_Year_Pass_Rate_Pct"] = out[-1]["Predicted_Pass_Rate"]

        x_scaled = scaler.transform(x[features])
        pred = float(model.predict(x_scaled)[0])
        pred = max(0.0, min(100.0, pred))

        # 95% confidence band
        lo = max(0.0, pred - prediction_error)
        hi = min(100.0, pred + prediction_error)

        # Year-over-year change
        change_pct = pred - prior_pred

        row = {
            "Year": year,
            "Predicted_Pass_Rate": round(pred, 2),
            "Lower_CI": round(lo, 2),
            "Upper_CI": round(hi, 2),
            "Change_Pct": round(change_pct, 2),
        }

        # Scenario row: also include the gap vs the baseline forecast for the same year
        if baseline_for_comparison is not None and offset < len(baseline_for_comparison):
            base_pred = baseline_for_comparison[offset]["Predicted_Pass_Rate"]
            row["Change_From_Baseline"] = round(pred - base_pred, 2)
        else:
            row["Change_From_Baseline"] = 0.0

        out.append(row)
        current["Previous_Year_Pass_Rate_Pct"] = pred
        prior_pred = pred

    return out


def _load_metrics() -> dict:
    """Pull headline metrics from the saved comparison table."""
    try:
        comp = pd.read_csv(TABLES_DIR / "model_comparison.csv")
        top = comp.iloc[0]
        return {
            "model_name": str(top["Model"]),
            "r2":   round(float(top["R2"]), 4),
            "rmse": round(float(top["RMSE"]), 3),
            "mae":  round(float(top["MAE"]), 3),
            "mape": None,  # not computed in our pipeline
            "cv_r2_mean": round(float(top["CV_R2_Mean"]), 4),
        }
    except Exception:
        return {}


def _load_feature_importances(top_n: int = 8):
    """Return [{Feature, Importance}, ...] for the top features."""
    try:
        imp = pd.read_csv(TABLES_DIR / "feature_importance.csv")
        rows = imp.head(top_n).to_dict(orient="records")
        return [{"Feature": r["Feature"],
                 "Importance": round(float(r["Importance"]), 4)}
                for r in rows]
    except Exception:
        return []


if __name__ == "__main__":
    df = load_dataset()
    pd_map = get_provinces_and_districts(df)
    sample_prov = next(iter(pd_map))
    sample_dist = pd_map[sample_prov][0]
    print(f"Test forecast: {sample_dist}, {sample_prov}")
    res = run_full_forecast(sample_prov, sample_dist, 2025, 5)
    print(json.dumps(res, indent=2, default=str)[:1000])
