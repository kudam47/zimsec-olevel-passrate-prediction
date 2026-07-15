"""
ZIMSEC O-Level Predictive Analytics — Web Application
=====================================================
FastAPI backend with Jinja2 templates and Bootstrap 5 sidebar UI.

Launch: python webapp/main.py
   or:  uvicorn webapp.main:app --reload

Author: Cesario Machinga (Academic Research Project)
"""

import os
import sys
import json
import time
import uuid
import hashlib
from datetime import datetime
from pathlib import Path

import uvicorn
import numpy as np
import pandas as pd
import joblib
import matplotlib
# Multi-glob ingestion: combines master file + uploaded files automatically
from ingestion import ingest_multi_glob, build_default_patterns, REQUIRED_COLS, DEDUP_KEYS
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

# ─── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
WEBAPP_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = OUTPUT_DIR / "models"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ─── App Setup ───────────────────────────────────────────────────────────
app = FastAPI(title="ZIMSEC O-Level Analytics", version="1.0.0")
app.add_middleware(SessionMiddleware, secret_key="zimsec-analytics-secret-key-2026")
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR / "static")), name="static")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
templates = Jinja2Templates(directory=str(WEBAPP_DIR / "templates"))

# ─── Auth Config ─────────────────────────────────────────────────────────
USERS = {
    "admin@zimsec.ac.zw": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "name": "Cesario Machinga",
        "role": "Researcher"
    },
    "supervisor@zimsec.ac.zw": {
        "password": hashlib.sha256("supervisor2026".encode()).hexdigest(),
        "name": "Dr. Supervisor",
        "role": "Supervisor"
    }
}

# ─── History Store ───────────────────────────────────────────────────────
history_store = []


def get_current_user(request: Request):
    """Check if user is authenticated."""
    user = request.session.get("user")
    if not user:
        return None
    return user


def require_auth(request: Request):
    """Dependency that requires authentication."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return user


def load_raw_data():
    """
    Load the dataset using multi-glob ingestion.

    Combines the master file with anything uploaded via the dashboard
    (data/uploads/*.csv) or dropped into data/raw/. Files missing required
    columns are skipped silently. Falls back to None if nothing matches.
    """
    try:
        patterns = build_default_patterns(DATA_DIR)
        df = ingest_multi_glob(
            patterns=patterns,
            required_cols=REQUIRED_COLS,
            dedup_keys=DEDUP_KEYS,
            exclude_files=["cleaned_data.csv"],
            verbose=False,
        )
        # Drop the provenance column so downstream code sees the original schema
        return df.drop(columns=["_source_file"], errors="ignore")
    except (FileNotFoundError, ValueError):
        return None


def load_model_results():
    """Load model comparison results."""
    path = TABLES_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_best_model():
    """Load the best trained model."""
    try:
        results = pd.read_csv(TABLES_DIR / "model_comparison.csv")
        best_name = results.iloc[0]['Model']
        safe_name = best_name.lower().replace(' ', '_')
        model = joblib.load(MODELS_DIR / f"{safe_name}.pkl")
        scaler = joblib.load(MODELS_DIR / "model_scaler.pkl")
        features = joblib.load(MODELS_DIR / "feature_columns.pkl")
        return model, scaler, features, best_name, results
    except Exception:
        return None, None, None, None, None


def generate_chart(chart_type="bar"):
    """Generate a matplotlib chart and return the path."""
    df = load_raw_data()
    if df is None:
        return None

    os.makedirs(str(WEBAPP_DIR / "static" / "img"), exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    if chart_type == "province_bar":
        fig, ax = plt.subplots(figsize=(10, 6))
        prov = df.groupby('Province')['Pass_Rate_Pct'].mean().sort_values(ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(prov)))
        prov.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
        ax.set_xlabel('Mean Pass Rate (%)')
        ax.set_title('Average Pass Rate by Province', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(prov.values):
            ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
        plt.tight_layout()
        path = WEBAPP_DIR / "static" / "img" / "province_chart.png"
        fig.savefig(path, dpi=120)
        plt.close()
        return "/static/img/province_chart.png"

    elif chart_type == "trend":
        fig, ax = plt.subplots(figsize=(10, 5))
        yearly = df.groupby('Year')['Pass_Rate_Pct'].mean()
        ax.plot(yearly.index, yearly.values, 'o-', color='#0d6efd', linewidth=2.5, markersize=8)
        ax.fill_between(yearly.index, yearly.values, alpha=0.1, color='#0d6efd')
        ax.set_xlabel('Year')
        ax.set_ylabel('Pass Rate (%)')
        ax.set_title('National Pass Rate Trend', fontweight='bold')
        ax.set_xticks(range(2015, 2025))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = WEBAPP_DIR / "static" / "img" / "trend_chart.png"
        fig.savefig(path, dpi=120)
        plt.close()
        return "/static/img/trend_chart.png"

    elif chart_type == "rural_urban":
        fig, ax = plt.subplots(figsize=(8, 5))
        cats = ['Rural', 'Semi-Urban', 'Urban']
        means = [df[df['Rural_Urban']==c]['Pass_Rate_Pct'].mean() for c in cats]
        colors = ['#dc3545', '#ffc107', '#0d6efd']
        bars = ax.bar(cats, means, color=colors, edgecolor='white', width=0.6)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f'{m:.1f}%', ha='center', fontweight='bold')
        ax.set_ylabel('Mean Pass Rate (%)')
        ax.set_title('Urban vs Rural Performance', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        path = WEBAPP_DIR / "static" / "img" / "rural_urban_chart.png"
        fig.savefig(path, dpi=120)
        plt.close()
        return "/static/img/rural_urban_chart.png"

    return None


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=303)
    return RedirectResponse(url="/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {
        "error": None
    })


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, email: str = Form(...), password: str = Form(...)):
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    user_data = USERS.get(email)
    if user_data and user_data["password"] == pwd_hash:
        request.session["user"] = {
            "email": email,
            "name": user_data["name"],
            "role": user_data["role"]
        }
        return RedirectResponse(url="/dashboard", status_code=303)

    return templates.TemplateResponse(request, "login.html", {
        "error": "Invalid email or password. Please try again."
    })


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    df = load_raw_data()
    model, scaler, features, best_name, results = load_best_model()

    # Generate charts
    province_chart = generate_chart("province_bar")
    trend_chart = generate_chart("trend")
    rural_chart = generate_chart("rural_urban")

    # Stats
    stats = {}
    if df is not None:
        stats = {
            "total_records": len(df),
            "districts": df['District'].nunique(),
            "provinces": df['Province'].nunique(),
            "year_range": f"{df['Year'].min()}-{df['Year'].max()}",
            "mean_pass_rate": round(df['Pass_Rate_Pct'].mean(), 1),
            "urban_rate": round(df[df['Rural_Urban']=='Urban']['Pass_Rate_Pct'].mean(), 1),
            "rural_rate": round(df[df['Rural_Urban']=='Rural']['Pass_Rate_Pct'].mean(), 1),
            "latest_year": int(df['Year'].max()),
        }

    model_stats = {}
    if results is not None:
        best = results.iloc[0]
        model_stats = {
            "best_model": best['Model'],
            "r2": round(best['R2'], 4),
            "rmse": round(best['RMSE'], 3),
            "mae": round(best['MAE'], 3),
        }

    # Feature importance
    imp_path = TABLES_DIR / "feature_importance.csv"
    top_features = []
    if imp_path.exists():
        imp_df = pd.read_csv(imp_path).head(5)
        top_features = imp_df.to_dict('records')

    return templates.TemplateResponse(request, "dashboard.html", {
        "user": user, "stats": stats,
        "model_stats": model_stats, "top_features": top_features,
        "province_chart": province_chart, "trend_chart": trend_chart,
        "rural_chart": rural_chart, "page": "dashboard"
    })


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(request, "upload.html", {
        "user": user, "page": "upload",
        "message": None, "error": None, "preview": None, "stats": None
    })


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request,
                      files: list[UploadFile] = File(...),
                      file_type: str = Form(...),
                      remove_duplicates: str = Form(None)):
    """
    Multi-file upload with automatic ingestion.

    Saves each uploaded file to data/uploads/ then runs the multi-glob
    ingestion across master + uploads to give a unified preview. The
    dashboard/analysis routes use the same ingestion via load_raw_data(),
    so files dropped here appear everywhere automatically.
    """
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    error = None
    message = None
    preview = None
    file_stats = None

    # Per-file validation and save
    saved_files = []          # list of (original_name, saved_path, size_bytes)
    rejected = []             # list of (original_name, reason)
    total_size_kb = 0

    for f in files:
        ext = f.filename.split(".")[-1].lower() if "." in f.filename else ""
        if file_type == "csv" and ext != "csv":
            rejected.append((f.filename, f"expected .csv, got .{ext}"))
            continue
        if file_type == "json" and ext != "json":
            rejected.append((f.filename, f"expected .json, got .{ext}"))
            continue
        try:
            contents = await f.read()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = UPLOAD_DIR / f"{ts}_{f.filename}"
            with open(save_path, "wb") as out:
                out.write(contents)
            saved_files.append((f.filename, save_path, len(contents)))
            total_size_kb += len(contents) / 1024
        except Exception as e:
            rejected.append((f.filename, f"save error: {e}"))

    if not saved_files:
        error = "No valid files were uploaded. " + "; ".join(
            f"{n}: {r}" for n, r in rejected)
        return templates.TemplateResponse(request, "upload.html", {
        "user": user, "page": "upload",
            "message": None, "error": error, "preview": None, "stats": None
    })

    # === CSV path: multi-glob to assemble combined dataset ===
    if file_type == "csv":
        try:
            patterns = build_default_patterns(DATA_DIR)
            df = ingest_multi_glob(
                patterns=patterns,
                required_cols=REQUIRED_COLS,
                dedup_keys=DEDUP_KEYS if remove_duplicates else None,
                exclude_files=["cleaned_data.csv"],
                verbose=False,
            )

            # How many rows came from the just-uploaded files?
            new_filenames = {sp.name for _, sp, _ in saved_files}
            new_rows = int(df["_source_file"].isin(new_filenames).sum())

            preview_df = df.drop(columns=["_source_file"]).tail(10)
            preview = preview_df.to_html(
                classes="table table-sm table-striped table-hover",
                index=False, border=0,
            )

            file_stats = {
                "filename": (saved_files[0][0]
                             if len(saved_files) == 1
                             else f"{len(saved_files)} files combined"),
                "records": len(df),
                "columns": df.shape[1] - 1,
                "duplicates_removed": 0,
                "null_columns": int((df.isnull().sum() > 0).sum()),
                "total_nulls": int(df.isnull().sum().sum()),
                "size_kb": round(total_size_kb, 1),
            }

            message = (f"Uploaded {len(saved_files)} file(s) — "
                       f"{new_rows:,} new row(s) added. "
                       f"Combined dataset now has {len(df):,} records.")

            for name, _, _ in saved_files:
                history_store.append({
                    "id": str(uuid.uuid4())[:8],
                    "filename": name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "records": new_rows if len(saved_files) == 1 else "—",
                    "columns": df.shape[1] - 1,
                    "status": "Uploaded & merged",
                })
        except (FileNotFoundError, ValueError) as e:
            error = f"Ingestion error after upload: {e}"
        except Exception as e:
            error = f"Error processing files: {e}"

    # === JSON path: legacy single-file handling ===
    else:
        try:
            _, save_path, _ = saved_files[0]
            df = pd.read_json(save_path)
            if remove_duplicates:
                df = df.drop_duplicates()
            preview = df.head(10).to_html(
                classes="table table-sm table-striped table-hover",
                index=False, border=0,
            )
            file_stats = {
                "filename": saved_files[0][0],
                "records": len(df),
                "columns": len(df.columns),
                "duplicates_removed": 0,
                "null_columns": int((df.isnull().sum() > 0).sum()),
                "total_nulls": int(df.isnull().sum().sum()),
                "size_kb": round(total_size_kb, 1),
            }
            message = f"JSON file '{saved_files[0][0]}' loaded ({len(df):,} records)."
            history_store.append({
                "id": str(uuid.uuid4())[:8],
                "filename": saved_files[0][0],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "records": len(df),
                "columns": len(df.columns),
                "status": "Uploaded (JSON)",
            })
        except Exception as e:
            error = f"Error processing JSON: {e}"

    if rejected:
        rmsg = "; ".join(f"{n}: {r}" for n, r in rejected)
        if error:
            error += f" | Rejected: {rmsg}"
        else:
            message = (message or "") + f" Rejected: {rmsg}"

    return templates.TemplateResponse(request, "upload.html", {
        "user": user, "page": "upload",
        "message": message, "error": error, "preview": preview,
        "stats": file_stats
    })



@app.get("/analysis", response_class=HTMLResponse)
async def analysis_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    model_exists = (MODELS_DIR / "xgboost.pkl").exists()
    return templates.TemplateResponse(request, "analysis.html", {
        "user": user, "page": "analysis",
        "model_exists": model_exists, "result": None
    })


@app.post("/analysis", response_class=HTMLResponse)
async def run_analysis(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    start = time.time()
    result = {"success": False}

    try:
        sys.path.insert(0, str(BASE_DIR / "src"))
        from data_preparation import run_data_preparation
        from eda import run_eda

        run_data_preparation()
        run_eda()

        elapsed = round(time.time() - start, 1)
        result = {
            "success": True,
            "time": elapsed,
            "message": f"Analysis completed successfully in {elapsed} seconds!"
        }

        history_store.append({
            "id": str(uuid.uuid4())[:8],
            "filename": "Full Pipeline",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "records": 1420,
            "columns": 21,
            "status": f"Completed ({elapsed}s)"
        })
    except Exception as e:
        result = {"success": False, "message": f"Error: {str(e)}"}

    return templates.TemplateResponse(request, "analysis.html", {
        "user": user, "page": "analysis",
        "model_exists": True, "result": result
    })


@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    # Check for figures
    figures = {}
    for fname in ['correlation_heatmap', 'pass_rate_trends', 'urban_rural_comparison',
                   'resource_vs_passrate', 'socioeconomic_impact', 'district_trends',
                   'shap_summary', 'shap_bar', 'feature_importance', 'model_comparison']:
        fpath = FIGURES_DIR / f"{fname}.png"
        if fpath.exists():
            figures[fname] = f"/output/figures/{fname}.png"

    return templates.TemplateResponse(request, "reports.html", {
        "user": user, "page": "reports", "figures": figures
    })


@app.post("/reports/simulate", response_class=HTMLResponse)
async def simulate(request: Request, ict_change: float = Form(0),
                   poverty_change: float = Form(0), ptr_change: float = Form(0)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    sim_result = None

    try:
        # Use the DRIVER model — no Previous_Year_Pass_Rate, so feature changes
        # have a real effect rather than being washed out by the autoregressive feature
        driver_model    = joblib.load(MODELS_DIR / "driver_random_forest.pkl")
        driver_scaler   = joblib.load(MODELS_DIR / "driver_scaler.pkl")
        driver_features = joblib.load(MODELS_DIR / "driver_feature_columns.pkl")

        # Baseline = national mean of the latest year (more representative than a single district)
        df = pd.read_csv(DATA_DIR / "cleaned_data.csv")
        latest_year = df["Year"].max()
        baseline_row = df[df["Year"] == latest_year][driver_features].mean()

        # Build X_base and X_scenario as DataFrames
        X_base     = pd.DataFrame([baseline_row[driver_features].values], columns=driver_features)
        X_scenario = X_base.copy()

        # Apply percent-change adjustments
        if "ICT_Resource_Index" in X_scenario.columns:
            X_scenario["ICT_Resource_Index"] *= (1 + ict_change / 100)
        if "Poverty_Incidence_Pct" in X_scenario.columns:
            X_scenario["Poverty_Incidence_Pct"] *= (1 + poverty_change / 100)
        if "Pupil_Teacher_Ratio" in X_scenario.columns:
            X_scenario["Pupil_Teacher_Ratio"] *= (1 + ptr_change / 100)

        pred_base     = float(driver_model.predict(driver_scaler.transform(X_base))[0])
        pred_scenario = float(driver_model.predict(driver_scaler.transform(X_scenario))[0])

        sim_result = {
            "baseline":       round(pred_base, 2),
            "scenario":       round(pred_scenario, 2),
            "delta":          round(pred_scenario - pred_base, 2),
            "ict_change":     ict_change,
            "poverty_change": poverty_change,
            "ptr_change":     ptr_change,
        }
    except FileNotFoundError as e:
        sim_result = {"error": f"Driver model not loaded — run the notebook to train it. ({e})"}
    except Exception as e:
        sim_result = {"error": str(e)}

    figures = {}
    for fname in ['correlation_heatmap', 'pass_rate_trends', 'urban_rural_comparison',
                   'resource_vs_passrate', 'socioeconomic_impact', 'district_trends',
                   'shap_summary', 'shap_bar', 'feature_importance', 'model_comparison']:
        fpath = FIGURES_DIR / f"{fname}.png"
        if fpath.exists():
            figures[fname] = f"/output/figures/{fname}.png"

    return templates.TemplateResponse(request, "reports.html", {
        "user": user, "page": "reports",
        "figures": figures, "sim_result": sim_result
    })


@app.get("/metrics", response_class=HTMLResponse)
async def metrics_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    model_results = load_model_results()
    df = load_raw_data()

    metrics = {}
    if model_results is not None:
        best = model_results.iloc[0]
        metrics = {
            "r2": round(best['R2'], 4),
            "adj_r2": round(best['Adjusted_R2'], 4),
            "rmse": round(best['RMSE'], 3),
            "mae": round(best['MAE'], 3),
            "cv_mean": round(best['CV_R2_Mean'], 4),
            "cv_std": round(best['CV_R2_Std'], 4),
            "best_model": best['Model']
        }

    data_quality = {}
    if df is not None:
        data_quality = {
            "completeness": round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2),
            "duplicate_rate": round(df.duplicated().sum() / len(df) * 100, 2),
            "total_records": len(df),
            "total_features": len(df.columns)
        }

    comparison_table = None
    if model_results is not None:
        comparison_table = model_results.round(4).to_html(
            classes="table table-sm table-striped table-hover", index=False, border=0)

    # Feature importance
    imp_path = TABLES_DIR / "feature_importance.csv"
    feature_chart = None
    if imp_path.exists():
        feature_chart = "/output/figures/feature_importance.png" if (FIGURES_DIR / "feature_importance.png").exists() else None

    return templates.TemplateResponse(request, "metrics.html", {
        "user": user, "page": "metrics",
        "metrics": metrics, "data_quality": data_quality,
        "comparison_table": comparison_table, "feature_chart": feature_chart
    })


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse(request, "history.html", {
        "user": user, "page": "history",
        "history": list(reversed(history_store))
    })


@app.get("/compliance", response_class=HTMLResponse)
async def compliance_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse(request, "compliance.html", {
        "user": user, "page": "compliance"
    })


@app.get("/download/{filename}")
async def download_file(filename: str, request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    path = DATA_DIR / filename
    if path.exists():
        return FileResponse(path, filename=filename)
    raise HTTPException(status_code=404, detail="File not found")


# ═══════════════════════════════════════════════════════════════════════════
# FORECASTING ROUTES
# ═══════════════════════════════════════════════════════════════════════════

def _load_forecasting_module():
    """Lazily import the forecasting module."""
    sys.path.insert(0, str(BASE_DIR / "src"))
    from forecasting import (
        load_dataset, get_provinces_and_districts, run_full_forecast
    )
    return load_dataset, get_provinces_and_districts, run_full_forecast


@app.get("/forecast", response_class=HTMLResponse)
async def forecast_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    load_dataset, get_provinces_and_districts, _ = _load_forecasting_module()
    df = load_dataset()

    provinces = []
    provinces_districts = {}
    if df is not None:
        provinces_districts = get_provinces_and_districts(df)
        provinces = sorted(provinces_districts.keys())

    return templates.TemplateResponse(request, "forecast.html", {
        "user": user, "page": "forecast",
        "provinces": provinces,
        "provinces_districts": provinces_districts,
        "districts_for_province": [],
        "selected_province": None,
        "selected_district": None,
        "selected_start_year": 2025,
        "scenario_values": {},
        "result": None,
        "error": None
    })


@app.post("/forecast", response_class=HTMLResponse)
async def run_forecast(request: Request,
                       province: str = Form(...),
                       district: str = Form(...),
                       start_year: int = Form(2025),
                       ptr_change: float = Form(0),
                       ict_change: float = Form(0),
                       infra_change: float = Form(0),
                       poverty_change: float = Form(0),
                       income_change: float = Form(0)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    load_dataset, get_provinces_and_districts, run_full_forecast = _load_forecasting_module()
    df = load_dataset()

    provinces = []
    provinces_districts = {}
    districts_for_province = []
    if df is not None:
        provinces_districts = get_provinces_and_districts(df)
        provinces = sorted(provinces_districts.keys())
        districts_for_province = provinces_districts.get(province, [])

    # Build scenario adjustments
    scenario = {
        'Pupil_Teacher_Ratio': ptr_change,
        'ICT_Resource_Index': ict_change,
        'School_Infrastructure_Index': infra_change,
        'Poverty_Incidence_Pct': poverty_change,
        'Avg_Household_Income_USD': income_change,
    }

    scenario_values = {
        'ptr_change': ptr_change,
        'ict_change': ict_change,
        'infra_change': infra_change,
        'poverty_change': poverty_change,
        'income_change': income_change,
    }

    result = None
    error = None

    try:
        result = run_full_forecast(
            province=province,
            district=district,
            start_year=start_year,
            horizon=5,
            scenario_adjustments=scenario
        )

        if 'error' in result:
            error = result['error']
            result = None

        # Add to history
        history_store.append({
            "id": str(uuid.uuid4())[:8],
            "filename": f"Forecast: {district}, {province}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "records": 5,
            "columns": 6,
            "status": "Forecast Generated"
        })

    except Exception as e:
        error = f"Forecasting error: {str(e)}"

    return templates.TemplateResponse(request, "forecast.html", {
        "user": user, "page": "forecast",
        "provinces": provinces,
        "provinces_districts": provinces_districts,
        "districts_for_province": districts_for_province,
        "selected_province": province,
        "selected_district": district,
        "selected_start_year": start_year,
        "scenario_values": scenario_values,
        "result": result,
        "error": error
    })


@app.get("/api/districts/{province}")
async def get_districts(province: str, request: Request):
    """JSON endpoint: return districts for a given province."""
    load_dataset, get_provinces_and_districts, _ = _load_forecasting_module()
    df = load_dataset()
    if df is None:
        return {"districts": []}
    mapping = get_provinces_and_districts(df)
    return {"districts": mapping.get(province, [])}


@app.post("/forecast/export-csv")
async def export_forecast_csv(request: Request,
                              province: str = Form(""),
                              district: str = Form(""),
                              start_year: int = Form(2025),
                              forecast_data: str = Form("[]")):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    import io
    import csv

    data = json.loads(forecast_data)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Year', 'Predicted_Pass_Rate_%', 'Lower_CI_%',
                     'Upper_CI_%', 'Change_from_Previous_%', 'Change_from_Baseline_%'])
    for row in data:
        writer.writerow([
            row.get('Year', ''),
            row.get('Predicted_Pass_Rate', ''),
            row.get('Lower_CI', ''),
            row.get('Upper_CI', ''),
            row.get('Change_Pct', ''),
            row.get('Change_From_Baseline', '')
        ])

    csv_content = output.getvalue()
    filename = f"forecast_{district}_{province}_{start_year}.csv"

    # Save to temp
    tmp_path = UPLOAD_DIR / filename
    with open(tmp_path, 'w', newline='') as f:
        f.write(csv_content)

    return FileResponse(tmp_path, filename=filename,
                       media_type='text/csv')


@app.post("/forecast/export-pdf")
async def export_forecast_pdf(request: Request,
                              province: str = Form(""),
                              district: str = Form(""),
                              start_year: int = Form(2025),
                              forecast_data: str = Form("[]"),
                              narrative: str = Form("")):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    data = json.loads(forecast_data)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        filename = f"forecast_report_{district}_{province}_{start_year}.pdf"
        tmp_path = UPLOAD_DIR / filename

        doc = SimpleDocTemplate(str(tmp_path), pagesize=A4,
                               topMargin=0.75*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('Title2', parent=styles['Title'],
                                     fontSize=16, spaceAfter=6)
        subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
                                        fontSize=11, textColor=colors.grey,
                                        spaceAfter=20)
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'],
                                       fontSize=13, spaceAfter=10,
                                       spaceBefore=16)
        body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                    fontSize=10, leading=14, spaceAfter=10)

        elements = []

        # Title
        elements.append(Paragraph(
            "District-Level Academic Performance Forecast", title_style))
        elements.append(Paragraph(
            f"Policy Briefing Report — {district}, {province} Province", subtitle_style))
        elements.append(Spacer(1, 10))

        # Summary
        elements.append(Paragraph("Forecast Summary", heading_style))
        end_year = start_year + len(data) - 1 if data else start_year
        elements.append(Paragraph(
            f"<b>District:</b> {district}<br/>"
            f"<b>Province:</b> {province}<br/>"
            f"<b>Forecast Period:</b> {start_year} – {end_year}<br/>"
            f"<b>Projected Pass Rate ({end_year}):</b> "
            f"{data[-1]['Predicted_Pass_Rate']}%" if data else "",
            body_style))

        # Narrative
        if narrative:
            elements.append(Paragraph("Analysis & Insight", heading_style))
            elements.append(Paragraph(narrative, body_style))

        # Table
        elements.append(Paragraph("Year-by-Year Projections", heading_style))
        table_data = [['Year', 'Pass Rate (%)', 'CI Range', 'Change']]
        for row in data:
            table_data.append([
                str(row.get('Year', '')),
                f"{row.get('Predicted_Pass_Rate', '')}%",
                f"{row.get('Lower_CI', '')}% – {row.get('Upper_CI', '')}%",
                f"{row.get('Change_Pct', '')}%"
            ])

        t = Table(table_data, colWidths=[1.2*inch, 1.5*inch, 2*inch, 1.2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0a2540')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e3e6f0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, colors.HexColor('#f8f9fc')]),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))

        # Footer
        elements.append(Paragraph(
            "<i>Generated by ZIMSEC O-Level Predictive Analytics System<br/>"
            "Academic Research Project — Cesario Machinga</i>",
            ParagraphStyle('Footer', parent=styles['Normal'],
                          fontSize=8, textColor=colors.grey, alignment=1)))

        doc.build(elements)
        return FileResponse(tmp_path, filename=filename,
                           media_type='application/pdf')

    except ImportError:
        # reportlab not installed — generate a text-based fallback
        filename = f"forecast_report_{district}_{province}_{start_year}.txt"
        tmp_path = UPLOAD_DIR / filename
        lines = [
            "DISTRICT-LEVEL ACADEMIC PERFORMANCE FORECAST",
            f"Policy Briefing — {district}, {province}",
            f"Forecast Period: {start_year}–{start_year + 4}",
            "",
            "PROJECTIONS:",
        ]
        for row in data:
            lines.append(
                f"  {row['Year']}: {row['Predicted_Pass_Rate']}% "
                f"(CI: {row['Lower_CI']}–{row['Upper_CI']}%)"
            )
        lines += ["", "INSIGHT:", narrative, "",
                   "Generated by ZIMSEC Analytics System"]
        with open(tmp_path, 'w') as f:
            f.write('\n'.join(lines))
        return FileResponse(tmp_path, filename=filename,
                           media_type='text/plain')


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import webbrowser
    import threading

    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
