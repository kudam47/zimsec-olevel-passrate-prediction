"""
Exploratory Data Analysis module — replaces the original eda.py.

Generates the charts the dashboard, reports, and metrics pages display
from output/figures/. Re-runnable: each call overwrites the existing PNGs.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — required for FastAPI server context
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "output" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


def run_eda():
    """
    Generate the standard EDA chart set:
      - pass_rate_trends.png      (national mean ± 1 SD by year)
      - urban_rural_comparison.png
      - province_chart.png
      - correlation_heatmap.png

    Reads the cleaned dataset from data/cleaned_data.csv. Falls back to
    the master file if cleaned_data.csv hasn't been generated yet.
    """
    cleaned_path = DATA_DIR / "cleaned_data.csv"
    master_path  = DATA_DIR / "zimsec_olevel_district_data.csv"

    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
    elif master_path.exists():
        df = pd.read_csv(master_path)
    else:
        raise FileNotFoundError(
            "No dataset found. Place a CSV in data/ or run data preparation first."
        )

    # 1. National trend
    fig, ax = plt.subplots(figsize=(10, 5))
    yearly = df.groupby("Year")["Pass_Rate_Pct"].agg(["mean", "std"])
    ax.plot(yearly.index, yearly["mean"], "o-",
            color="#1e40af", lw=2.5, ms=9, label="National mean")
    ax.fill_between(yearly.index,
                    yearly["mean"] - yearly["std"],
                    yearly["mean"] + yearly["std"],
                    alpha=0.2, color="#1e40af", label="±1 SD")
    ax.set(title="National O-Level Pass Rate Trend",
           xlabel="Year", ylabel="Pass Rate (%)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "pass_rate_trends.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2. Urban vs Rural
    if "Rural_Urban" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x="Rural_Urban", y="Pass_Rate_Pct", ax=ax,
                    palette=["#dc2626", "#2563eb"])
        ax.set(title="Pass-Rate Distribution: Urban vs Rural",
               ylabel="Pass Rate (%)", xlabel="")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "urban_rural_comparison.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

    # 3. Province
    if "Province" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        prov = df.groupby("Province")["Pass_Rate_Pct"].mean().sort_values()
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(prov)))
        prov.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
        ax.set(title="Average Pass Rate by Province", xlabel="Mean Pass Rate (%)")
        for i, v in enumerate(prov.values):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "province_chart.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

    # 4. Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 2:
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, ax=ax, annot_kws={"size": 7},
                    cbar_kws={"label": "Pearson r"})
        ax.set_title("Correlation Matrix — Numeric Features", fontweight="bold")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "correlation_heatmap.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

    return {
        "rows": len(df),
        "charts_generated": 4,
        "output_dir": str(FIGURES_DIR),
    }


if __name__ == "__main__":
    summary = run_eda()
    print(f"EDA complete: {summary}")
