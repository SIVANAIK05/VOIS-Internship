"""
Seasonal Agriculture Performance Analysis
==========================================

Run:
    python src/analysis.py

The script:
1. Loads and validates the dataset.
2. Cleans missing values and duplicates.
3. Generates descriptive summaries.
4. Performs seasonal, crop, regional, environmental and water-efficiency analysis.
5. Generates visualizations.
6. Writes machine-readable result tables and a text findings report.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "seasonal_agriculture_performance_dataset.csv"
OUTPUT_DIR = ROOT / "visualizations"
REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Required columns
# -----------------------------
REQUIRED_COLUMNS = [
    "Farm_ID",
    "State",
    "District",
    "Crop",
    "Season",
    "Farm_Area_Hectares",
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_pct",
    "Sunlight_Hours_Day",
    "Soil_pH",
    "Soil_Moisture_pct",
    "Nitrogen_kg_ha",
    "Phosphorus_kg_ha",
    "Potassium_kg_ha",
    "Irrigation_Method",
    "Fertilizer_kg_ha",
    "Pesticide_Litre_ha",
    "Seed_Quality_Score",
    "Yield_Tonnes_Ha",
    "Production_Tonnes",
    "Market_Price_INR_Tonne",
    "Total_Cost_INR",
    "Revenue_INR",
    "Profit_INR",
    "Water_Used_m3",
    "Water_Efficiency_t_per_1000m3",
    "Disease_Pest_Risk_pct",
]

NUMERIC_COLUMNS = [
    "Farm_Area_Hectares",
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_pct",
    "Sunlight_Hours_Day",
    "Soil_pH",
    "Soil_Moisture_pct",
    "Nitrogen_kg_ha",
    "Phosphorus_kg_ha",
    "Potassium_kg_ha",
    "Fertilizer_kg_ha",
    "Pesticide_Litre_ha",
    "Seed_Quality_Score",
    "Yield_Tonnes_Ha",
    "Production_Tonnes",
    "Market_Price_INR_Tonne",
    "Total_Cost_INR",
    "Revenue_INR",
    "Profit_INR",
    "Water_Used_m3",
    "Water_Efficiency_t_per_1000m3",
    "Disease_Pest_Risk_pct",
]


def load_data():
    """Load and validate the source CSV."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns:\n"
            + "\n".join(f"- {c}" for c in missing_columns)
        )

    return data


def clean_data(data):
    """Clean duplicates, whitespace and numeric data types."""
    cleaned = data.copy()

    # Remove exact duplicate records.
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before - len(cleaned)

    # Strip whitespace from text columns.
    for col in cleaned.select_dtypes(include="object").columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    # Convert numeric fields safely.
    for col in NUMERIC_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Fill numeric missing values with median.
    # Median is robust to extreme values and keeps the number of observations.
    imputed = {}
    for col in NUMERIC_COLUMNS:
        count = int(cleaned[col].isna().sum())
        if count > 0:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
            imputed[col] = count

    return cleaned, duplicates_removed, imputed


def save_table(table, filename):
    path = OUTPUT_DIR / filename
    table.to_csv(path)
    return path


def plot_bar(series, title, xlabel, ylabel, filename, horizontal=False):
    fig, ax = plt.subplots(figsize=(9, 5))
    if horizontal:
        series.sort_values().plot(kind="barh", ax=ax)
    else:
        series.plot(kind="bar", ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("SEASONAL AGRICULTURE PERFORMANCE ANALYSIS")
    print("=" * 70)

    # 1. Load
    raw = load_data()
    print(f"\nRaw dataset shape: {raw.shape}")

    # 2. Quality checks before cleaning
    missing_before = raw.isna().sum()
    duplicate_count = int(raw.duplicated().sum())

    print("\nMissing values before cleaning:")
    print(missing_before[missing_before > 0])
    print(f"\nDuplicate rows: {duplicate_count}")

    # 3. Clean
    df, duplicates_removed, imputed = clean_data(raw)

    print(f"\nCleaned dataset shape: {df.shape}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Numeric fields imputed: {imputed}")

    # Save cleaned dataset
    df.to_csv(OUTPUT_DIR / "cleaned_agriculture_dataset.csv", index=False)

    # 4. Descriptive statistics
    descriptive = df[NUMERIC_COLUMNS].describe().T
    descriptive.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

    # 5. Seasonal analysis
    seasonal = df.groupby("Season").agg(
        Farms=("Farm_ID", "count"),
        Avg_Yield_Tonnes_Ha=("Yield_Tonnes_Ha", "mean"),
        Total_Production_Tonnes=("Production_Tonnes", "sum"),
        Avg_Production_Tonnes=("Production_Tonnes", "mean"),
        Avg_Profit_INR=("Profit_INR", "mean"),
        Total_Profit_INR=("Profit_INR", "sum"),
        Avg_Water_Used_m3=("Water_Used_m3", "mean"),
        Avg_Water_Efficiency=("Water_Efficiency_t_per_1000m3", "mean"),
        Avg_Disease_Pest_Risk_pct=("Disease_Pest_Risk_pct", "mean"),
        Avg_Rainfall_mm=("Rainfall_mm", "mean"),
    ).sort_values("Avg_Yield_Tonnes_Ha", ascending=False)

    save_table(seasonal, "seasonal_summary.csv")
    plot_bar(
        seasonal["Avg_Yield_Tonnes_Ha"],
        "Average Yield by Season",
        "Season",
        "Yield (tonnes/ha)",
        "seasonal_yield.png",
    )

    # Yield + profit chart
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(seasonal))
    width = 0.36
    ax1.bar(
        x - width / 2,
        seasonal["Avg_Yield_Tonnes_Ha"],
        width,
        label="Average Yield (t/ha)",
    )
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        seasonal["Avg_Profit_INR"] / 1000,
        marker="o",
        linewidth=2,
        label="Average Profit (₹000)",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(seasonal.index)
    ax1.set_ylabel("Yield (tonnes/ha)")
    ax2.set_ylabel("Profit (₹000)")
    ax1.set_title("Seasonal Yield and Profit", fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "seasonal_yield_profit.png", dpi=180)
    plt.close(fig)

    # 6. Crop analysis
    crop = df.groupby("Crop").agg(
        Farms=("Farm_ID", "count"),
        Avg_Yield_Tonnes_Ha=("Yield_Tonnes_Ha", "mean"),
        Avg_Profit_INR=("Profit_INR", "mean"),
        Avg_Water_Efficiency=("Water_Efficiency_t_per_1000m3", "mean"),
        Avg_Water_Used_m3=("Water_Used_m3", "mean"),
    ).sort_values("Avg_Profit_INR", ascending=False)

    save_table(crop, "crop_summary.csv")
    plot_bar(
        crop["Avg_Profit_INR"],
        "Average Profit by Crop",
        "Average Profit (INR)",
        "Crop",
        "crop_profitability.png",
        horizontal=True,
    )

    # 7. State analysis
    state = df.groupby("State").agg(
        Farms=("Farm_ID", "count"),
        Avg_Yield_Tonnes_Ha=("Yield_Tonnes_Ha", "mean"),
        Avg_Profit_INR=("Profit_INR", "mean"),
        Avg_Water_Efficiency=("Water_Efficiency_t_per_1000m3", "mean"),
        Avg_Disease_Pest_Risk_pct=("Disease_Pest_Risk_pct", "mean"),
    ).sort_values("Avg_Yield_Tonnes_Ha", ascending=False)

    save_table(state, "state_summary.csv")
    plot_bar(
        state["Avg_Yield_Tonnes_Ha"],
        "Average Yield by State",
        "Average Yield (tonnes/ha)",
        "State",
        "state_yield.png",
        horizontal=True,
    )

    # 8. Environmental analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(
        data=df,
        x="Rainfall_mm",
        y="Yield_Tonnes_Ha",
        ax=axes[0],
    )
    axes[0].set_title("Rainfall vs Yield", fontweight="bold")

    sns.scatterplot(
        data=df,
        x="Avg_Temperature_C",
        y="Yield_Tonnes_Ha",
        ax=axes[1],
    )
    axes[1].set_title("Temperature vs Yield", fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "environment_vs_yield.png", dpi=180)
    plt.close(fig)

    # 9. Resource analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(
        data=df,
        x="Water_Used_m3",
        y="Production_Tonnes",
        ax=axes[0],
    )
    axes[0].set_title("Water Used vs Production", fontweight="bold")

    sns.scatterplot(
        data=df,
        x="Fertilizer_kg_ha",
        y="Yield_Tonnes_Ha",
        ax=axes[1],
    )
    axes[1].set_title("Fertilizer vs Yield", fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "resource_vs_performance.png", dpi=180)
    plt.close(fig)

    # 10. Water efficiency
    water_season = (
        df.groupby("Season")["Water_Efficiency_t_per_1000m3"]
        .mean()
        .sort_values(ascending=False)
    )
    save_table(water_season.to_frame("Average_Water_Efficiency"), "water_efficiency_summary.csv")
    plot_bar(
        water_season,
        "Water Efficiency by Season",
        "Season",
        "Tonnes per 1,000 m³",
        "water_efficiency.png",
    )

    # 11. Correlation analysis
    numeric = df.select_dtypes(include=np.number)
    corr = numeric.corr()
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(13, 9))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    yield_corr = corr["Yield_Tonnes_Ha"].drop("Yield_Tonnes_Ha").sort_values(ascending=False)
    profit_corr = corr["Profit_INR"].drop("Profit_INR").sort_values(ascending=False)

    # 12. Season consistency by region
    state_season = (
        df.groupby(["State", "Season"])["Yield_Tonnes_Ha"]
        .mean()
        .unstack()
    )
    state_season.to_csv(OUTPUT_DIR / "state_season_yield.csv")

    # 13. Irrigation comparison
    irrigation = df.groupby("Irrigation_Method").agg(
        Avg_Yield_Tonnes_Ha=("Yield_Tonnes_Ha", "mean"),
        Avg_Profit_INR=("Profit_INR", "mean"),
        Avg_Water_Efficiency=("Water_Efficiency_t_per_1000m3", "mean"),
    ).sort_values("Avg_Yield_Tonnes_Ha", ascending=False)
    irrigation.to_csv(OUTPUT_DIR / "irrigation_summary.csv")

    plot_bar(
        irrigation["Avg_Yield_Tonnes_Ha"],
        "Average Yield by Irrigation Method",
        "Irrigation Method",
        "Yield (tonnes/ha)",
        "irrigation_yield.png",
    )

    # 14. Generate findings report
    best_season_yield = seasonal["Avg_Yield_Tonnes_Ha"].idxmax()
    worst_season_yield = seasonal["Avg_Yield_Tonnes_Ha"].idxmin()
    best_season_profit = seasonal["Avg_Profit_INR"].idxmax()
    best_crop_profit = crop["Avg_Profit_INR"].idxmax()
    worst_crop_profit = crop["Avg_Profit_INR"].idxmin()
    best_state_yield = state["Avg_Yield_Tonnes_Ha"].idxmax()
    best_water_season = water_season.idxmax()

    report = f"""SEASONAL AGRICULTURE PERFORMANCE ANALYSIS
===========================================

DATASET
-------
Rows: {len(raw):,}
Columns: {len(raw.columns)}
Rows after cleaning: {len(df):,}
Duplicates removed: {duplicates_removed}

MISSING VALUES IMPUTED
----------------------
{chr(10).join(f"- {k}: {v}" for k, v in imputed.items()) if imputed else "None"}

KEY FINDINGS
------------
1. Highest average yield season: {best_season_yield}
   Average yield: {seasonal.loc[best_season_yield, "Avg_Yield_Tonnes_Ha"]:.2f} tonnes/ha

2. Lowest average yield season: {worst_season_yield}
   Average yield: {seasonal.loc[worst_season_yield, "Avg_Yield_Tonnes_Ha"]:.2f} tonnes/ha

3. Highest average profit season: {best_season_profit}
   Average profit: ₹{seasonal.loc[best_season_profit, "Avg_Profit_INR"]:,.2f}

4. Highest average-profit crop: {best_crop_profit}
   Average profit: ₹{crop.loc[best_crop_profit, "Avg_Profit_INR"]:,.2f}

5. Lowest average-profit crop: {worst_crop_profit}
   Average profit: ₹{crop.loc[worst_crop_profit, "Avg_Profit_INR"]:,.2f}

6. Highest average-yield state: {best_state_yield}
   Average yield: {state.loc[best_state_yield, "Avg_Yield_Tonnes_Ha"]:.2f} tonnes/ha

7. Highest average water-efficiency season: {best_water_season}
   Efficiency: {water_season.loc[best_water_season]:.2f} tonnes per 1,000 m³

STRONGEST OBSERVED ASSOCIATIONS WITH YIELD
------------------------------------------
{yield_corr.head(5).to_string()}

STRONGEST OBSERVED ASSOCIATIONS WITH PROFIT
-------------------------------------------
{profit_corr.head(5).to_string()}

INTERPRETATION NOTE
-------------------
Correlation describes observed association and does not establish causation.

RECOMMENDATIONS
---------------
1. Use seasonal yield and profitability comparisons to support crop planning.
2. Focus water-management improvements on lower-efficiency seasons or crops.
3. Investigate lower-performing regions for soil, irrigation, climate or operational differences.
4. Combine environmental indicators with production and cost information for planning.
5. Extend the dataset with historical weather, soil and market-price data.
6. Build an interactive BI dashboard for decision-makers.
7. Explore predictive models only after establishing a reliable analytical baseline.
"""

    (REPORT_DIR / "key_findings.txt").write_text(report, encoding="utf-8")

    print("\n" + report)
    print("\nAll outputs saved to:")
    print(OUTPUT_DIR)
    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
