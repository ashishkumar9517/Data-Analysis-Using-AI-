"""
Coffee Shop Sales Analysis
Author: Ashish Kumar

Analyzes six months of transaction data (Jan-Jun 2023) across three
coffee shop locations to find out what's actually driving revenue growth
and where the business should focus next.

Dataset: Coffee Shop Sales.xlsx (Transactions sheet)
149,116 rows, no cleaning issues found - no nulls, no duplicates, no
negative quantities or prices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

FILE_PATH = "Coffee Shop Sales.xlsx"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Transactions")
    print(f"Loaded {len(df):,} rows, {df['transaction_id'].nunique():,} unique transactions")
    return df


def check_quality(df: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "Rows": len(df),
        "Missing values": int(df.isna().sum().sum()),
        "Duplicate rows": int(df.duplicated().sum()),
        "Duplicate transaction IDs": int(df["transaction_id"].duplicated().sum()),
        "Zero/negative quantity": int((df["transaction_qty"] <= 0).sum()),
        "Zero/negative price": int((df["unit_price"] <= 0).sum()),
    }
    return pd.DataFrame(checks.items(), columns=["check", "result"])


def clean_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["transaction_time"] = pd.to_datetime(
        df["transaction_time"].astype(str), format="mixed", errors="coerce"
    )
    df["transaction_qty"] = pd.to_numeric(df["transaction_qty"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["revenue"] = df["transaction_qty"] * df["unit_price"]
    df["month"] = df["transaction_date"].dt.to_period("M").astype(str)
    df["day_name"] = df["transaction_date"].dt.day_name()
    df["hour"] = df["transaction_time"].dt.hour
    df = df.dropna(subset=["transaction_date", "transaction_qty", "unit_price"])
    return df


def build_summaries(df: pd.DataFrame) -> dict:
    monthly = (
        df.groupby("month")
        .agg(revenue=("revenue", "sum"), units=("transaction_qty", "sum"),
             transactions=("transaction_id", "nunique"))
        .reset_index()
    )
    monthly["avg_transaction_value"] = monthly["revenue"] / monthly["transactions"]
    monthly["mom_revenue_growth_pct"] = monthly["revenue"].pct_change() * 100

    store = (
        df.groupby("store_location")
        .agg(revenue=("revenue", "sum"), units=("transaction_qty", "sum"),
             transactions=("transaction_id", "nunique"), avg_price=("unit_price", "mean"))
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    store["revenue_share_pct"] = store["revenue"] / store["revenue"].sum() * 100

    category = (
        df.groupby("product_category")
        .agg(revenue=("revenue", "sum"), units=("transaction_qty", "sum"),
             transactions=("transaction_id", "nunique"))
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    category["revenue_share_pct"] = category["revenue"] / category["revenue"].sum() * 100

    product = (
        df.groupby(["product_category", "product_detail"])
        .agg(revenue=("revenue", "sum"), units=("transaction_qty", "sum"),
             transactions=("transaction_id", "nunique"))
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    product["revenue_share_pct"] = product["revenue"] / product["revenue"].sum() * 100

    hourly = df.groupby("hour").agg(revenue=("revenue", "sum"),
                                     transactions=("transaction_id", "nunique")).reset_index()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = (
        df.groupby("day_name")
        .agg(revenue=("revenue", "sum"), transactions=("transaction_id", "nunique"))
        .reindex(day_order)
        .reset_index()
    )

    return {
        "monthly": monthly, "store": store, "category": category,
        "product": product, "hourly": hourly, "daily": daily,
    }


def plot_overview(summaries: dict, out_path: str = "overview.png") -> None:
    monthly, category, store, hourly = (
        summaries["monthly"], summaries["category"], summaries["store"], summaries["hourly"]
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.lineplot(data=monthly, x="month", y="revenue", marker="o", ax=axes[0, 0])
    axes[0, 0].set_title("Monthly Revenue Trend")
    axes[0, 0].tick_params(axis="x", rotation=45)

    sns.barplot(data=category, y="product_category", x="revenue", ax=axes[0, 1])
    axes[0, 1].set_title("Revenue by Product Category")
    axes[0, 1].set_ylabel("")

    sns.barplot(data=store, x="store_location", y="revenue", ax=axes[1, 0])
    axes[1, 0].set_title("Revenue by Store")
    axes[1, 0].tick_params(axis="x", rotation=20)

    sns.lineplot(data=hourly, x="hour", y="revenue", marker="o", ax=axes[1, 1])
    axes[1, 1].set_title("Revenue by Hour of Day")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


def export_workbook(df: pd.DataFrame, quality: pd.DataFrame, summaries: dict,
                     out_path: str = "coffee_shop_sales_cleaned_analysis.xlsx") -> None:
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Cleaned Transactions", index=False)
        quality.to_excel(writer, sheet_name="Data Quality", index=False)
        summaries["monthly"].to_excel(writer, sheet_name="Monthly Summary", index=False)
        summaries["store"].to_excel(writer, sheet_name="Store Summary", index=False)
        summaries["category"].to_excel(writer, sheet_name="Category Summary", index=False)
        summaries["product"].to_excel(writer, sheet_name="Product Summary", index=False)
        summaries["hourly"].to_excel(writer, sheet_name="Hourly Summary", index=False)
        summaries["daily"].to_excel(writer, sheet_name="Daily Summary", index=False)
    print(f"Saved workbook: {out_path}")


if __name__ == "__main__":
    raw = load_data(FILE_PATH)
    quality_report = check_quality(raw)
    print(quality_report)

    clean = clean_and_enrich(raw)
    summaries = build_summaries(clean)

    print("\nMonthly summary:\n", summaries["monthly"])
    print("\nStore summary:\n", summaries["store"])
    print("\nCategory summary:\n", summaries["category"])
    print("\nTop 10 products:\n", summaries["product"].head(10))
    print("\nBest hours:\n", summaries["hourly"].sort_values("revenue", ascending=False).head(5))

    plot_overview(summaries)
    export_workbook(clean, quality_report, summaries)
