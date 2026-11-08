"""
Additional results EDA for the thesis (no retrain).

Uses the shared Plotly aesthetic from eda_support_functions.py
(Times New Roman, teal palette, bordered white charts).

Writes:
  - thesis_assets/charts/
  - thesis_assets/tables/ only with --export-csv
  - docs/additional_results_eda.md
  - mirrors PNGs into ../box-office-prediction-thesis/latex/images/
"""

from __future__ import annotations

import json
import argparse
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.model_selection import train_test_split

# Allow running as `python code/_eda_5_additional_results.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eda_support_functions import (  # noqa: E402
    COLORS,
    apply_common_style,
    apply_layout_overrides,
    save_figure_safe,
)

warnings.filterwarnings("ignore")
pio.renderers.default = "png"

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "metadata"
ML = ROOT / "data/ml_ready_data"
PROC = ROOT / "data/processed_data"
OUT_CHARTS = ROOT / "thesis_assets/charts"
OUT_TABLES = ROOT / "thesis_assets/tables"
OUT_DOCS = ROOT / "docs"
TEX_IMG = ROOT.parent / "box-office-prediction-thesis/latex/images"

STEP_PREFIX = "eda"
EXPORT_CSV = False


def ensure_dirs() -> None:
    for p in [OUT_CHARTS, OUT_DOCS]:
        p.mkdir(parents=True, exist_ok=True)
    if TEX_IMG.parent.exists():
        TEX_IMG.mkdir(parents=True, exist_ok=True)


def save_summary_csv(frame: pd.DataFrame, filename: str) -> None:
    """Optional review export; figures use the in-memory data directly."""
    if EXPORT_CSV:
        OUT_TABLES.mkdir(parents=True, exist_ok=True)
        frame.to_csv(OUT_TABLES / filename, index=False)


def mirror_to_thesis(filename: str) -> None:
    src = OUT_CHARTS / filename
    if TEX_IMG.parent.exists() and src.exists():
        TEX_IMG.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, TEX_IMG / filename)


def save_chart(fig: go.Figure, filename: str) -> None:
    # Fail visibly if export fails; never report a stale mirrored chart as new.
    OUT_CHARTS.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(OUT_CHARTS / filename), scale=3)
    mirror_to_thesis(filename)
    print("chart", filename)


def style_bar(
    fig: go.Figure,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    y_values: np.ndarray,
    *,
    orientation: str = "v",
    legend: bool = False,
    bottom_margin: int = 120,
    height: int = 780,
    width: int = 1300,
    left_margin: Optional[int] = None,
) -> go.Figure:
    fig = apply_common_style(fig, title, xaxis_title, yaxis_title, list(y_values))
    apply_layout_overrides(
        fig,
        legend=legend,
        bottom_margin=bottom_margin,
        height=height,
        width=width,
    )
    if left_margin is not None:
        fig.update_layout(margin=dict(l=left_margin, r=40, t=130, b=bottom_margin))
    # apply_common_style always sets axis type to linear/log. Restore categorical
    # axes so Plotly bar charts actually render labels/bars.
    if orientation == "h":
        fig.update_yaxes(type="category")
        fig.update_xaxes(type="linear")
    else:
        fig.update_xaxes(type="category")
        fig.update_yaxes(type="linear")
    # Reserve space inside the axes for outside labels, including negative bars.
    values = np.asarray(y_values, dtype=float)
    values = values[np.isfinite(values)]
    if fig.layout.barmode == "stack":
        totals = {}
        for trace in fig.data:
            for category, value in zip(trace.x, trace.y):
                totals[category] = totals.get(category, 0.0) + float(value)
        values = np.asarray(list(totals.values()), dtype=float)
    if len(values):
        low, high = min(0.0, float(values.min())), max(0.0, float(values.max()))
        padding = max(high - low, 1e-9) * 0.18
        bounds = [low - padding if low < 0 else 0, high + padding]
        if orientation == "h":
            fig.update_xaxes(range=bounds)
        else:
            fig.update_yaxes(range=bounds)
    fig.update_traces(cliponaxis=False)
    return fig


def family(feat: str) -> str:
    f = feat
    for pref in ["numerical__", "binary__", "categorical__"]:
        f = f.replace(pref, "")
    if "kpis" in f:
        return f.split("__")[0] if "__" in f else f.split("_kpis")[0] + "_kpis"
    if f.startswith("is_"):
        return f.split("__")[0] if "__" in f else "_".join(f.split("_")[:2])
    return "other/" + f.split("_")[0]


def plot_class_distributions(lines: List[str]) -> pd.DataFrame:
    class_rows = []
    for segment in [
        "full",
        "large_productions",
        "medium_productions",
        "small_productions",
    ]:
        for task, tcol in [
            ("binary_classification", "binary_classification"),
            ("multi_class_classification", "multi_class_classification"),
        ]:
            path = ML / f"{segment}__{task}__with_outliers__none.csv"
            df = pd.read_csv(path, usecols=[tcol])
            vc = df[tcol].value_counts(dropna=False)
            total = len(df)
            for label, cnt in vc.items():
                class_rows.append(
                    {
                        "dataset": segment,
                        "task": task,
                        "class": str(label),
                        "count": int(cnt),
                        "pct": round(100 * cnt / total, 2),
                    }
                )
    class_df = pd.DataFrame(class_rows)
    save_summary_csv(class_df, "table_eda_class_distributions.csv")

    # Two separate thesis-ready charts (cleaner than a subplot with the shared style).
    for task, title, fname in [
        (
            "binary_classification",
            "Full Dataset — Binary Class Share",
            "eda_class_distributions_full_binary.png",
        ),
        (
            "multi_class_classification",
            "Full Dataset — Multi-class Class Share",
            "eda_class_distributions_full_multi.png",
        ),
    ]:
        sub = (
            class_df[(class_df.dataset == "full") & (class_df.task == task)]
            .sort_values("pct", ascending=False)
            .copy()
        )
        fig = px.bar(
            sub,
            x="class",
            y="pct",
            text="pct",
            color_discrete_sequence=[COLORS[0]],
        )
        fig.update_traces(
            texttemplate="<b>%{text:.1f}%</b>",
            textposition="outside",
            marker_line_color="#000000",
            marker_line_width=1.5,
        )
        fig = style_bar(
            fig,
            title,
            "Class",
            "Share (%)",
            sub["pct"].to_numpy(),
            orientation="v",
            legend=False,
        )
        save_chart(fig, fname)

    # Keep a combined filename expected by the current thesis draft.
    # Build a grouped chart with task as color.
    full = class_df[class_df.dataset == "full"].copy()
    full["task_label"] = full["task"].map(
        {
            "binary_classification": "Binary",
            "multi_class_classification": "Multi-class",
        }
    )
    fig = px.bar(
        full,
        x="class",
        y="pct",
        color="task_label",
        barmode="group",
        text="pct",
        color_discrete_sequence=COLORS[:2],
    )
    fig.update_traces(
        texttemplate="<b>%{text:.1f}%</b>",
        textposition="outside",
        marker_line_color="#000000",
        marker_line_width=1.5,
    )
    fig = style_bar(
        fig,
        "Full Dataset — Class Shares by Task",
        "Class",
        "Share (%)",
        full["pct"].to_numpy(),
        orientation="v",
        legend=True,
        bottom_margin=160,
    )
    save_chart(fig, "eda_class_distributions_full.png")

    lines.append("\n## Class balance (Full, outliers kept)\n")
    lines.append(class_df[class_df.dataset == "full"].to_markdown(index=False))
    lines.append("\n")
    return class_df


def plot_temporal_split(lines: List[str]) -> None:
    path = ML / "full__binary_classification__with_outliers__complex.csv"
    df = pd.read_csv(path, usecols=["year", "binary_classification"])
    y = df["binary_classification"]
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    train_years = df.iloc[tr]["year"]
    test_years = df.iloc[te]["year"]
    years = sorted(df["year"].dropna().unique())
    train_counts = train_years.value_counts().reindex(years, fill_value=0)
    test_counts = test_years.value_counts().reindex(years, fill_value=0)

    plot_df = pd.DataFrame(
        {
            "year": list(years) + list(years),
            "count": list(train_counts.values) + list(test_counts.values),
            "split": ["Train (80%)"] * len(years) + ["Test (20%)"] * len(years),
        }
    )
    fig = px.bar(
        plot_df,
        x="year",
        y="count",
        color="split",
        barmode="stack",
        color_discrete_sequence=[COLORS[0], COLORS[1]],
    )
    fig.update_traces(marker_line_color="#000000", marker_line_width=0.4)
    fig = style_bar(
        fig,
        "Random Stratified Split Mixes Release Eras",
        "Release Year",
        "Number of Films",
        plot_df["count"].to_numpy(),
        orientation="v",
        legend=True,
        bottom_margin=180,
        height=820,
    )
    # Year is numeric; keep a linear x-axis for a continuous timeline look.
    fig.update_xaxes(type="linear")
    save_chart(fig, "eda_temporal_train_test_year_overlap.png")

    pct_test_with_later_train = (
        100 * sum((train_years > df.iloc[i]["year"]).any() for i in te) / len(te)
    )
    lines.append("## Temporal structure of the MOTR random split\n")
    lines.append(
        f"- Train years: {int(train_years.min())}–{int(train_years.max())} "
        f"(median {train_years.median():.0f})\n"
    )
    lines.append(
        f"- Test years: {int(test_years.min())}–{int(test_years.max())} "
        f"(median {test_years.median():.0f})\n"
    )
    lines.append(
        "- Share of test films for which the training set contains *later* release years: "
        f"**{pct_test_with_later_train:.1f}%**\n"
    )
    lines.append(
        "- Interpretation: random K-fold / holdout evaluates in-distribution performance, "
        "not strict chronological forecasting.\n"
    )


def plot_small_outlier_reasons(lines: List[str]) -> None:
    movie = pd.read_csv(PROC / "movie.csv", low_memory=False)
    small = pd.read_csv(
        ML / "small_productions__regression__with_outliers__none.csv",
        usecols=["movie_id", "is_outlier"],
    )
    m = small.merge(
        movie[
            [
                "movie_id",
                "budget_usd_adj",
                "revenue_usd_adj",
                "release_category",
                "ratio_adj",
            ]
        ],
        on="movie_id",
        how="left",
    )
    scope_all = movie[
        (movie.budget_usd_adj > 25000)
        & (movie.revenue_usd_adj > 25000)
        & (movie.release_category == "Far streaming release")
    ]
    lo, hi = scope_all["ratio_adj"].quantile(0.05), scope_all["ratio_adj"].quantile(0.90)
    out = m[m.is_outlier].copy()

    def reason(r):
        in_scope = (
            (r.budget_usd_adj > 25000)
            and (r.revenue_usd_adj > 25000)
            and (r.release_category == "Far streaming release")
        )
        if not in_scope:
            return (
                f"Out of scope: {r.release_category}"
                if pd.notna(r.release_category)
                else "Out of scope"
            )
        if r.ratio_adj < lo:
            return "In-scope: ratio < P05"
        if r.ratio_adj > hi:
            return "In-scope: ratio > P90"
        return "Other"

    out["reason"] = out.apply(reason, axis=1)
    reason_df = (
        out["reason"]
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="count")
        .sort_values("count", ascending=True)
    )
    reason_df["pct"] = (100 * reason_df["count"] / reason_df["count"].sum()).round(1)
    save_summary_csv(reason_df, "table_eda_small_outlier_reasons.csv")

    fig = px.bar(
        reason_df,
        x="count",
        y="reason",
        orientation="h",
        text="count",
        color_discrete_sequence=[COLORS[0]],
    )
    fig.update_traces(
        texttemplate="<b>%{text}</b>",
        textposition="outside",
        marker_line_color="#000000",
        marker_line_width=1.5,
    )
    fig = style_bar(
        fig,
        "Why Small Productions Are Removed by Filtering",
        "Removed Films",
        "Reason",
        reason_df["count"].to_numpy(),
        orientation="h",
        legend=False,
        left_margin=320,
        bottom_margin=120,
        height=820,
    )
    save_chart(fig, "eda_small_outlier_reasons.png")

    lines.append("\n## Small-production outlier reasons\n")
    lines.append(reason_df.sort_values("count", ascending=False).to_markdown(index=False))
    lines.append("\n")


def plot_feature_importances_and_families(lines: List[str]) -> None:
    configs = [
        (
            "full__binary_classification__with_outliers__complex__xgboost_classifier__20260213_084518.json",
            "binary_xgb",
            "Top Features — Full Binary XGBoost (Complex, Outliers Kept)",
            "step_4_feature_importance_full_binary_xgb_complex_with_outliers.png",
        ),
        (
            "full__regression__no_outliers__complex__xgboost_regressor__20260213_084518.json",
            "reg_xgb",
            "Top Features — Full Regression XGBoost (Complex, Outliers Removed)",
            "step_4_feature_importance_full_reg_xgb_complex_no_outliers.png",
        ),
        (
            "full__multi_class_classification__with_outliers__complex__xgboost_classifier__20260213_084518.json",
            "multi_xgb",
            "Top Features — Full Multi-class XGBoost (Complex, Outliers Kept)",
            "step_4_feature_importance_full_multi_xgb_complex_with_outliers.png",
        ),
    ]

    fi_rows = []
    for meta_name, label, title, out_name in configs:
        path = META / meta_name
        if not path.exists():
            print("missing", meta_name)
            continue
        data = json.loads(path.read_text())
        fi = data.get("feature_importance") or []
        if not fi:
            print("no FI", meta_name)
            continue

        # Recreate feature-importance chart with categorical Y restored.
        fi_df = pd.DataFrame(fi).copy()
        fi_df["Feature"] = fi_df["Feature"].apply(
            lambda x: (
                x.replace("numerical__", "")
                .replace("binary__", "")
                .replace("categorical__", "")
                .replace("_kpis", " KPIs")
                .replace("_", " ")
                .title()
            )
        )
        fi_df["AbsImportance"] = fi_df["Importance"].abs()
        fi_df = fi_df.sort_values("AbsImportance", ascending=True).tail(15)
        fig = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
            color_discrete_sequence=[COLORS[0]],
        )
        fig.update_traces(
            texttemplate="<b>%{text:.4f}</b>",
            textposition="outside",
            textfont=dict(size=14, color="#000000"),
            marker_line_color="#000000",
            marker_line_width=1.5,
        )
        fig = style_bar(
            fig,
            title,
            "Importance Score",
            "Feature",
            fi_df["Importance"].to_numpy(),
            orientation="h",
            legend=False,
            left_margin=280,
            height=900,
            width=1200,
        )
        save_chart(fig, out_name)

        tmp = pd.DataFrame(fi)
        tmp["family"] = tmp["Feature"].map(family)
        g = tmp.groupby("family", as_index=False)["Importance"].sum()
        g["share"] = g["Importance"] / g["Importance"].sum()
        g["model"] = label
        fi_rows.append(g)

        top = g.sort_values("share", ascending=True).tail(12).copy()
        top["share_pct"] = top["share"] * 100
        fig = px.bar(
            top,
            x="share_pct",
            y="family",
            orientation="h",
            text="share_pct",
            color_discrete_sequence=[COLORS[0]],
        )
        fig.update_traces(
            texttemplate="<b>%{text:.1f}%</b>",
            textposition="outside",
            marker_line_color="#000000",
            marker_line_width=1.5,
        )
        fig = style_bar(
            fig,
            f"Importance by Feature Family — {label}",
            "Share of Total Importance (%)",
            "Feature Family",
            top["share_pct"].to_numpy(),
            orientation="h",
            legend=False,
            left_margin=280,
            height=820,
        )
        fname = f"eda_fi_family_{label}.png"
        save_chart(fig, fname)

    if fi_rows:
        fi_all = pd.concat(fi_rows, ignore_index=True)
        save_summary_csv(fi_all, "table_eda_fi_by_family.csv")
        lines.append("\n## Feature importance by family (share of total)\n")
        for model, sub in fi_all.groupby("model"):
            lines.append(f"\n### {model}\n")
            lines.append(
                sub.sort_values("share", ascending=False).head(10).to_markdown(index=False)
            )
            lines.append("\n")


def plot_release_category_roi(lines: List[str]) -> None:
    """Notebook-inspired finding from _eda_2_data_preprocessing."""
    movie = pd.read_csv(PROC / "movie.csv", low_memory=False)
    df = movie[
        (movie["budget_usd_adj"] > 25000)
        & (movie["revenue_usd_adj"] > 25000)
        & movie["roi"].notna()
        & movie["release_category"].notna()
    ].copy()

    # Clip extreme ROI for readable boxplot while reporting full counts.
    summary = (
        df.groupby("release_category")
        .agg(
            n=("roi", "size"),
            median_roi=("roi", "median"),
            mean_roi=("roi", "mean"),
            p25=("roi", lambda s: s.quantile(0.25)),
            p75=("roi", lambda s: s.quantile(0.75)),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    save_summary_csv(summary, "table_eda_roi_by_release_category.csv")

    fig = px.bar(
        summary.sort_values("median_roi"),
        x="median_roi",
        y="release_category",
        orientation="h",
        text="median_roi",
        color_discrete_sequence=[COLORS[0]],
        hover_data=["n", "mean_roi"],
    )
    fig.update_traces(
        texttemplate="<b>%{text:.2f}</b>",
        textposition="outside",
        marker_line_color="#000000",
        marker_line_width=1.5,
    )
    fig = style_bar(
        fig,
        "Median ROI by Release Category",
        "Median ROI (surplus / budget)",
        "Release Category",
        summary["median_roi"].to_numpy(),
        orientation="h",
        legend=False,
        left_margin=300,
        height=780,
    )
    save_chart(fig, "eda_median_roi_by_release_category.png")

    # Count chart
    fig = px.bar(
        summary.sort_values("n"),
        x="n",
        y="release_category",
        orientation="h",
        text="n",
        color_discrete_sequence=[COLORS[1]],
    )
    fig.update_traces(
        texttemplate="<b>%{text}</b>",
        textposition="outside",
        marker_line_color="#000000",
        marker_line_width=1.5,
    )
    fig = style_bar(
        fig,
        "Film Counts by Release Category",
        "Number of Films (budget & revenue > $25k)",
        "Release Category",
        summary["n"].to_numpy(),
        orientation="h",
        legend=False,
        left_margin=300,
        height=780,
    )
    save_chart(fig, "eda_counts_by_release_category.png")

    lines.append("\n## ROI by release category\n")
    lines.append(summary.to_markdown(index=False))
    lines.append("\n")


def plot_success_rate_by_segment(lines: List[str]) -> None:
    rows = []
    for segment in [
        "small_productions",
        "medium_productions",
        "large_productions",
        "full",
    ]:
        path = ML / f"{segment}__binary_classification__with_outliers__none.csv"
        df = pd.read_csv(path, usecols=["binary_classification"])
        share = (df["binary_classification"] == "Success").mean() * 100
        rows.append(
            {
                "dataset": segment,
                "success_pct": round(share, 2),
                "n": len(df),
            }
        )
    out = pd.DataFrame(rows)
    out["dataset_label"] = out["dataset"].map(
        {
            "small_productions": "Small",
            "medium_productions": "Medium",
            "large_productions": "Large",
            "full": "Full",
        }
    )
    save_summary_csv(out, "table_eda_binary_success_by_segment.csv")

    order = ["Small", "Medium", "Large", "Full"]
    out["dataset_label"] = pd.Categorical(out["dataset_label"], categories=order, ordered=True)
    out = out.sort_values("dataset_label")

    fig = px.bar(
        out,
        x="dataset_label",
        y="success_pct",
        text="success_pct",
        color_discrete_sequence=[COLORS[0]],
    )
    fig.update_traces(
        texttemplate="<b>%{text:.1f}%</b>",
        textposition="outside",
        marker_line_color="#000000",
        marker_line_width=1.5,
    )
    fig = style_bar(
        fig,
        "Binary Success Rate by Budget Segment",
        "Budget Segment",
        "Success Share (%)",
        out["success_pct"].to_numpy(),
        orientation="v",
        legend=False,
    )
    save_chart(fig, "eda_binary_success_by_segment.png")

    lines.append("\n## Binary success rate by segment\n")
    lines.append(out.to_markdown(index=False))
    lines.append("\n")


def plot_success_rate_by_decade(lines: List[str]) -> None:
    path = ML / "full__binary_classification__with_outliers__complex.csv"
    df = pd.read_csv(path, usecols=["year", "binary_classification"])
    df = df.dropna(subset=["year"]).copy()
    df["decade"] = (df["year"] // 10 * 10).astype(int).astype(str) + "s"
    g = (
        df.groupby("decade")
        .agg(
            n=("binary_classification", "size"),
            success_pct=(
                "binary_classification",
                lambda s: 100 * (s == "Success").mean(),
            ),
        )
        .reset_index()
    )
    save_summary_csv(g, "table_eda_binary_success_by_decade.csv")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=g["decade"],
            y=g["success_pct"],
            text=[f"{v:.1f}%" for v in g["success_pct"]],
            textposition="outside",
            marker=dict(color=COLORS[0], line=dict(color="#000000", width=1.5)),
            name="Success share",
        )
    )
    fig = style_bar(
        fig,
        "Binary Success Rate by Decade",
        "Decade",
        "Success Share (%)",
        g["success_pct"].to_numpy(),
        orientation="v",
        legend=False,
    )
    save_chart(fig, "eda_binary_success_by_decade.png")

    lines.append("\n## Binary success rate by decade\n")
    lines.append(g.to_markdown(index=False))
    lines.append("\n")


def main(export_csv: bool = False) -> None:
    global EXPORT_CSV
    EXPORT_CSV = export_csv
    ensure_dirs()
    lines = [
        "# Additional Results EDA\n",
        "Generated from existing artifacts with the shared Plotly thesis aesthetic (no retrain).\n",
        "Unfiltered model scores and importance rankings are provisional: historical models used a revenue-derived outlier flag. Regenerating these figures does not repair those fitted models.\n",
    ]

    plot_class_distributions(lines)
    plot_temporal_split(lines)
    plot_small_outlier_reasons(lines)
    plot_feature_importances_and_families(lines)
    plot_release_category_roi(lines)
    plot_success_rate_by_segment(lines)
    plot_success_rate_by_decade(lines)

    lines.append("\n## Notes for thesis\n")
    lines.append("1. Random split mixes eras almost completely — state as a limitation.\n")
    lines.append("2. Multi-class is highly imbalanced; Small Success is rare (~5%).\n")
    lines.append("3. Small outlier removals are mostly scope, not only ROI extremes.\n")
    lines.append(
        "4. Importance mass concentrates in KPI/market families; sparse one-hots contribute little individually.\n"
    )
    lines.append(
        "5. Release-category ROI gaps motivate keeping Far streaming as the in-scope theatrical window.\n"
    )
    lines.append(
        "6. Success rates vary by budget segment and decade — useful context for external validity.\n"
    )

    out_md = OUT_DOCS / "additional_results_eda.md"
    out_md.write_text("\n".join(lines))
    print("wrote", out_md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-csv", action="store_true", help="Also save optional chart-summary CSVs")
    main(export_csv=parser.parse_args().export_csv)
