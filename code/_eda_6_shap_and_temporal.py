"""
SHAP + temporal-sensitivity EDA for the thesis (no retrain).

Story:
  1) Random split remains the main reported protocol (tables + impurity FI).
  2) SHAP on the *same random protocol* complements FI with attribution.
  3) Temporal split is only a sensitivity check on metrics: fixed earlier non-grid configurations,
     forward holdout — how do the metrics change? No temporal SHAP charts.

Charts:
  - eda_shap_top_binary_xgb_random.png
  - eda_shap_family_binary_xgb_random.png
  - eda_shap_top_reg_xgb_random.png
  - eda_shap_family_reg_xgb_random.png
  - eda_temporal_vs_random_binary_roc.png
  - eda_temporal_vs_random_binary_f1.png
  - eda_temporal_vs_random_reg_r2.png
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eda_support_functions import COLORS  # noqa: E402
from _eda_5_additional_results import family, style_bar  # noqa: E402

warnings.filterwarnings("ignore")
pio.renderers.default = "png"

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "metadata"
SHAP_ROOT = ROOT / "shap"
OUT_CHARTS = ROOT / "thesis_assets/charts"
OUT_TABLES = ROOT / "thesis_assets/tables"
OUT_DOCS = ROOT / "docs"
SNAPSHOT = ROOT / "docs/thesis_results/temporal_vs_random_compare.csv"
SUPPLEMENT = ROOT / "docs/thesis_results/supplementary_results.json"
OUT_MD = OUT_DOCS / "shap_temporal_eda.md"
TEX_IMG = ROOT.parent / "box-office-prediction-thesis/latex/images"
EXPORT_CSV = False

# Random-split SHAP only — sits next to the impurity FI charts.
SHAP_SPECS = (
    (
        "full__binary_classification__no_outliers__complex",
        "xgboost_classifier",
        "Top Features — Full Binary XGBoost SHAP (Random Split)",
        "eda_shap_top_binary_xgb_random.png",
        "SHAP by Feature Family — Full Binary XGBoost (Random Split)",
        "eda_shap_family_binary_xgb_random.png",
    ),
    (
        "full__regression__no_outliers__complex",
        "xgboost_regressor",
        "Top Features — Full Regression XGBoost SHAP (Random Split)",
        "eda_shap_top_reg_xgb_random.png",
        "SHAP by Feature Family — Full Regression XGBoost (Random Split)",
        "eda_shap_family_reg_xgb_random.png",
    ),
)


def ensure_dirs() -> None:
    OUT_CHARTS.mkdir(parents=True, exist_ok=True)
    OUT_DOCS.mkdir(parents=True, exist_ok=True)
    if TEX_IMG.parent.exists():
        TEX_IMG.mkdir(parents=True, exist_ok=True)


def save_summary_csv(frame: pd.DataFrame, filename: str) -> None:
    if EXPORT_CSV:
        OUT_TABLES.mkdir(parents=True, exist_ok=True)
        frame.to_csv(OUT_TABLES / filename, index=False)


def save_chart(fig: go.Figure, filename: str) -> None:
    OUT_CHARTS.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(OUT_CHARTS / filename), scale=3)
    src = OUT_CHARTS / filename
    if TEX_IMG.parent.exists() and src.exists():
        shutil.copy2(src, TEX_IMG / filename)
    print("chart", filename)


def pretty_feature(feat: str) -> str:
    return (
        feat.replace("numerical__", "")
        .replace("binary__", "")
        .replace("categorical__", "")
        .replace("_kpis", " KPIs")
        .replace("__", " ")
        .replace("_", " ")
        .title()
    )


def _bar_h(frame, *, x, y, title, xlabel, ylabel, filename, height=900):
    fig = px.bar(
        frame,
        x=x,
        y=y,
        orientation="h",
        text=x,
        color_discrete_sequence=[COLORS[0]],
    )
    fig.update_traces(
        texttemplate="<b>%{text:.3f}</b>" if x != "share_pct" else "<b>%{text:.1f}%</b>",
        textposition="outside",
        textfont=dict(size=14, color="#000000"),
        marker_line_color="#000000",
        marker_line_width=1.5,
    )
    fig = style_bar(
        fig,
        title,
        xlabel,
        ylabel,
        frame[x].to_numpy(),
        orientation="h",
        left_margin=280,
        height=height,
        width=1200,
    )
    save_chart(fig, filename)


def _grouped_bar(long_df, *, metric, title, ylabel, filename):
    long_df = long_df.copy()
    long_df["model"] = long_df["model"].map(
        lambda name: name.replace("_classifier", "").replace("_regressor", "")
        .replace("_", " ").replace("xgboost", "XGBoost").replace("lightgbm", "LightGBM")
    )
    fig = px.bar(
        long_df,
        x="model",
        y=metric,
        color="split",
        barmode="group",
        text=metric,
        color_discrete_sequence=COLORS[:2],
    )
    fig.update_traces(
        texttemplate="<b>%{text:.3f}</b>",
        textposition="outside",
        textfont=dict(size=14, color="#000000"),
        marker_line_color="#000000",
        marker_line_width=1.5,
    )
    fig = style_bar(
        fig,
        title,
        "Model",
        ylabel,
        long_df[metric].to_numpy(),
        legend=True,
        height=780,
        width=1200,
    )
    fig.update_xaxes(tickangle=-25, title_standoff=12)
    fig.update_layout(
        margin=dict(b=180),
        legend=dict(yref="container", y=0.01, yanchor="bottom"),
    )
    save_chart(fig, filename)


def load_shap_random() -> pd.DataFrame:
    """Read the pinned, tracked SHAP summaries, independent of local run folders."""
    records = json.loads(SUPPLEMENT.read_text())["experiments"]
    rows = []
    for record in records:
        shap = record["random"].get("shap")
        if not shap:
            continue
        frame = pd.DataFrame(shap["mean_abs_shap"])
        frame["stem"] = record["stem"]
        frame["model"] = record["model"]
        frame["feature_label"] = frame["feature"].map(pretty_feature)
        frame["family"] = frame["feature"].map(family)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_shap_random(lines: List[str], shap_df: pd.DataFrame) -> None:
    lines.append("## SHAP (random split — alongside feature importance)\n")
    if shap_df.empty:
        lines.append("No random-split `shap/` artifacts found.\n")
        return

    plotted = 0
    for stem, model, top_title, top_file, fam_title, fam_file in SHAP_SPECS:
        sub = shap_df[(shap_df["stem"] == stem) & (shap_df["model"] == model)]
        if sub.empty:
            continue

        top = sub.sort_values("mean_abs_shap", ascending=True).tail(15)
        _bar_h(
            top,
            x="mean_abs_shap",
            y="feature_label",
            title=top_title,
            xlabel="Mean |SHAP| (log10 revenue)" if "__regression__" in stem else "Mean |SHAP| (log-odds)",
            ylabel="Feature",
            filename=top_file,
        )
        save_summary_csv(
            top.sort_values("mean_abs_shap", ascending=False),
            top_file.replace(".png", ".csv"),
        )
        lines.append(f"### {top_title}\n")
        lines.append(
            top.sort_values("mean_abs_shap", ascending=False)[
                ["feature_label", "mean_abs_shap", "family"]
            ].to_markdown(index=False)
        )
        lines.append("")

        fam = sub.groupby("family", as_index=False)["mean_abs_shap"].sum()
        fam["share_pct"] = 100 * fam["mean_abs_shap"] / fam["mean_abs_shap"].sum()
        fam = fam.sort_values("share_pct", ascending=True).tail(12)
        _bar_h(
            fam,
            x="share_pct",
            y="family",
            title=fam_title,
            xlabel="Share of Total Mean |SHAP| (%)",
            ylabel="Feature Family",
            filename=fam_file,
            height=820,
        )
        lines.append(f"### {fam_title}\n")
        lines.append(
            fam.sort_values("share_pct", ascending=False).to_markdown(index=False)
        )
        lines.append("")
        plotted += 1

    if plotted == 0:
        lines.append("Requested random SHAP specs not found under `shap/`.\n")


def load_compare_table() -> pd.DataFrame:
    if SNAPSHOT.exists():
        return pd.read_csv(SNAPSHOT)
    compare_csvs = sorted(META.glob("_replay_compare__*.csv"))
    if compare_csvs:
        return pd.read_csv(compare_csvs[-1])
    legacy = META / "_compare_nongrid_20260207_vs_temporal.csv"
    return pd.read_csv(legacy) if legacy.exists() else pd.DataFrame()


def _focus(compare: pd.DataFrame) -> pd.DataFrame:
    focus = compare.copy()
    for col, value in (("features", "complex"), ("outliers", "no_outliers")):
        if col in focus.columns:
            focus = focus[focus[col] == value]
    if "model" in focus.columns:
        focus = focus[~focus["model"].astype(str).str.contains("dummy")]
    return focus


def _to_long(plot_df, random_col, temporal_col, metric):
    rows = []
    for _, row in plot_df.iterrows():
        rows.append({"model": row["model"], "split": "random", metric: row[random_col]})
        rows.append(
            {"model": row["model"], "split": "temporal", metric: row[temporal_col]}
        )
    return pd.DataFrame(rows)


def plot_temporal_sensitivity(lines: List[str], compare: pd.DataFrame) -> None:
    lines.append("## Temporal sensitivity (metrics only)\n")
    lines.append(
        "Paired refits hold the reported Full, complex, outlier-removed "
        "configurations fixed and re-score them on a forward temporal holdout. "
        "This is a restricted sensitivity check on a retrospectively "
        "revenue-filtered cohort, not operational forecasting.\n"
    )
    if compare.empty:
        lines.append("No comparison table found.\n")
        return

    focus = _focus(compare)
    binary = (
        focus[focus["task"] == "binary_classification"]
        if "task" in focus.columns
        else focus
    )
    if not binary.empty and {
        "random__ROC AUC Score",
        "temporal__ROC AUC Score",
    }.issubset(binary.columns):
        plot_df = (
            binary[binary["dataset"] == "full"]
            if "dataset" in binary.columns
            else binary
        )
        _grouped_bar(
            _to_long(
                plot_df,
                "random__ROC AUC Score",
                "temporal__ROC AUC Score",
                "ROC AUC",
            ),
            metric="ROC AUC",
            title="Temporal Sensitivity — Full Binary ROC AUC",
            ylabel="ROC AUC",
            filename="eda_temporal_vs_random_binary_roc.png",
        )
        if {"random__F1 Score", "temporal__F1 Score"}.issubset(plot_df.columns):
            _grouped_bar(
                _to_long(plot_df, "random__F1 Score", "temporal__F1 Score", "F1"),
                metric="F1",
                title="Temporal Sensitivity — Full Binary F1",
                ylabel="F1 Score",
                filename="eda_temporal_vs_random_binary_f1.png",
            )
        cols = [
            c
            for c in (
                "model",
                "random__ROC AUC Score",
                "temporal__ROC AUC Score",
                "delta__ROC AUC Score",
                "random__F1 Score",
                "temporal__F1 Score",
                "delta__F1 Score",
            )
            if c in plot_df.columns
        ]
        lines.append("### Full binary complex\n")
        lines.append(plot_df[cols].to_markdown(index=False))
        lines.append("")

    reg = (
        focus[focus["task"] == "regression"]
        if "task" in focus.columns
        else pd.DataFrame()
    )
    if not reg.empty and {"random__R2", "temporal__R2"}.issubset(reg.columns):
        plot_df = reg[reg["dataset"] == "full"] if "dataset" in reg.columns else reg
        _grouped_bar(
            _to_long(plot_df, "random__R2", "temporal__R2", "R2"),
            metric="R2",
            title="Temporal Sensitivity — Full Regression R²",
            ylabel="R²",
            filename="eda_temporal_vs_random_reg_r2.png",
        )
        cols = [
            c
            for c in (
                "model",
                "random__R2",
                "temporal__R2",
                "delta__R2",
                "random__MAPE",
                "temporal__MAPE",
                "delta__MAPE",
            )
            if c in plot_df.columns
        ]
        lines.append("### Full regression complex\n")
        lines.append(plot_df[cols].to_markdown(index=False))
        lines.append("")


def main(export_csv: bool = False) -> None:
    global EXPORT_CSV
    EXPORT_CSV = export_csv
    ensure_dirs()
    lines = [
        "# SHAP and Temporal Sensitivity EDA\n",
        "**Story.** Random split is the main protocol (metrics + impurity FI). "
        "SHAP here is computed on that same random protocol so it can sit next "
        "to the importance charts. Temporal split is only a sensitivity check: "
        "hold the reported Full, complex, outlier-removed configurations fixed and "
        "compare their random holdout with a forward temporal holdout on the "
        "retrospectively filtered cohort.\n",
    ]
    plot_shap_random(lines, load_shap_random())
    plot_temporal_sensitivity(lines, load_compare_table())
    lines.extend(
        [
            "## Notes for thesis\n",
            "- Keep random-split tables as the primary results.",
            "- Use SHAP (random) as complementary attribution beside impurity FI.",
            "- Describe temporal deltas as a restricted sensitivity check, with retrospective filtering disclosed.",
            "- Do not mix temporal SHAP into the explanation narrative.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("wrote", OUT_MD)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-csv", action="store_true")
    args = parser.parse_args()
    main(export_csv=args.export_csv)
