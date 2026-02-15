import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from eda_support_functions import (
    COLORS,
    apply_common_style,
    apply_layout_overrides,
    generate_filename,
    normalize_feature_group_label,
    save_analysis_table_bundle,
    save_figure_safe,
)


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 120)

pio.renderers.default = "notebook"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ML_READY_DIR = REPO_ROOT / "data" / "ml_ready_data"

OUTPUT_DIR = REPO_ROOT / "thesis_assets"
TABLES_DIR = OUTPUT_DIR / "tables"
CHARTS_DIR = OUTPUT_DIR / "charts"

STEP_PREFIX = "step_3"
DATASET_ORDER = ["small_productions", "medium_productions", "large_productions", "full"]
PROBLEM_ORDER = ["regression", "binary_classification", "multi_class_classification"]
FEATURE_ORDER = ["none", "complex"]
CONFIG_ORDER = [
    "FE: none<br>With Outliers",
    "FE: complex<br>With Outliers",
    "FE: none<br>No Outliers",
    "FE: complex<br>No Outliers",
]

TARGET_COLUMN_BY_PROBLEM = {
    "regression": "revenue_usd_adj",
    "binary_classification": "binary_classification",
    "multi_class_classification": "multi_class_classification",
}

TRUE_VALUES = {"true", "1", "1.0", "yes", "y", "t"}
FALSE_VALUES = {"false", "0", "0.0", "no", "n", "f"}

DATASET_LABELS = {
    "small_productions": "small",
    "medium_productions": "medium",
    "large_productions": "large",
    "full": "full",
}


def ensure_output_dirs() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)


def save_step3_figure(fig, filename: str) -> None:
    save_figure_safe(fig, filename, output_dir=str(CHARTS_DIR))


def save_step3_table(
    df: pd.DataFrame,
    filename_base: str,
    caption: str,
    latex_label: str,
) -> None:
    save_analysis_table_bundle(
        df,
        filename_base=filename_base,
        caption=caption,
        latex_label=latex_label,
        output_dir=str(TABLES_DIR),
        latex_float_format="%.2f",
    )


def resolve_ml_ready_dir(explicit_path: Optional[str] = None) -> Path:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    candidates.extend(
        [
            Path("data/ml_ready_data"),
            Path("../data/ml_ready_data"),
            DEFAULT_ML_READY_DIR,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched_paths = [str(path) for path in candidates]
    raise FileNotFoundError(
        f"Could not locate ml_ready_data directory. Checked: {searched_paths}"
    )


def parse_dataset_filename(file_path: Path) -> Optional[Dict[str, object]]:
    parts = file_path.stem.split("__")
    if len(parts) != 4:
        print(f"Skipping unrecognized dataset filename: {file_path.name}")
        return None

    dataset_name, problem_type, outlier_tag, feature_engineering = parts
    if outlier_tag not in {"with_outliers", "no_outliers"}:
        print(f"Skipping dataset with unknown outlier tag: {file_path.name}")
        return None

    return {
        "dataset_name": dataset_name,
        "problem_type": problem_type,
        "feature_engineering": feature_engineering,
        "has_outliers_removed": outlier_tag == "no_outliers",
        "file_name": file_path.name,
        "file_path": str(file_path.resolve()),
    }


def to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(TRUE_VALUES)


def is_bool_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True

    unique_values = set(series.dropna().astype(str).str.strip().str.lower().unique())
    if not unique_values:
        return False
    return unique_values.issubset(TRUE_VALUES.union(FALSE_VALUES))


def create_config_label(feature_engineering: str, has_outliers_removed: bool) -> str:
    outlier_label = "No Outliers" if has_outliers_removed else "With Outliers"
    return f"FE: {feature_engineering}<br>{outlier_label}"


def create_outlier_label(has_outliers_removed: bool) -> str:
    return "no_outliers" if has_outliers_removed else "with_outliers"


def load_dataset_inventory(ml_ready_dir: Path) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    for file_path in sorted(ml_ready_dir.glob("*.csv")):
        parsed = parse_dataset_filename(file_path)
        if not parsed:
            continue

        header_df = pd.read_csv(file_path, nrows=0)
        columns = list(header_df.columns)
        target_column = TARGET_COLUMN_BY_PROBLEM.get(parsed["problem_type"])

        read_columns = []
        if "is_outlier" in columns:
            read_columns.append("is_outlier")
        if target_column in columns:
            read_columns.append(target_column)
        if not read_columns:
            read_columns = [columns[0]]

        sampled_df = pd.read_csv(file_path, usecols=read_columns, low_memory=False)
        row_count = int(len(sampled_df))
        outlier_count = (
            int(to_bool_series(sampled_df["is_outlier"]).sum())
            if "is_outlier" in sampled_df.columns
            else 0
        )
        outlier_rate_pct = (100.0 * outlier_count / row_count) if row_count else 0.0

        target_cardinality = np.nan
        majority_class_pct = np.nan
        if target_column in sampled_df.columns:
            target_series = sampled_df[target_column]
            target_cardinality = int(target_series.nunique(dropna=True))
            if parsed["problem_type"] != "regression":
                class_share = target_series.value_counts(normalize=True, dropna=False)
                if not class_share.empty:
                    majority_class_pct = float(class_share.iloc[0] * 100.0)

        records.append(
            {
                **parsed,
                "target_column": target_column,
                "rows": row_count,
                "n_columns": int(len(columns)),
                "outlier_count": outlier_count,
                "outlier_rate_pct": outlier_rate_pct,
                "target_cardinality": target_cardinality,
                "majority_class_pct": majority_class_pct,
                "file_size_mb": round(file_path.stat().st_size / (1024**2), 2),
            }
        )

    inventory = pd.DataFrame(records)
    if inventory.empty:
        return inventory

    inventory["config_label"] = inventory.apply(
        lambda row: create_config_label(
            row["feature_engineering"], row["has_outliers_removed"]
        ),
        axis=1,
    )
    inventory["outlier_strategy"] = inventory["has_outliers_removed"].map(
        create_outlier_label
    )

    return inventory.sort_values(
        [
            "dataset_name",
            "problem_type",
            "has_outliers_removed",
            "feature_engineering",
        ]
    ).reset_index(drop=True)


def build_outlier_impact_table(inventory_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        inventory_df.groupby(
            ["dataset_name", "problem_type", "feature_engineering", "has_outliers_removed"]
        )["rows"]
        .max()
        .reset_index()
    )

    pivot = grouped.pivot_table(
        index=["dataset_name", "problem_type", "feature_engineering"],
        columns="has_outliers_removed",
        values="rows",
        aggfunc="max",
    )

    pivot = pivot.rename(
        columns={
            False: "rows_with_outliers",
            True: "rows_no_outliers",
        }
    ).fillna(0)
    pivot["rows_with_outliers"] = pivot["rows_with_outliers"].astype(int)
    pivot["rows_no_outliers"] = pivot["rows_no_outliers"].astype(int)
    pivot["rows_removed"] = pivot["rows_with_outliers"] - pivot["rows_no_outliers"]
    pivot["rows_removed_pct"] = np.where(
        pivot["rows_with_outliers"] > 0,
        100.0 * pivot["rows_removed"] / pivot["rows_with_outliers"],
        0.0,
    )

    return pivot.reset_index().sort_values(
        ["dataset_name", "problem_type", "feature_engineering"]
    )


def build_feature_engineering_impact_table(inventory_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        inventory_df.groupby(
            [
                "dataset_name",
                "problem_type",
                "has_outliers_removed",
                "feature_engineering",
            ]
        )[["rows", "n_columns"]]
        .max()
        .reset_index()
    )

    pivot = grouped.pivot_table(
        index=["dataset_name", "problem_type", "has_outliers_removed"],
        columns="feature_engineering",
        values=["rows", "n_columns"],
        aggfunc="max",
    )
    pivot.columns = [f"{metric}_{feature}" for metric, feature in pivot.columns]
    pivot = pivot.reset_index()

    if "n_columns_none" not in pivot.columns or "n_columns_complex" not in pivot.columns:
        return pd.DataFrame()

    for col in ["rows_none", "rows_complex", "n_columns_none", "n_columns_complex"]:
        if col in pivot.columns:
            pivot[col] = pivot[col].fillna(0).astype(int)

    pivot["columns_added_by_complex"] = pivot["n_columns_complex"] - pivot["n_columns_none"]
    pivot["columns_added_pct"] = np.where(
        pivot["n_columns_none"] > 0,
        100.0 * pivot["columns_added_by_complex"] / pivot["n_columns_none"],
        np.nan,
    )
    pivot["row_delta_complex_vs_none"] = pivot["rows_complex"] - pivot["rows_none"]

    return pivot.sort_values(["dataset_name", "problem_type", "has_outliers_removed"])


def build_dataset_shape_summary(inventory_df: pd.DataFrame) -> pd.DataFrame:
    base_rows = inventory_df[
        (inventory_df["problem_type"] == "regression")
        & (inventory_df["feature_engineering"] == "none")
    ][["dataset_name", "has_outliers_removed", "rows"]]
    row_pivot = (
        base_rows.pivot_table(
            index="dataset_name",
            columns="has_outliers_removed",
            values="rows",
            aggfunc="max",
        )
        .rename(columns={False: "rows_with_outliers", True: "rows_no_outliers"})
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    row_pivot["rows_removed"] = row_pivot["rows_with_outliers"] - row_pivot["rows_no_outliers"]
    row_pivot["rows_removed_pct"] = np.where(
        row_pivot["rows_with_outliers"] > 0,
        100.0 * row_pivot["rows_removed"] / row_pivot["rows_with_outliers"],
        0.0,
    )

    width_rows = inventory_df[
        (inventory_df["problem_type"] == "regression")
        & (~inventory_df["has_outliers_removed"])
    ][["dataset_name", "feature_engineering", "n_columns"]]
    width_pivot = (
        width_rows.pivot_table(
            index="dataset_name",
            columns="feature_engineering",
            values="n_columns",
            aggfunc="max",
        )
        .rename(columns={"none": "n_columns_none", "complex": "n_columns_complex"})
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    width_pivot["columns_added_by_complex"] = (
        width_pivot["n_columns_complex"] - width_pivot["n_columns_none"]
    )

    coverage = (
        inventory_df.groupby("dataset_name")
        .agg(
            files_generated=("file_name", "count"),
            problem_types=("problem_type", "nunique"),
            outlier_strategies=("has_outliers_removed", "nunique"),
            feature_modes=("feature_engineering", "nunique"),
        )
        .reset_index()
    )

    summary_df = (
        row_pivot.merge(width_pivot, on="dataset_name", how="left")
        .merge(coverage, on="dataset_name", how="left")
        .fillna(0)
    )
    summary_df["dataset_label"] = summary_df["dataset_name"].map(DATASET_LABELS)
    return summary_df.sort_values(
        "dataset_name",
        key=lambda series: series.map({name: idx for idx, name in enumerate(DATASET_ORDER)}),
    )


def build_feature_width_summary(inventory_df: pd.DataFrame) -> pd.DataFrame:
    width_df = (
        inventory_df[
            (inventory_df["problem_type"] == "regression")
            & (~inventory_df["has_outliers_removed"])
        ]
        .groupby("feature_engineering")
        .agg(n_columns=("n_columns", "median"))
        .reset_index()
    )
    width_df["n_columns"] = width_df["n_columns"].astype(int)
    width_df["feature_engineering"] = pd.Categorical(
        width_df["feature_engineering"], categories=FEATURE_ORDER, ordered=True
    )
    return width_df.sort_values("feature_engineering")


def get_feature_group(column_name: str) -> str:
    if "__" in column_name:
        return column_name.split("__")[0]
    return "base"


def select_reference_dataset(inventory_df: pd.DataFrame) -> Optional[str]:
    preferred = inventory_df[
        (inventory_df["dataset_name"] == "full")
        & (inventory_df["problem_type"] == "regression")
        & (inventory_df["feature_engineering"] == "complex")
        & (~inventory_df["has_outliers_removed"])
    ]
    if not preferred.empty:
        return preferred.iloc[0]["file_path"]

    fallback = inventory_df[
        (inventory_df["feature_engineering"] == "complex")
        & (inventory_df["problem_type"] == "regression")
    ]
    if not fallback.empty:
        return fallback.iloc[0]["file_path"]
    return None


def summarize_complex_feature_dataset(
    dataset_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(dataset_path, low_memory=False).convert_dtypes(infer_objects=True)

    target_column = "revenue_usd_adj" if "revenue_usd_adj" in df.columns else None
    non_feature_columns = {"movie_id", "production_size", "is_outlier"}
    if target_column:
        non_feature_columns.add(target_column)

    feature_columns = [col for col in df.columns if col not in non_feature_columns]
    if not feature_columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    feature_groups = (
        pd.Series(feature_columns, name="feature_name")
        .to_frame()
        .assign(feature_group=lambda frame: frame["feature_name"].map(get_feature_group))
    )
    feature_group_summary = (
        feature_groups.groupby("feature_group")
        .agg(feature_count=("feature_name", "count"))
        .reset_index()
        .sort_values("feature_count", ascending=False)
    )
    total_features = int(feature_group_summary["feature_count"].sum())
    feature_group_summary["feature_share_pct"] = (
        100.0 * feature_group_summary["feature_count"] / total_features
    )

    activation_records: List[Dict[str, object]] = []
    for column in feature_columns:
        if not is_bool_like(df[column]):
            continue
        bool_series = to_bool_series(df[column])
        true_count = int(bool_series.sum())
        activation_rate = (100.0 * true_count / len(df)) if len(df) else 0.0
        activation_records.append(
            {
                "feature_name": column,
                "feature_group": get_feature_group(column),
                "true_count": true_count,
                "activation_rate_pct": activation_rate,
            }
        )

    activation_df = pd.DataFrame(activation_records)
    if activation_df.empty:
        return feature_group_summary, pd.DataFrame(), pd.DataFrame()

    activation_df = activation_df.sort_values(
        "activation_rate_pct", ascending=False
    ).reset_index(drop=True)
    activation_group_df = (
        activation_df.groupby("feature_group")
        .agg(
            bool_feature_count=("feature_name", "count"),
            mean_activation_pct=("activation_rate_pct", "mean"),
            min_activation_pct=("activation_rate_pct", "min"),
            max_activation_pct=("activation_rate_pct", "max"),
        )
        .reset_index()
        .sort_values("bool_feature_count", ascending=False)
    )

    return feature_group_summary, activation_df, activation_group_df


def plot_dataset_size_before_after_outlier(
    dataset_shape_df: pd.DataFrame,
    display_chart: bool,
) -> None:
    plot_df = dataset_shape_df[
        ["dataset_name", "rows_with_outliers", "rows_no_outliers", "rows_removed_pct"]
    ].copy()
    plot_df["dataset_name"] = pd.Categorical(
        plot_df["dataset_name"], categories=DATASET_ORDER, ordered=True
    )
    plot_df = plot_df.sort_values("dataset_name")
    plot_df["dataset_label"] = plot_df["dataset_name"].map(DATASET_LABELS)

    bar_df = plot_df.melt(
        id_vars=["dataset_name", "dataset_label", "rows_removed_pct"],
        value_vars=["rows_with_outliers", "rows_no_outliers"],
        var_name="dataset_variant",
        value_name="rows",
    )
    bar_df["dataset_variant"] = bar_df["dataset_variant"].map(
        {
            "rows_with_outliers": "with outliers",
            "rows_no_outliers": "no outliers",
        }
    )

    fig = px.bar(
        bar_df,
        x="dataset_label",
        y="rows",
        color="dataset_variant",
        barmode="group",
        text="rows",
        color_discrete_sequence=COLORS[:2],
        labels={
            "dataset_label": "Production Size Dataset",
            "rows": "Number of Rows",
            "dataset_variant": "Dataset Variant",
        },
    )
    fig.update_traces(
        texttemplate="<b>%{text:,.0f}</b>",
        textposition="outside",
        marker_line_color="#000000",
        marker_line_width=1.5,
    )

    fig = apply_common_style(
        fig,
        title="Dataset Size Before and After Outlier Removal",
        xaxis_title="Production Size Dataset",
        yaxis_title="Number of Rows",
        y_values=bar_df["rows"].to_numpy(),
    )
    apply_layout_overrides(fig, legend=True, bottom_margin=140)
    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(type="linear")

    global_annotation_y = float(bar_df["rows"].max()) * 1.16
    for _, row in plot_df.iterrows():
        label = row["dataset_label"]
        fig.add_annotation(
            x=label,
            y=global_annotation_y,
            text=f"<b>removed {row['rows_removed_pct']:.1f}%</b>",
            showarrow=False,
            font=dict(size=18, color="#000000"),
        )

    filename = f"{STEP_PREFIX}_dataset_size_before_after_outlier.png"
    save_step3_figure(fig, filename)
    if display_chart:
        fig.show()


def plot_feature_width_none_vs_complex(
    feature_width_df: pd.DataFrame,
    display_chart: bool,
) -> None:
    if feature_width_df.empty:
        return

    plot_df = feature_width_df.copy()
    plot_df["feature_engineering"] = plot_df["feature_engineering"].astype(str)

    fig = px.bar(
        plot_df,
        x="feature_engineering",
        y="n_columns",
        color="feature_engineering",
        text="n_columns",
        color_discrete_sequence=[COLORS[0], COLORS[3]],
        labels={
            "feature_engineering": "Feature Engineering",
            "n_columns": "Number of Columns",
        },
    )
    fig.update_traces(
        texttemplate="<b>%{text:.0f}</b>",
        textposition="outside",
        marker_line_color="#000000",
        marker_line_width=1.5,
    )

    fig = apply_common_style(
        fig,
        title="Feature Width Added by Complex Engineering",
        xaxis_title="Feature Engineering",
        yaxis_title="Number of Columns",
        y_values=plot_df["n_columns"].to_numpy(),
    )
    apply_layout_overrides(fig, legend=False, bottom_margin=120, width=900, height=700)
    fig.update_xaxes(tickangle=0)

    if {"none", "complex"}.issubset(set(plot_df["feature_engineering"].tolist())):
        none_cols = int(plot_df.loc[plot_df["feature_engineering"] == "none", "n_columns"].iloc[0])
        complex_cols = int(
            plot_df.loc[plot_df["feature_engineering"] == "complex", "n_columns"].iloc[0]
        )
        fig.add_annotation(
            x="complex",
            y=complex_cols * 1.08,
            text=f"<b>+{complex_cols - none_cols} columns</b>",
            showarrow=False,
            font=dict(size=18, color="#000000"),
        )

    filename = f"{STEP_PREFIX}_feature_width_none_vs_complex.png"
    save_step3_figure(fig, filename)
    if display_chart:
        fig.show()


def plot_complex_feature_group_composition(
    feature_group_summary_df: pd.DataFrame,
    display_chart: bool,
) -> None:
    if feature_group_summary_df.empty:
        return

    plot_df = feature_group_summary_df.copy()
    plot_df["feature_group_chart"] = plot_df["feature_group"].apply(
        lambda group: "historical_kpis" if group.endswith("_kpis") else group
    )
    plot_df = (
        plot_df.groupby("feature_group_chart", as_index=False)
        .agg(feature_count=("feature_count", "sum"))
    )
    plot_df["feature_group_label"] = plot_df["feature_group_chart"].map(
        normalize_feature_group_label
    )
    plot_df = plot_df.sort_values("feature_count", ascending=False)

    fig = px.bar(
        plot_df,
        x="feature_group_label",
        y="feature_count",
        text="feature_count",
        color="feature_count",
        color_continuous_scale="Blues",
        labels={
            "feature_count": "Number of Features",
            "feature_group_label": "Feature Group",
        },
    )
    fig.update_traces(
        texttemplate="<b>%{text:.0f}</b>",
        textposition="outside",
        marker_line_color="#000000",
        marker_line_width=1.2,
    )
    fig.update_coloraxes(showscale=False)

    fig = apply_common_style(
        fig,
        title="Complex Dataset Composition by Feature Group",
        xaxis_title="Feature Group",
        yaxis_title="Number of Features",
        y_values=plot_df["feature_count"].to_numpy(),
    )
    apply_layout_overrides(fig, legend=False, bottom_margin=230, width=1300, height=850)
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(type="linear")

    filename = f"{STEP_PREFIX}_complex_feature_group_composition.png"
    save_step3_figure(fig, filename)
    if display_chart:
        fig.show()


def plot_boolean_activation_group_summary(
    activation_group_df: pd.DataFrame,
    display_chart: bool,
) -> None:
    if activation_group_df.empty:
        return

    plot_df = activation_group_df.copy()
    plot_df["feature_group_label"] = plot_df["feature_group"].map(normalize_feature_group_label)
    plot_df = plot_df.sort_values("mean_activation_pct", ascending=False)
    plot_df["label_text"] = plot_df.apply(
        lambda row: f"{row['mean_activation_pct']:.1f}% (n={int(row['bool_feature_count'])})",
        axis=1,
    )

    fig = px.bar(
        plot_df,
        x="feature_group_label",
        y="mean_activation_pct",
        text="label_text",
        color="bool_feature_count",
        color_continuous_scale="Tealgrn",
        labels={
            "feature_group_label": "Feature Group",
            "mean_activation_pct": "Mean Activation Rate (%)",
            "bool_feature_count": "Boolean Feature Count",
        },
    )
    fig.update_traces(
        textposition="outside",
        marker_line_color="#000000",
        marker_line_width=1.2,
    )

    fig = apply_common_style(
        fig,
        title="Boolean Activation Rate by Feature Group",
        xaxis_title="Feature Group",
        yaxis_title="Mean Activation Rate (%)",
        y_values=plot_df["mean_activation_pct"].to_numpy(),
    )
    apply_layout_overrides(fig, legend=False, bottom_margin=180, width=1350, height=820)
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(type="linear")
    fig.update_coloraxes(showscale=False)

    filename = f"{STEP_PREFIX}_boolean_activation_group_summary.png"
    save_step3_figure(fig, filename)
    if display_chart:
        fig.show()


def plot_inventory_metric(
    inventory_df: pd.DataFrame,
    metric_column: str,
    yaxis_title: str,
    title: str,
    filename_suffix: str,
    display_chart: bool,
) -> None:
    plot_df = inventory_df.copy()
    plot_df["dataset_name"] = pd.Categorical(
        plot_df["dataset_name"], categories=DATASET_ORDER, ordered=True
    )
    plot_df["problem_type"] = pd.Categorical(
        plot_df["problem_type"], categories=PROBLEM_ORDER, ordered=True
    )
    plot_df["config_label"] = pd.Categorical(
        plot_df["config_label"], categories=CONFIG_ORDER, ordered=True
    )
    plot_df = plot_df.sort_values(["problem_type", "dataset_name", "config_label"])

    fig = px.bar(
        plot_df,
        x="dataset_name",
        y=metric_column,
        color="config_label",
        facet_col="problem_type",
        barmode="group",
        category_orders={
            "dataset_name": DATASET_ORDER,
            "problem_type": PROBLEM_ORDER,
            "config_label": CONFIG_ORDER,
        },
        color_discrete_sequence=COLORS,
        text=metric_column,
        labels={
            "dataset_name": "Production Size Dataset",
            "config_label": "Configuration",
            metric_column: yaxis_title,
            "problem_type": "Problem Type",
        },
    )

    text_template = "<b>%{text:.0f}</b>"
    if metric_column == "rows":
        text_template = "<b>%{text:,.0f}</b>"
    fig.update_traces(
        texttemplate=text_template,
        textposition="outside",
        marker_line_color="#000000",
        marker_line_width=1.5,
    )

    fig = apply_common_style(
        fig,
        title=title,
        xaxis_title="Production Size",
        yaxis_title=yaxis_title,
        y_values=plot_df[metric_column].to_numpy(),
    )
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(showlegend=True)

    filename = generate_filename(
        "dataset_inventory",
        {"metric": filename_suffix},
        prefix=STEP_PREFIX,
    )
    save_step3_figure(fig, filename)

    if display_chart:
        fig.show()


def plot_outlier_removal_impact(
    outlier_impact_df: pd.DataFrame,
    display_chart: bool,
) -> None:
    plot_df = outlier_impact_df.copy()
    plot_df["dataset_name"] = pd.Categorical(
        plot_df["dataset_name"], categories=DATASET_ORDER, ordered=True
    )
    plot_df["problem_type"] = pd.Categorical(
        plot_df["problem_type"], categories=PROBLEM_ORDER, ordered=True
    )
    plot_df["feature_engineering"] = pd.Categorical(
        plot_df["feature_engineering"], categories=FEATURE_ORDER, ordered=True
    )
    plot_df = plot_df.sort_values(["problem_type", "dataset_name", "feature_engineering"])

    fig = px.bar(
        plot_df,
        x="dataset_name",
        y="rows_removed_pct",
        color="feature_engineering",
        facet_col="problem_type",
        barmode="group",
        category_orders={
            "dataset_name": DATASET_ORDER,
            "problem_type": PROBLEM_ORDER,
            "feature_engineering": FEATURE_ORDER,
        },
        color_discrete_sequence=COLORS,
        text="rows_removed_pct",
        labels={
            "dataset_name": "Production Size Dataset",
            "rows_removed_pct": "Rows Removed (%)",
            "feature_engineering": "Feature Engineering",
            "problem_type": "Problem Type",
        },
    )

    fig.update_traces(
        texttemplate="<b>%{text:.1f}%</b>",
        textposition="outside",
        marker_line_color="#000000",
        marker_line_width=1.5,
    )

    fig = apply_common_style(
        fig,
        title="Outlier Removal Impact by Dataset and Task",
        xaxis_title="Production Size",
        yaxis_title="Rows Removed (%)",
        y_values=plot_df["rows_removed_pct"].to_numpy(),
    )
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    fig.update_xaxes(tickangle=-45)

    filename = generate_filename(
        "outlier_impact",
        {"metric": "rows_removed_pct"},
        prefix=STEP_PREFIX,
    )
    save_step3_figure(fig, filename)

    if display_chart:
        fig.show()


def plot_top_boolean_activations(
    activation_df: pd.DataFrame,
    top_n: int,
    display_chart: bool,
) -> None:
    if activation_df.empty:
        return

    plot_df = activation_df.head(top_n).copy()
    plot_df = plot_df.sort_values("activation_rate_pct", ascending=False)

    fig = px.bar(
        plot_df,
        x="feature_name",
        y="activation_rate_pct",
        color="feature_group",
        text="activation_rate_pct",
        color_discrete_sequence=COLORS,
        labels={
            "feature_name": "Feature",
            "activation_rate_pct": "Activation Rate (%)",
            "feature_group": "Feature Group",
        },
    )
    fig.update_traces(
        texttemplate="<b>%{text:.1f}%</b>",
        textposition="outside",
        marker_line_color="#000000",
        marker_line_width=1.5,
    )

    fig = apply_common_style(
        fig,
        title=f"Top {top_n} Boolean Feature Activations",
        xaxis_title="Feature",
        yaxis_title="Activation Rate (%)",
        y_values=plot_df["activation_rate_pct"].to_numpy(),
    )
    fig.update_xaxes(tickangle=-70)

    filename = generate_filename(
        "feature_activation",
        {"top_n": top_n},
        prefix=STEP_PREFIX,
    )
    save_step3_figure(fig, filename)

    if display_chart:
        fig.show()


def run_feature_engineering_analysis(
    ml_ready_path: Optional[str] = None,
    display_chart: bool = False,
    top_n_activation: int = 30,
) -> None:
    ensure_output_dirs()
    ml_ready_dir = resolve_ml_ready_dir(ml_ready_path)
    print(f"Loading feature-engineering datasets from: {ml_ready_dir}")

    inventory_df = load_dataset_inventory(ml_ready_dir)
    if inventory_df.empty:
        print("No datasets found in ml_ready_data. Nothing to analyze.")
        return

    print(f"Loaded {len(inventory_df)} dataset variants.")

    inventory_export = inventory_df[
        [
            "dataset_name",
            "problem_type",
            "feature_engineering",
            "outlier_strategy",
            "rows",
            "n_columns",
            "outlier_count",
            "outlier_rate_pct",
            "target_cardinality",
            "majority_class_pct",
            "file_size_mb",
            "file_name",
        ]
    ].copy()
    save_step3_table(
        inventory_export,
        "table_step3_dataset_inventory",
        "Dataset Inventory for Feature Engineering Outputs",
        "tab:step3_dataset_inventory",
    )

    outlier_impact_df = build_outlier_impact_table(inventory_df)
    save_step3_table(
        outlier_impact_df,
        "table_step3_outlier_impact",
        "Outlier Removal Impact Across Dataset Variants",
        "tab:step3_outlier_impact",
    )

    feature_impact_df = build_feature_engineering_impact_table(inventory_df)
    save_step3_table(
        feature_impact_df,
        "table_step3_feature_impact",
        "Feature Engineering Impact (Complex vs None)",
        "tab:step3_feature_impact",
    )

    dataset_shape_summary_df = build_dataset_shape_summary(inventory_df)
    save_step3_table(
        dataset_shape_summary_df,
        "table_step3_dataset_shape_summary",
        "Dataset Variant Summary (Rows, Outlier Impact, and Feature Width)",
        "tab:step3_dataset_shape_summary",
    )

    feature_width_df = build_feature_width_summary(inventory_df)
    save_step3_table(
        feature_width_df,
        "table_step3_feature_width_summary",
        "Feature Width Summary (None vs Complex)",
        "tab:step3_feature_width_summary",
    )

    plot_dataset_size_before_after_outlier(
        dataset_shape_summary_df,
        display_chart=display_chart,
    )
    plot_feature_width_none_vs_complex(
        feature_width_df,
        display_chart=display_chart,
    )

    reference_dataset_path = select_reference_dataset(inventory_df)
    if reference_dataset_path:
        print(f"Running complex-feature deep dive on: {reference_dataset_path}")
        group_summary_df, activation_df, activation_group_df = summarize_complex_feature_dataset(
            reference_dataset_path
        )
        save_step3_table(
            group_summary_df,
            "table_step3_feature_group_summary",
            "Feature Group Summary for Complex Regression Dataset",
            "tab:step3_feature_group_summary",
        )
        save_step3_table(
            activation_group_df,
            "table_step3_feature_activation_group_summary",
            "Boolean Activation Summary by Feature Group",
            "tab:step3_feature_activation_group_summary",
        )

        top_activation_df = activation_df.head(top_n_activation).copy()
        save_step3_table(
            top_activation_df,
            "table_step3_top_boolean_activations",
            f"Top {top_n_activation} Boolean Feature Activations",
            "tab:step3_top_boolean_activations",
        )
        plot_complex_feature_group_composition(
            group_summary_df,
            display_chart=display_chart,
        )
        plot_boolean_activation_group_summary(
            activation_group_df,
            display_chart=display_chart,
        )

    print("Feature engineering analysis complete. Check thesis_assets/charts and thesis_assets/tables.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze feature-engineering datasets generated in data/ml_ready_data."
    )
    parser.add_argument(
        "--ml-ready-path",
        type=str,
        default=None,
        help="Optional explicit path to the ml_ready_data directory.",
    )
    parser.add_argument(
        "--display-chart",
        action="store_true",
        help="Display charts interactively while running.",
    )
    parser.add_argument(
        "--top-n-activation",
        type=int,
        default=30,
        help="Number of top boolean activation features to export and chart.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_feature_engineering_analysis(
        ml_ready_path=args.ml_ready_path,
        display_chart=args.display_chart,
        top_n_activation=args.top_n_activation,
    )
