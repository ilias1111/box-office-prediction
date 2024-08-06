import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Union, Any
from datetime import datetime
from glob import glob
import os
import json
import pandas as pd
from typing import List, Dict, Union, Tuple


def load_metadata(type: str, metadata_path: str) -> pd.DataFrame:
    json_files = glob(os.path.join(metadata_path, "*.json"))
    results = []

    for file in json_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)

                if type == "metrics":
                    parametric_data = data.get("metrics", {})
                elif type == "parameters":
                    parametric_data = data.get("model_parameters", {})
                elif type == "feature_importance":
                    parametric_data = data.get("feature_importance", {})

                experiment_info = {
                    "run_id": data.get("run_id"),
                    "timestamp": datetime.strptime(
                        data.get("timestamp", ""), "%Y%m%d_%H%M%S"
                    ),
                    "model_type": data.get("model_type"),
                    "problem_type": data.get("problem_type"),
                    "dataset_name": data.get("dataset_name"),
                    "grid_type": data.get("grid_type"),
                    "has_outliers_removed": data.get("has_outliers_removed"),
                    "feature_engineering": data.get("feature_engineering"),
                    "scaler": data.get("scaler"),
                    "variance_threshold": data.get("variance_threshold"),
                    "duration": data.get("duration"),
                    "number_of_combinations": data.get("number_of_combinations"),
                    **parametric_data,
                }
                results.append(experiment_info)
        except Exception as e:
            print(f"Error loading file {file}: {e}")

    return pd.DataFrame(results)


def generate_filename(base: str, params: Dict[str, Any]) -> str:
    """Generate a smart, descriptive filename for the chart."""
    param_str = "_".join([f"{k}_{v}" for k, v in params.items() if v is not None])
    return f"{base}_{param_str}.png".replace(" ", "_").lower()


def save_figure(fig: go.Figure, filename: str, output_dir: str = "charts") -> None:
    """Save the figure as a PNG file in the specified output directory."""
    os.makedirs(output_dir, exist_ok=True)
    fig.write_image(os.path.join(output_dir, filename), scale=2)


# Define a consistent color palette
COLORS = [
    "#66C2A5",
    "#FC8D62",
    "#8DA0CB",
    "#E78AC3",
    "#A6D854",
    "#FFD92F",
    "#E5C494",
    "#B3B3B3",
]


def select_best_models(
    experiment_df: pd.DataFrame,
    problem_types: Union[str, List[str]],
    groupby_columns: List[str],
    metrics: Dict[str, Dict[str, str]],
    display_columns: Dict[str, List[str]],
) -> Dict[str, pd.DataFrame]:
    """
    Select the best models for each problem type based on specified metrics.

    Parameters:
    - experiment_df: DataFrame containing experiment results
    - problem_types: String or list of strings specifying the problem types to analyze
    - groupby_columns: List of columns to group by (e.g., ['dataset_name', 'has_outliers_removed', 'feature_engineering'])
    - metrics: Dictionary specifying the metric to optimize for each problem type and whether to maximize or minimize
               e.g., {'regression': {'metric': 'MAPE', 'operation': 'min'},
                      'classification': {'metric': 'F1 Score', 'operation': 'max'}}
    - display_columns: Dictionary specifying which columns to display in the result for each problem type
               e.g., {'regression': ['model_type', 'scaler', 'MAPE', 'R2'],
                      'classification': ['model_type', 'scaler', 'F1 Score']}

    Returns:
    - Dictionary with problem types as keys and DataFrames of best models as values
    """
    if isinstance(problem_types, str):
        problem_types = [problem_types]

    results = {}

    for problem_type in problem_types:
        df_subset = experiment_df[experiment_df["problem_type"] == problem_type]
        metric = metrics[problem_type]["metric"]
        operation = metrics[problem_type]["operation"]

        grouped = df_subset.groupby(["problem_type"] + groupby_columns)

        if operation == "max":
            idx = grouped[metric].idxmax()
        elif operation == "min":
            idx = grouped[metric].idxmin()
        else:
            raise ValueError(
                f"Invalid operation '{operation}' for problem type '{problem_type}'"
            )

        best_models = df_subset.loc[idx]

        columns_to_display = (
            ["problem_type"] + groupby_columns + display_columns[problem_type]
        )
        results[problem_type] = best_models[columns_to_display]

    return results


def should_use_log_scale(values):
    """Determine if log scale should be used based on data range."""
    min_val = np.min(values)
    max_val = np.max(values)
    return (
        max_val / max(min_val, 1e-10) > 100
    )  # Use log scale if range spans more than 2 orders of magnitude


def apply_common_style(
    fig: go.Figure,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    y_values: List[float],
) -> go.Figure:
    """Apply common style to all charts."""
    use_log_scale = should_use_log_scale(y_values)

    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=20, color="#333333"),
        },
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font=dict(family="Arial", size=12, color="#333333"),
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridcolor="#E5E5E5",
            tickfont=dict(size=10),
            tickangle=0,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#E5E5E5",
            tickfont=dict(size=10),
            zeroline=True,
            zerolinecolor="#E5E5E5",
            type="log" if use_log_scale else "linear",
        ),
        margin=dict(l=60, r=30, t=80, b=50),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.15,
    )
    return fig


def plot_one_metric_of_different_datasets_per_feature_engineering_outliers_with_plotly(
    experiment_df: pd.DataFrame,
    problem_type: str = "binary_classification",
    metric: str = "F1 Score",
    metric_agg: str = "max",
    display_chart: bool = True,
    output_dir: str = "charts",
) -> pd.DataFrame:
    experiment_df = experiment_df[experiment_df["problem_type"] == problem_type]

    feature_engineering_order = ["none", "complex"]
    has_outliers_removed_order = [False, True]

    grouped_data = (
        experiment_df.groupby(
            ["dataset_name", "has_outliers_removed", "feature_engineering"]
        )
        .agg({metric: metric_agg})
        .reset_index()
    )

    for dataset_name, data in grouped_data.groupby("dataset_name"):
        fig = px.bar(
            data,
            x="feature_engineering",
            y=metric,
            color="has_outliers_removed",
            text=metric,
            category_orders={
                "feature_engineering": feature_engineering_order,
                "has_outliers_removed": has_outliers_removed_order,
            },
            labels={
                "feature_engineering": "Feature Engineering Type",
                "has_outliers_removed": "Outliers Removed",
                metric: f"{metric} Value",
            },
            color_discrete_sequence=COLORS,
            barmode="group",
        )

        fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")

        fig = apply_common_style(
            fig,
            title=f"{metric} for {dataset_name}",
            xaxis_title="Feature Engineering Type",
            yaxis_title=f"{metric} Value",
            y_values=data[metric],
        )

        filename = generate_filename(
            "dataset_comparison",
            {
                "dataset": dataset_name,
                "problem_type": problem_type,
                "metric": metric,
                "agg": metric_agg,
            },
        )
        save_figure(fig, filename, output_dir)

        if display_chart:
            fig.show()

    return experiment_df


def plot_one_metric_of_different_models_per_dataset_with_plotly(
    experiment_df: pd.DataFrame,
    problem_type: str = "regression",
    metric: str = "MAPE",
    metric_agg: str = "min",
    benchmark_model: str = "dummy_regressor",
    display_chart: bool = True,
    output_dir: str = "charts",
) -> pd.DataFrame:
    GROUP_ORDER = ["none_False", "complex_False", "none_True", "complex_True"]
    NEW_LABELS = {
        "none_False": "FE: none<br>With Outliers",
        "none_True": "FE: none<br>No Outliers",
        "complex_False": "FE: complex<br>With Outliers",
        "complex_True": "FE: complex<br>No Outliers",
    }

    experiment_df = experiment_df[experiment_df["problem_type"] == problem_type]
    grouped_data = (
        experiment_df.groupby(
            [
                "dataset_name",
                "feature_engineering",
                "has_outliers_removed",
                "model_type",
            ]
        )
        .agg({metric: metric_agg})
        .reset_index()
    )
    grouped_data["group"] = (
        grouped_data["feature_engineering"]
        + "_"
        + grouped_data["has_outliers_removed"].astype(str)
    )

    for dataset in grouped_data["dataset_name"].unique():
        dataset_data = grouped_data[grouped_data["dataset_name"] == dataset]

        fig = go.Figure()

        benchmark_data = dataset_data[dataset_data["model_type"] == benchmark_model]
        fig.add_trace(
            go.Scatter(
                name=f"{benchmark_model} (Benchmark)",
                x=benchmark_data["group"],
                y=benchmark_data[metric],
                mode="markers+text",
                marker=dict(color="#000000", size=12, symbol="diamond"),
                text=benchmark_data[metric].round(2),
                textposition="top center",
                textfont=dict(color="#000000", size=10),
            )
        )

        model_colors = {
            model: COLORS[i % len(COLORS)]
            for i, model in enumerate(dataset_data["model_type"].unique())
            if model != benchmark_model
        }

        for model in dataset_data["model_type"].unique():
            if model != benchmark_model:
                model_data = (
                    dataset_data[dataset_data["model_type"] == model]
                    .set_index("group")
                    .reindex(GROUP_ORDER)
                    .reset_index()
                )
                fig.add_trace(
                    go.Bar(
                        name=model,
                        x=model_data["group"],
                        y=model_data[metric],
                        text=model_data[metric].round(2),
                        textposition="outside",
                        marker_color=model_colors[model],
                        textfont=dict(size=10),
                    )
                )

        fig = apply_common_style(
            fig,
            title=f"{metric} Comparison for {dataset}",
            xaxis_title="Feature Engineering and Outlier Removal",
            yaxis_title=f"{metric} Value",
            y_values=dataset_data[metric],
        )

        fig.update_xaxes(
            ticktext=list(NEW_LABELS.values()), tickvals=list(NEW_LABELS.keys())
        )

        filename = generate_filename(
            "model_comparison",
            {
                "dataset": dataset,
                "problem_type": problem_type,
                "metric": metric,
                "agg": metric_agg,
                "benchmark": benchmark_model,
            },
        )
        save_figure(fig, filename, output_dir)

        if display_chart:
            fig.show()

    return experiment_df


def plot_and_export_categorical_distribution(
    df: pd.DataFrame,
    agg_column: str,
    other_threshold: int,
    sort_by_value: bool = True,
    display_chart: bool = True,
    output_dir: str = "charts",
) -> pd.DataFrame:
    counts = df[agg_column].value_counts().reset_index()
    counts.columns = [agg_column, "count"]

    counts["grouped"] = counts.apply(
        lambda x: "Other" if x["count"] < other_threshold else x[agg_column], axis=1
    )

    grouped_counts = counts.groupby("grouped").agg(total=("count", "sum")).reset_index()
    if sort_by_value:
        grouped_counts = grouped_counts.sort_values(
            "total", ascending=False
        ).reset_index(drop=True)
    else:
        grouped_counts = grouped_counts.sort_values(
            "grouped", ascending=True
        ).reset_index(drop=True)

    grouped_counts["grouped"] = grouped_counts["grouped"].astype(str)

    color_map = {
        category: COLORS[i % len(COLORS)]
        for i, category in enumerate(grouped_counts["grouped"].unique())
    }
    color_map["Other"] = "#B3B3B3"  # Always use gray for 'Other'

    fig = px.bar(
        grouped_counts,
        x="grouped",
        y="total",
        text="total",
        color="grouped",
        color_discrete_map=color_map,
    )

    fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")

    fig = apply_common_style(
        fig,
        title=f"Count of Movies by {agg_column}",
        xaxis_title=f"{agg_column}",
        yaxis_title="Number of Movies",
        y_values=grouped_counts["total"],
    )

    fig.update_xaxes(tickangle=-45)

    filename = generate_filename(
        "categorical_distribution",
        {
            "column": agg_column,
            "threshold": other_threshold,
            "sort": "value" if sort_by_value else "name",
        },
    )
    save_figure(fig, filename, output_dir)

    if display_chart:
        fig.show()

    return df
