import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Union, Any, Optional
from datetime import datetime
from glob import glob
import os
import json
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# Define constants
COLORS = [
    "#66C2A5",  # Teal
    "#8DA0CB",  # Blue-gray
    "#E78AC3",  # Pink
    "#A6D854",  # Light green
    "#FFD92F",  # Yellow
    "#E5C494",  # Tan
    "#B3B3B3",  # Gray
]

DEFAULT_CHART_CONFIG = {
    "font_family": "Times New Roman",
    "base_font_size": 26,
    "title_font_size": 28,
    "axis_font_size": 20,
    "label_font_size": 20,
    "text_color": "#000000",
    "grid_color": "#E5E5E5",
    "border_color": "#000000",
    "border_width": 2.5,
    "margin": dict(l=100, r=50, t=160, b=300), # Increased bottom margin for legend
    "height": 1000, # Increased height to accommodate legend
    "width": 1200,
    "legend_font_size": 16,
    "legend_y_anchor": "top",
    "legend_x_anchor": "center",
    "legend_orientation": "h",
    "legend_y": -0.25,
    "legend_x": 0.5,
}


def should_use_log_scale(values: np.ndarray) -> bool:
    """Determine if log scale should be used based on data range."""
    min_val = np.min(values)
    max_val = np.max(values)
    return max_val / max(min_val, 1e-10) > 100


def generate_filename(base: str, params: Dict[str, Any], prefix: Optional[str] = None) -> str:
    """Generate a smart, descriptive filename for the chart."""
    param_str = "_".join(f"{k}_{v}" for k, v in params.items() if v is not None)
    filename = f"{base}_{param_str}.png".replace(" ", "_").lower()
    return f"{prefix}_{filename}" if prefix else filename


def save_figure(fig: go.Figure, filename: str, output_dir: str = "charts") -> None:
    """Save the figure as a PNG file in the specified output directory."""
    os.makedirs(output_dir, exist_ok=True)
    fig.write_image(os.path.join(output_dir, filename), scale=3)


def apply_common_style(
    fig: go.Figure,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    y_values: List[float],
) -> go.Figure:
    """Apply common style to all charts."""
    cfg = DEFAULT_CHART_CONFIG
    use_log_scale = should_use_log_scale(y_values)

    # Title configuration
    title_config = {
        "text": f"<b>{title}</b>",
        "y": 0.95,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
        "font": dict(size=cfg["title_font_size"], color=cfg["text_color"]),
    }

    # Axis configuration
    axis_config = dict(
        showgrid=True,
        gridcolor=cfg["grid_color"],
        gridwidth=1,
        tickfont=dict(size=cfg["base_font_size"]),
        linecolor=cfg["border_color"],
        linewidth=2,
        ticks="outside",
        tickwidth=2,
    )

    # Layout update
    fig.update_layout(
        title=title_config,
        xaxis_title=dict(
            text=f"<b>{xaxis_title}</b>",
            font=dict(size=cfg["axis_font_size"], color=cfg["text_color"])
        ),
        yaxis_title=dict(
            text=f"<b>{yaxis_title}</b>",
            font=dict(size=cfg["axis_font_size"], color=cfg["text_color"])
        ),
        font=dict(
            family=cfg["font_family"],
            size=cfg["base_font_size"],
            color=cfg["text_color"]
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=axis_config,
        yaxis={
            **axis_config,
            "zeroline": True,
            "zerolinecolor": cfg["border_color"],
            "zerolinewidth": 2,
            "type": "log" if use_log_scale else "linear",
        },
        margin=cfg["margin"],
        showlegend=True,
        legend=dict(
            orientation=cfg["legend_orientation"],
            yanchor=cfg["legend_y_anchor"],
            y=cfg["legend_y"],
            xanchor=cfg["legend_x_anchor"],
            x=cfg["legend_x"],
            font=dict(size=cfg["legend_font_size"]),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=cfg["border_color"],
            borderwidth=cfg["border_width"],
        ),
        bargap=0.2,
        height=cfg["height"],
        width=cfg["width"],
    )

    # Add border
    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color=cfg["border_color"], width=cfg["border_width"]),
                fillcolor="rgba(0,0,0,0)",
            )
        ]
    )
    
    return fig


def load_metadata(type: str, metadata_path: str) -> pd.DataFrame:
    """Load and process metadata from JSON files."""
    json_files = glob(os.path.join(metadata_path, "*.json"))
    results = []

    for file in json_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                
                # Extract common experiment info
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
                }
                
                # Select appropriate data based on type
                if type == "metrics":
                    metrics_data = data.get("metrics", {})
                    experiment_info.update(metrics_data)
                    # Add regression prediction data if available
                    if data.get("problem_type") == "regression":
                        experiment_info["predicted_vs_actual"] = json.dumps(
                            data.get("predicted_vs_actual", {})
                        )
                elif type == "parameters":
                    experiment_info.update(data.get("model_parameters", {}))
                elif type == "confusion_matrix":
                    experiment_info["conf_matrix"] = data.get("conf_matrix", {})
                else:
                    raise ValueError(f"Unknown metadata type: {type}")

                results.append(experiment_info)
        except Exception as e:
            print(f"Error loading file {file}: {e}")

    return pd.DataFrame(results)


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


def plot_one_metric_of_different_datasets_per_feature_engineering_outliers_with_plotly(
    experiment_df: pd.DataFrame,
    problem_type: str = "binary_classification",
    metric: str = "F1 Score",
    metric_agg: str = "max",
    display_chart: bool = True,
    output_dir: str = "charts",
    print_stats: bool = False,
    filename_prefix: Optional[str] = None,
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

        fig.update_traces(
            texttemplate="<b>%{text:.2f}</b>",  # Bold text
            textposition="outside",
            textfont=dict(size=16, color="#000000"),  # Larger, darker text
            textangle=0,  # Horizontal text
            marker_line_color="#000000",  # Black border for bars
            marker_line_width=1.5,
        )

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
            prefix=filename_prefix
        )
        save_figure(fig, filename, output_dir)

        if display_chart:
            fig.show()

    if print_stats:
        print(grouped_data)

    return grouped_data


def plot_one_metric_of_different_models_per_dataset_with_plotly(
    experiment_df: pd.DataFrame,
    problem_type: str = "regression",
    metric: str = "MAPE",
    metric_agg: str = "min",
    benchmark_model: str = "dummy_regressor",
    display_chart: bool = True,
    output_dir: str = "charts",
    print_stats: bool = False,
    filename_prefix: Optional[str] = None,
) -> pd.DataFrame:
    GROUP_ORDER = ["none_False", "complex_False", "none_True", "complex_True"]
    NEW_LABELS = {
        "none_False": "FE: none<br>With Outliers",
        "complex_False": "FE: complex<br>With Outliers",
        "none_True": "FE: none<br>No Outliers",
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
        benchmark_data = (
            benchmark_data.set_index("group").reindex(GROUP_ORDER).reset_index()
        )
        fig.add_trace(
            go.Scatter(
                name=f"{benchmark_model} (Benchmark)",
                x=GROUP_ORDER,
                y=benchmark_data[metric],
                mode="markers+text",
                marker=dict(
                    color="#000000",
                    size=14,
                    symbol="diamond",
                    line=dict(color="#000000", width=2)
                ),
                text=benchmark_data[metric].round(2),
                textposition="top center",
                textfont=dict(color="#000000", size=16),
                texttemplate='<b>%{text:.2f}</b>',
            )
        )

        # Get unique models excluding benchmark
        non_benchmark_models = [
            model for model in dataset_data["model_type"].unique() 
            if model != benchmark_model
        ]
        
        # Assign colors only to non-benchmark models
        model_colors = {
            model: COLORS[i % len(COLORS)]
            for i, model in enumerate(non_benchmark_models)
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
                        x=GROUP_ORDER,
                        y=model_data[metric],
                        text=model_data[metric].round(2),
                        textposition="outside",
                        marker_color=model_colors[model],
                        marker_line_color="#000000",
                        marker_line_width=1.5,
                        textfont=dict(size=16, color="#000000"),
                        textangle=0,
                        texttemplate='<b>%{text:.2f}</b>',
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
            ticktext=[NEW_LABELS[group] for group in GROUP_ORDER], tickvals=GROUP_ORDER
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
            prefix=filename_prefix
        )
        save_figure(fig, filename, output_dir)

        if display_chart:
            fig.show()

    if print_stats:
        print(grouped_data)

    return grouped_data


def plot_and_export_categorical_distribution(
    df: pd.DataFrame,
    agg_column: str,
    nickname_agg_column: str = None,
    other_threshold: int = 10,
    sort_by_value: bool = True,
    display_chart: bool = True,
    output_dir: str = "charts",
    format_as_int: bool = False,
    print_stats: bool = False,
    filename_prefix: Optional[str] = None,
) -> pd.DataFrame:

    if nickname_agg_column is not None:
        df[nickname_agg_column] = df[agg_column]
        agg_column = nickname_agg_column
    
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

    color_map = {category: "#66C2A5" for category in grouped_counts["grouped"].unique()}
    color_map["Other"] = "#E78AC3"

    fig = px.bar(
        grouped_counts,
        x="grouped",
        y="total",
        text="total",
        color="grouped",
        color_discrete_map=color_map,
    )

    if format_as_int:
        fig.update_traces(
            texttemplate="<b>%{text:.0f}</b>",
            textposition="outside",
            textfont=dict(size=16, color="#000000"),
            textangle=0,
            marker_line_color="#000000",
            marker_line_width=1.5,
        )
    else:
        fig.update_traces(
            texttemplate="<b>%{text:.2s}</b>",
            textposition="outside",
            textfont=dict(size=16, color="#000000"),
            textangle=0,
            marker_line_color="#000000",
            marker_line_width=1.5,
        )

    fig = apply_common_style(
        fig,
        title=f"Number of Movies by {agg_column}",
        xaxis_title=f"{agg_column}",
        yaxis_title="Number of Movies",
        y_values=grouped_counts["total"],
    )

    fig.update_xaxes(tickangle=-45)

    # Remove the legend
    fig.update_layout(showlegend=False)

    filename = generate_filename(
        "categorical_distribution",
        {
            "column": agg_column,
            "threshold": other_threshold,
            "sort": "value" if sort_by_value else "name",
        },
        prefix=filename_prefix
    )
    save_figure(fig, filename, output_dir)

    if display_chart:
        fig.show()

    if print_stats:
        ## Pretty print the stats
        print(grouped_counts)


    return grouped_counts


def plot_confusion_matrix(
    conf_matrix_data: Union[List[Dict[str, int]], Dict[str, Dict[str, int]]],
    display_chart: bool = True,
    output_dir: str = "charts",
    print_stats: bool = False,
    filename: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> None:
    """Create and save a styled confusion matrix visualization."""
    # Convert list format to dictionary format if needed
    if isinstance(conf_matrix_data, list):
        labels = sorted(list(set([k for d in conf_matrix_data for k in d.keys()])))
        matrix = np.zeros((len(labels), len(labels)))
        for i, true_dict in enumerate(conf_matrix_data):
            for pred_label, count in true_dict.items():
                j = labels.index(pred_label)
                matrix[i][j] = count
    else:
        labels = sorted(list(set(
            list(conf_matrix_data.keys()) + 
            [k for d in conf_matrix_data.values() for k in d.keys()]
        )))
        matrix = np.zeros((len(labels), len(labels)))
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                matrix[i][j] = conf_matrix_data[true_label].get(pred_label, 0)
    
    # Calculate percentages
    matrix_sum = matrix.sum()
    matrix_pct = (matrix / matrix_sum) * 100
    
    # Create annotations
    annotations = []
    for i in range(len(labels)):
        row_annotations = []
        for j in range(len(labels)):
            row_annotations.append(
                f"<b>{matrix[i][j]:.0f}</b><br>({matrix_pct[i][j]:.1f}%)"
            )
        annotations.append(row_annotations)
    
    # Split title into main title and subtitle
    title_parts = title.split("\n") if title else ["Confusion Matrix"]
    main_title = title_parts[0]
    subtitle = "<br>".join(title_parts[1:]) if len(title_parts) > 1 else ""
    
    # Create heatmap with updated styling
    fig = ff.create_annotated_heatmap(
        z=matrix,
        x=labels,
        y=labels,
        annotation_text=annotations,
        colorscale=[
            [0, DEFAULT_CHART_CONFIG["grid_color"]], 
            [1, COLORS[0]]
        ],
        showscale=True,
        hoverongaps=False,
        hoverinfo='z',
    )
    
    # Update layout with improved styling
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{main_title}</b>" +
                (f"<br><sup>{subtitle}</sup>" if subtitle else "")
            ),
            y=0.95,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(
                size=DEFAULT_CHART_CONFIG["title_font_size"],
                color=DEFAULT_CHART_CONFIG["text_color"]
            ),
        ),
        xaxis_title=dict(
            text="<b>Predicted Label</b>",
            font=dict(
                size=DEFAULT_CHART_CONFIG["axis_font_size"],
                color=DEFAULT_CHART_CONFIG["text_color"]
            )
        ),
        yaxis_title=dict(
            text="<b>True Label</b>",
            font=dict(
                size=DEFAULT_CHART_CONFIG["axis_font_size"],
                color=DEFAULT_CHART_CONFIG["text_color"]
            )
        ),
        font=dict(
            family=DEFAULT_CHART_CONFIG["font_family"],
            size=DEFAULT_CHART_CONFIG["base_font_size"],
            color=DEFAULT_CHART_CONFIG["text_color"]
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=DEFAULT_CHART_CONFIG["width"],
        height=DEFAULT_CHART_CONFIG["height"],
        margin=DEFAULT_CHART_CONFIG["margin"],
    )
    
    # Update axes
    fig.update_xaxes(
        side="bottom",
        tickfont=dict(size=DEFAULT_CHART_CONFIG["axis_font_size"]),
        showgrid=True,
        gridcolor=DEFAULT_CHART_CONFIG["grid_color"],
        gridwidth=1,
        linecolor=DEFAULT_CHART_CONFIG["border_color"],
        linewidth=2,
        ticks="outside",
        tickwidth=2,
    )
    
    fig.update_yaxes(
        tickfont=dict(size=DEFAULT_CHART_CONFIG["axis_font_size"]),
        showgrid=True,
        gridcolor=DEFAULT_CHART_CONFIG["grid_color"],
        gridwidth=1,
        linecolor=DEFAULT_CHART_CONFIG["border_color"],
        linewidth=2,
        ticks="outside",
        tickwidth=2,
    )
    
    # Add border
    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(
                    color=DEFAULT_CHART_CONFIG["border_color"],
                    width=DEFAULT_CHART_CONFIG["border_width"]
                ),
                fillcolor="rgba(0,0,0,0)",
            )
        ]
    )
    
    # Add hover template
    fig.update_traces(
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
    )
    
    # Save figure with provided filename
    save_figure(fig, filename, output_dir)
    
    if display_chart:
        fig.show()
        
    if print_stats:
        print("\nConfusion Matrix:")
        print(pd.DataFrame(matrix, index=labels, columns=labels))
        print(f"\nTotal samples: {matrix_sum:.0f}")


def plot_confusion_matrices_from_metadata(
    metadata_path: str,
    display_chart: bool = True,
    output_dir: str = "charts",
    print_stats: bool = False,
    filename_prefix: Optional[str] = None,
) -> None:
    """
    Load confusion matrices from metadata files and create visualizations.
    
    Parameters:
    - metadata_path: Path to metadata directory
    - display_chart: Whether to display charts interactively
    - output_dir: Directory to save output images
    - print_stats: Whether to print confusion matrix statistics
    - filename_prefix: Optional prefix for output filenames
    """
    # Load confusion matrix data
    df = load_metadata("confusion_matrix", metadata_path)
    
    # Process each experiment
    for _, row in df.iterrows():
        # Create title
        title = (
            f"Confusion Matrix - {row['model_type']}\n"
            f"Dataset: {row['dataset_name']} | "
            f"FE: {row['feature_engineering']} | "
            f"Outliers Removed: {row['has_outliers_removed']}"
        )
        
        # Generate unique prefix for this experiment
        exp_prefix = f"{filename_prefix}_{row['run_id']}" if filename_prefix else row['run_id']
        
        # Update filename generation in the loop
        filename = generate_filename(
            "confusion_matrix_metadata",
            {
                "dataset": row['dataset_name'],
                "model": row['model_type'],
                "fe": row['feature_engineering'],
                "outliers": row['has_outliers_removed']
            },
            prefix=filename_prefix
        )
        
        # Plot confusion matrix
        plot_confusion_matrix(
            conf_matrix_data=row['conf_matrix'],
            display_chart=display_chart,
            output_dir=output_dir,
            print_stats=print_stats,
            filename=filename,
            title=title
        )

    return None


def plot_feature_importance(
    feature_importance_data: List[Dict[str, Any]],
    top_n: int = 20,
    display_chart: bool = True,
    output_dir: str = "charts",
    filename_prefix: Optional[str] = None,
    title: str = "Feature Importance",
) -> None:
    """
    Plot top feature importances.
    
    Parameters:
    - feature_importance_data: List of dicts with 'Feature' and 'Importance' keys
    - top_n: Number of top features to show
    """
    if not feature_importance_data:
        print("Warning: No feature importance data provided.")
        return

    df = pd.DataFrame(feature_importance_data)
    
    if df.empty:
        print("Warning: Feature importance data is empty.")
        return

    # Clean feature names for better display
    df["Feature"] = df["Feature"].apply(lambda x: x.replace("numerical__", "").replace("binary__", "").replace("categorical__", "").replace("_kpis", " KPIs").replace("_", " ").title())
    
    # Sort and take top N by Absolute Value
    df["AbsImportance"] = df["Importance"].abs()
    df = df.sort_values("AbsImportance", ascending=True).tail(top_n)
    
    fig = px.bar(
        df,
        x="Importance",
        y="Feature",
        orientation='h',
        text="Importance"
    )
    
    fig.update_traces(
        texttemplate="<b>%{text:.4f}</b>", # Added formatting
        textposition="outside",
        textfont=dict(size=14, color="#000000"),
        marker_color=COLORS[0],
        marker_line_color="#000000",
        marker_line_width=1.5,
    )
    
    fig = apply_common_style(
        fig,
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        y_values=df["Importance"]
    )
    
    # Ensure y-axis labels are readable and not cut off
    fig.update_layout(
        yaxis=dict(
            tickmode='linear',
            automargin=True
        ),
        margin=dict(l=200) # Increase left margin for long feature names
    )
    
    filename = generate_filename(
        "feature_importance",
        {"top": top_n},
        prefix=filename_prefix
    )
    save_figure(fig, filename, output_dir)
    
    if display_chart:
        fig.show()

def plot_predicted_vs_actual(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    display_chart: bool = True,
    output_dir: str = "charts",
    filename_prefix: Optional[str] = None,
    title: str = "Predicted vs Actual",
) -> None:
    """
    Plot Predicted vs Actual values scatter plot with perfect prediction line.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Create scatter plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(
            color=COLORS[0],
            size=8,
            opacity=0.6,
            line=dict(width=1, color=DEFAULT_CHART_CONFIG["border_color"])
        ),
        name='Predictions'
    ))
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color=COLORS[2], width=3, dash='dash'),
        name='Perfect Prediction'
    ))
    
    fig = apply_common_style(
        fig,
        title=title,
        xaxis_title="Actual Value",
        yaxis_title="Predicted Value",
        y_values=y_pred
    )
    
    # Generate filename
    filename = generate_filename(
        "predicted_vs_actual",
        {},
        prefix=filename_prefix
    )
    save_figure(fig, filename, output_dir)
    
    if display_chart:
        fig.show()


def plot_residuals(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    display_chart: bool = True,
    output_dir: str = "charts",
    filename_prefix: Optional[str] = None,
    title: str = "Residual Analysis",
) -> None:
    """
    Plot Residuals vs Predicted values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_true - y_pred
    
    # Create scatter plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(
            color=COLORS[1],
            size=8,
            opacity=0.6,
            line=dict(width=1, color=DEFAULT_CHART_CONFIG["border_color"])
        ),
        name='Residuals'
    ))
    
    # Add zero line
    min_pred = y_pred.min()
    max_pred = y_pred.max()
    
    fig.add_trace(go.Scatter(
        x=[min_pred, max_pred],
        y=[0, 0],
        mode='lines',
        line=dict(color=COLORS[2], width=3, dash='dash'),
        name='Zero Error'
    ))
    
    fig = apply_common_style(
        fig,
        title=title,
        xaxis_title="Predicted Value",
        yaxis_title="Residuals (Actual - Predicted)",
        y_values=residuals
    )
    
    # Generate filename
    filename = generate_filename(
        "residuals",
        {},
        prefix=filename_prefix
    )
    save_figure(fig, filename, output_dir)
    
    if display_chart:
        fig.show()



def generate_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    columns_to_include: Optional[List[str]] = None,
    column_renames: Optional[Dict[str, str]] = None,
    float_format: str = "%.3f"
) -> str:
    """
    Generate a LaTeX table from a DataFrame using booktabs style.
    """
    if columns_to_include:
        df = df[columns_to_include].copy()
    else:
        df = df.copy()
        
    if column_renames:
        df = df.rename(columns=column_renames)
        
    # Generate LaTeX
    latex_code = df.to_latex(
        index=False,
        float_format=float_format,
        caption=caption,
        label=label,
        position="htbp",
        column_format="l" * len(df.columns), # Left align by default
        bold_rows=False,
    )
    
    # Add booktabs commands if not present (pandas to_latex usually adds them if asked, but let's ensure standard formatting)
    # Note: to_latex is deprecated in favor of style.to_latex but for basic usage it's fine. 
    # Let's clean up the output to be more "thesis-like"
    
    return latex_code



def plot_best_confusion_matrices_from_metadata(
    experiment_df: pd.DataFrame,
    conf_matrix_df: pd.DataFrame,
    problem_types: List[str],
    groupby_columns: List[str],
    metrics: Dict[str, Dict[str, str]],
    display_chart: bool = True,
    output_dir: str = "charts",
    print_stats: bool = False,
    filename_prefix: Optional[str] = None,
) -> None:
    """Create confusion matrix visualizations for the best models."""
    # Label mapping for binary classification
    binary_labels = {"0": "Bankrupt", "1": "Success"}
    
    for problem_type in problem_types:
        if print_stats:
            print(f"\nProcessing {problem_type}...")
            
        problem_exp_df = experiment_df[experiment_df["problem_type"] == problem_type].copy()
        problem_conf_df = conf_matrix_df[conf_matrix_df["problem_type"] == problem_type].copy()
        
        if len(problem_exp_df) == 0 or len(problem_conf_df) == 0:
            print(f"No data found for problem type: {problem_type}")
            continue
            
        metric = metrics[problem_type]["metric"]
        operation = metrics[problem_type]["operation"]
        
        grouped = problem_exp_df.groupby(groupby_columns)
        if operation == "max":
            idx = grouped[metric].idxmax()
        else:
            idx = grouped[metric].idxmin()
        
        best_models = problem_exp_df.loc[idx]
        
        for _, row in best_models.iterrows():
            matrix_row = problem_conf_df[
                (problem_conf_df["dataset_name"] == row["dataset_name"]) &
                (problem_conf_df["has_outliers_removed"] == row["has_outliers_removed"]) &
                (problem_conf_df["feature_engineering"] == row["feature_engineering"]) &
                (problem_conf_df["model_type"] == row["model_type"])
            ]
            
            if len(matrix_row) == 0:
                if print_stats:
                    print(f"No confusion matrix found for configuration: {row.to_dict()}")
                continue
            
            # Format dataset name for title
            dataset_name = row["dataset_name"].replace("_", " ").title()
            
            # Create descriptive title
            title = (
                f"Confusion Matrix for {dataset_name}\n"
                f"{row['model_type']} Model | "
                f"FE: {row['feature_engineering']} | "
                f"{'Outliers Removed' if row['has_outliers_removed'] else 'With Outliers'}"
            )
            
            # Create subtitle with metrics only
            subtitle = (
                f"F1: {row['F1 Score']:.3f} | "
                f"Accuracy: {row['Accuracy']:.3f} | "
                f"Precision: {row['Precision']:.3f} | "
                f"Recall: {row['Recall']:.3f}"
            )
            
            # Get confusion matrix data and map labels if needed
            conf_matrix = matrix_row.iloc[0]["conf_matrix"]
            
            # Determine number of classes from the confusion matrix
            if isinstance(conf_matrix, list):
                num_classes = len(conf_matrix)
                # Map labels if binary classification and labels are "0"/"1"
                if problem_type == "binary_classification":
                    mapped_conf_matrix = []
                    for d in conf_matrix:
                        mapped_conf_matrix.append({
                            binary_labels.get(str(k), k): v for k, v in d.items()
                        })
                    conf_matrix = mapped_conf_matrix
            else:
                num_classes = len(conf_matrix.keys())
            
            # Generate filename using consistent pattern
            filename = generate_filename(
                "matrix",
                {
                    "classes": num_classes,
                    "dataset": row['dataset_name'],
                    "model": row['model_type'],
                    "fe": row['feature_engineering'],
                    "outliers": row['has_outliers_removed']
                },
                prefix=filename_prefix
            )
            
            plot_confusion_matrix(
                conf_matrix_data=conf_matrix,
                display_chart=display_chart,
                output_dir=output_dir,
                print_stats=print_stats,
                filename=filename,
                title=title,
                subtitle=subtitle
            )


def plot_best_regression_results_from_metadata(
    experiment_df: pd.DataFrame,
    problem_types: List[str] = ["regression"],
    groupby_columns: List[str] = ["dataset_name", "has_outliers_removed", "feature_engineering"],
    metrics: Dict[str, Dict[str, str]] = {"regression": {"metric": "MAPE", "operation": "min"}},
    display_chart: bool = True,
    output_dir: str = "charts",
    print_stats: bool = False,
    filename_prefix: Optional[str] = None,
) -> None:
    """Create regression result visualizations for the best models."""
    # Filter for regression problems
    regression_df = experiment_df[experiment_df["problem_type"].isin(problem_types)].copy()
    
    if len(regression_df) == 0:
        print("No regression data found")
        return
    
    # Get best models for each dataset configuration
    metric = metrics["regression"]["metric"]
    operation = metrics["regression"]["operation"]
    
    if print_stats:
        print(f"\n🔍 Finding best models using {metric} ({operation})")
    
    grouped = regression_df.groupby(groupby_columns)
    if operation == "max":
        idx = grouped[metric].idxmax()
    else:
        idx = grouped[metric].idxmin()
    
    best_models = regression_df.loc[idx]
    
    if print_stats:
        print(f"\n📈 Found {len(best_models)} best models:")
        for _, row in best_models.iterrows():
            print(f"\n• Dataset: {row['dataset_name']}")
            print(f"  Model: {row['model_type']}")
            print(f"  MAPE: {row['MAPE']*100:.1f}%")
            print(f"  MAE: ${row['MAE']:,.0f}")
            print(f"  R²: {row['R2']:.3f}")
    
    for _, row in best_models.iterrows():
        if pd.isna(row.get("predicted_vs_actual")):
            if print_stats:
                print(f"\n⚠️ No prediction data found for {row['dataset_name']}")
            continue
        
        # Format dataset name
        dataset_name = row["dataset_name"].replace("_", " ").title()
        
        # Create title
        title = (
            f"{dataset_name} Revenue Prediction Results\n"
            f"{'with outliers removed' if row['has_outliers_removed'] else 'with all data'} "
            f"and {row['feature_engineering']} feature engineering"
        )
        
        # Create metrics text
        metrics_text = (
            f"Model: {row['model_type']} | "
            f"MAE: ${row['MAE']:,.0f} | "
            f"MAPE: {row['MAPE']*100:.1f}% | "
            f"R²: {row['R2']:.3f}"
        )
        
        # Update filename generation to match standard pattern
        filename = generate_filename(
            "best_regression_results",
            {
                "dataset": row['dataset_name'],
                "model": row['model_type'],
                "fe": row['feature_engineering'],
                "outliers": row['has_outliers_removed'],
                "mape": f"{row['MAPE']*100:.1f}",
                "r2": f"{row['R2']:.3f}"
            },
            prefix=filename_prefix
        )
        
    plot_regression_results(
        predicted_vs_actual=row["predicted_vs_actual"],
        display_chart=display_chart,
        output_dir=output_dir,
        print_stats=print_stats,
        filename_prefix=filename_prefix,
        title=title,
        subtitle=metrics_text,
        dataset_name=row['dataset_name']
    )


def plot_regression_results(
    predicted_vs_actual: str,
    display_chart: bool = True,
    output_dir: str = "charts",
    print_stats: bool = False,
    filename_prefix: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> None:
    """Format regression results into a LaTeX table and save it."""
    # Parse the JSON string into a dictionary
    data = json.loads(predicted_vs_actual)
    if isinstance(data, str):
        data = json.loads(data)
    
    # Create a cleaner DataFrame for the thesis LaTeX table
    stats_data = [
        ["Mean Absolute Error (MAE)", f"${data['absolute_error']['mean']:,.2f}"],
        ["Mean Absolute Percentage Error (MAPE)", f"{data['absolute_percentage_error']['mean']*100:.2f}\\%"],
        ["Mean Squared Error (MSE)", f"{data['squared_error']['mean']:.2e}"],
        ["Actual Mean Revenue", f"${data['actual']['mean']:,.2f}"],
        ["Predicted Mean Revenue", f"${data['predicted']['mean']:,.2f}"],
        ["Actual Median Revenue", f"${data['actual']['50%']:,.2f}"],
        ["Predicted Median Revenue", f"${data['predicted']['50%']:,.2f}"],
        ["Revenue Range (Actual)", f"${data['actual']['min']:,.2f} to ${data['actual']['max']:,.2f}"],
        ["Revenue Range (Predicted)", f"${data['predicted']['min']:,.2f} to ${data['predicted']['max']:,.2f}"],
        ["Actual Std Dev", f"${data['actual']['std']:,.2f}"],
        ["Predicted Std Dev", f"${data['predicted']['std']:,.2f}"],
    ]
    
    df_latex = pd.DataFrame(stats_data, columns=["Metric", "Value"])
    
    # Save as LaTeX table
    if dataset_name:
        table_filename = f"table_reg_results_{dataset_name}.tex"
        caption = f"Regression Detailed Results for {dataset_name.replace('_', ' ').title()}"
        label = f"tab:reg_results_{dataset_name}"
        
        # We need a way to save this. Since this function is in support_functions, 
        # it might not have access to save_latex from the main script.
        # Let's return the LaTeX string or just print it if we can't save it directly.
        # But wait, _eda_4_analysis.py has a save_latex function.
        # Actually, let's just generate the latex string and print a message.
        
        latex_str = generate_latex_table(df_latex, caption=caption, label=label)
        
        # Determine tables directory (relative to code/)
        tables_dir = os.path.join("thesis_assets", "tables")
        os.makedirs(tables_dir, exist_ok=True)
        
        with open(os.path.join(tables_dir, table_filename), "w") as f:
            f.write(latex_str)
            
        if print_stats:
            print(f"Saved LaTeX regression results to {os.path.join(tables_dir, table_filename)}")

    if print_stats:
        print("\n📊 Model Performance Summary (Regression):")
        print(f"• Average Error: ${data['absolute_error']['mean']:,.0f}")
        print(f"• MAPE: {data['absolute_percentage_error']['mean']*100:.2f}%")
