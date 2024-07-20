import pandas as pd
import numpy as np
import shap
import plotly.express as px
import os
import json
from glob import glob
from datetime import datetime

# def shap_calculation(model, X_test):

#     transformer = model.named_steps["preprocessor"]
#     estimator = model.named_steps["model"]

#     X_test_transformed = transformer.transform(X_test)
#     X_sample = shap.utils.sample(
#         X_test_transformed, int(X_test_transformed.shape[0] / 10)
#     )  # Adjust sample size based on needs
#     explainer = None
#     shap_values = None

#     if hasattr(estimator, "predict_proba"):
#         explainer = shap.Explainer(estimator.predict_proba, X_sample, algorithm="auto")
#     else:
#         explainer = shap.Explainer(estimator.predict, X_sample, algorithm="auto")

#     return None

def plot_one_metric_of_different_datasets_per_feature_engineering_outliers_with_plotly(
    experiment_df,
    problem_type="binary_classification",
    metric="F1 Score",
    metric_agg="max",
):
    experiment_df = experiment_df[experiment_df["problem_type"] == problem_type]

    feature_engineering_order = ["none", "simple", "complex"]
    has_outliers_removed_order = [
        False,
        True,
    ]

    # Group the data by dataset name, outliers removed, and feature engineering
    grouped_data = (
        experiment_df.groupby(
            ["dataset_name", "has_outliers_removed", "feature_engineering"]
        )
        .agg({metric: metric_agg})
        .reset_index()
    )

    # Iterate over each dataset and create a plot
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
            title=f"{metric} for {dataset_name}",
            labels={
                "feature_engineering": "Feature Engineering Type",
                "has_outliers_removed": "Outliers Removed",
                metric: f"{metric} Value",
            },
            color_discrete_sequence=px.colors.qualitative.Prism,
            barmode="group",
        )

        fig.update_traces(texttemplate="%{text:.2s}", textposition="inside")
        fig.update_layout(showlegend=True, legend_title_text="Outliers Removed")

        fig.show()


def load_metadata(type: str, metadata_path: str) -> pd.DataFrame:

    json_files = glob(os.path.join(metadata_path, "*.json"))

    results = []

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)

            if type == "metrics":
                parametric_data = data.get("metrics", {})
            elif type == "parameters":
                parametric_data = data.get("model_parameters", {})
            else:
                parametric_data = {}

            experiment_info = {
                "run_id": data.get("run_id", None),
                "timestamp": datetime.strptime(data.get("timestamp", None), "%Y%m%d_%H%M%S"),
                "model_type": data.get("model_type", None),
                "problem_type": data.get("problem_type", None),
                "dataset_name": data.get("dataset_name", None),
                "grid_type": data.get("grid_type", None),
                "has_outliers_removed": data.get("has_outliers_removed", None),
                "feature_engineering": data.get("feature_engineering", None),
                "scaler": data.get("scaler", None),
                "variance_threshold": data.get("variance_threshold", None),
                "duration": data.get("duration", None),
                "number_of_combinations": data.get("number_of_combinations", None),
                **parametric_data,
            }
            results.append(experiment_info)

    results_df = pd.DataFrame(results)

    return results_df

def plot_and_export_categorical_distribution(df, agg_column, other_threshold, axis_type = 'log', sort_by_value=True):
    counts = df[agg_column].value_counts().reset_index()
    counts.columns = [agg_column, 'count']

    counts['grouped'] = counts.apply(lambda x: 'Other' if x['count'] < other_threshold else x[agg_column], axis=1)

    grouped_counts = counts.groupby('grouped').agg(total=('count', 'sum')).reset_index()
    if sort_by_value:
        grouped_counts = grouped_counts.sort_values("total", ascending=False).reset_index(drop=True)
    else:
        grouped_counts = grouped_counts.sort_values("grouped", ascending=True).reset_index(drop=True)
    
    grouped_counts["grouped"] = grouped_counts["grouped"].astype(str)


    color = {k: 'rgba(56, 166, 165, 1)' if k != 'Other' else 'rgba(204, 80, 62, 1)' for k in grouped_counts['grouped'].unique()}


    fig = px.bar(
        grouped_counts,
        x='grouped',
        y='total',
        title=f'Count of Movies by {agg_column}',
        labels={'grouped': f'{agg_column}', 'total': 'Number of Movies'},
        text='total',
        color='grouped',
        color_discrete_map=color,
        height=600
    )

    fig.update_layout(
        xaxis_title=f"{agg_column}",
        yaxis_title="Number of Movies",
        yaxis_type=axis_type,
        xaxis_tickangle=-45,
        xaxis_tickfont_size=12,
        yaxis_tickfont_size=12,
        showlegend=False
    )
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    fig.show()

    return grouped_counts
