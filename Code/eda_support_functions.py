import pandas as pd
import numpy as np
import shap
import plotly.express as px
import os
import json
from glob import glob
from datetime import datetime
import plotly.graph_objects as go

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

    feature_engineering_order = ["none", "complex"]
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

    ratio = grouped_data.groupby("dataset_name").size().max() / grouped_data.groupby("dataset_name").size().min()



    # Iterate over each dataset and create a plot
    for dataset_name, data in grouped_data.groupby("dataset_name"):

        ratio = data[metric].max() / data[metric].min()

        if ratio > 15:
            scale_log = True
        else:
            scale_log = False
    
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
            log_y=scale_log
        )

        fig.update_traces(texttemplate="%{text:.2s}", textposition="inside")
        fig.update_layout(showlegend=True, legend_title_text="Outliers Removed")

        fig.show()


def plot_one_metric_of_different_models_per_dataset_with_plotly(
    experiment_df,
    problem_type="binary_classification",
    metric="F1 Score",
    metric_agg="max",
):
    experiment_df = experiment_df[experiment_df["problem_type"] == problem_type]

    # Define the order we want
    feature_engineering_order = ["none", "complex"]
    has_outliers_removed_order = [False, True]  # False first for "with outliers"

    # Create a list of all combinations in the desired order
    group_order = [f"{fe}_{str(out)}" for fe in feature_engineering_order for out in has_outliers_removed_order]

    # Group the data
    grouped_data = (
        experiment_df.groupby(
            ["dataset_name", "feature_engineering", "has_outliers_removed", "model_type"]
        )
        .agg({metric: metric_agg})
        .reset_index()
    )

    # Create the group column
    grouped_data['group'] = grouped_data['feature_engineering'] + '_' + grouped_data['has_outliers_removed'].astype(str)

    # Get unique datasets and models
    datasets = grouped_data['dataset_name'].unique()
    models = grouped_data['model_type'].unique()

    # Create a separate chart for each dataset
    for dataset in datasets:
        dataset_data = grouped_data[grouped_data['dataset_name'] == dataset]
        
        fig = go.Figure()
        
        for model in models:
            model_data = dataset_data[dataset_data['model_type'] == model]
            model_data = model_data.set_index('group').reindex(group_order).reset_index()
            
            fig.add_trace(
                go.Bar(
                    name=model,
                    x=model_data['group'],
                    y=model_data[metric],
                    text=model_data[metric].round(2),
                    textposition='auto',
                    marker_color=px.colors.qualitative.Plotly[models.tolist().index(model)]
                )
            )

        # Update layout
        fig.update_layout(
            title_text=f"{metric} Comparison for {dataset}",
            xaxis_title="Feature Engineering and Outlier Removal",
            yaxis_title=f"{metric} Value",
            barmode='group'
        )

        # Update x-axis labels
        new_labels = {
            "none_False": "FE: none<br>With Outliers",
            "none_True": "FE: none<br>No Outliers",
            "complex_False": "FE: complex<br>With Outliers",
            "complex_True": "FE: complex<br>No Outliers"
        }
        fig.update_xaxes(ticktext=list(new_labels.values()), tickvals=list(new_labels.keys()))

        # Check if log scale is needed
        ratio = dataset_data[metric].max() / dataset_data[metric].min()
        if ratio > 15:
            fig.update_layout(yaxis_type="log")

        fig.show()

    return None  # Since we're showing each figure individually


def load_metadata(type: str, metadata_path: str) -> pd.DataFrame:

    json_files = glob(os.path.join(metadata_path, "*.json"))

    results = []
    experiment_info = {}

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
                    #list of dictionaries to one dictionary
                    

                experiment_info = {
                    **experiment_info,
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
        except Exception as e:
            print(f"Error loading file {file}: {e}")
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
