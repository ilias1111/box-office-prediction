
# Cell 0
import pandas as pd
from eda_support_functions import *

# Cell 1
pd.set_option("display.max_columns", None)
pd.set_option("display.max_info_columns", 1000)

# Cell 2
RUNS_IDS = ["20240905_183321","20240905_231518"]

# Cell 3
parameters_df = load_metadata(
    "parameters", "/Users/iliasx/Documents/GitHub/box-office-prediction/metadata"
)
experiment_df = load_metadata(
    "metrics", "/Users/iliasx/Documents/GitHub/box-office-prediction/metadata"
)
conf_matrix_df = load_metadata(
    "confusion_matrix", "/Users/iliasx/Documents/GitHub/box-office-prediction/metadata"
)


# Cell 4
conf_matrix_df

# Cell 5
experiment_df

# Cell 6
experiment_df.groupby("run_id").aggregate(
    {
        "run_id": "count",
        "dataset_name": "nunique",
        "model_type": "nunique",
        "timestamp": "max",
        "duration": "sum"
    }
).rename(columns={"run_id": "count"}).sort_values("timestamp", ascending=False)

# Cell 7
parameters_df = parameters_df[parameters_df["run_id"].isin(RUNS_IDS)]
experiment_df = experiment_df[experiment_df["run_id"].isin(RUNS_IDS)]
conf_matrix_df = conf_matrix_df[conf_matrix_df["run_id"].isin(RUNS_IDS)]

# Cell 8
parameters_df[
    [
        "problem_type",
        "dataset_name",
        "has_outliers_removed",
        "feature_engineering",
        "model_type",
        "scaler",
        "variance_threshold",
        "class_weight",
        "C",
        "min_child_samples",
        "solver",
        "num_leaves",
        "n_estimators",
    ]
].sort_values(by=["model_type", "problem_type", "dataset_name", "has_outliers_removed"])

# Cell 9
parameters_df[parameters_df["model_type"] == "xgboost_regressor"][
    [
        "timestamp",
        "problem_type",
        "dataset_name",
        "has_outliers_removed",
        "feature_engineering",
        "model_type",
        "scaler",
        "variance_threshold",
        "n_estimators",
        "max_depth",
        "learning_rate"
    ]
]

# Cell 10
display(experiment_df.sort_values(by="timestamp", ascending=True).tail(10))


# Cell 11
problem_types = ["regression", "binary_classification", "multi_class_classification"]
groupby_columns = ["dataset_name", "has_outliers_removed", "feature_engineering"]
metrics = {
    "regression": {"metric": "MAPE", "operation": "min"},
    "binary_classification": {"metric": "F1 Score", "operation": "max"},
    "multi_class_classification": {"metric": "F1 Score", "operation": "max"},
}
display_columns = {
    "regression": [
        "model_type",
        "scaler",
        "MAPE",
        "Threshold Probability Accuracy",
        "R2",
        "duration",
    ],
    "binary_classification": ["model_type", "scaler", "F1 Score", 
                              "Accuracy","Precision","Recall",
                              "duration"],
    "multi_class_classification": ["model_type", "scaler", "F1 Score", 
                                   "Accuracy","Precision","Recall",
                                   "duration"],
}

best_models = select_best_models(
    experiment_df, problem_types, groupby_columns, metrics, display_columns
)

# Access results
result_reg = best_models["regression"]
result_class = best_models["binary_classification"]
result_multi_class = best_models["multi_class_classification"]

# Cell 12
print(
    result_reg.to_string(
        index=False, justify="center", float_format="%.2f", decimal="."
    )
)

# Cell 13
print(
    result_class.to_string(
        index=False, justify="center", float_format="%.2f", decimal="."
    )
)

# Cell 14
print(
    result_multi_class.to_string(
        index=False, justify="center", float_format="%.2f", decimal="."
    )
)

# Cell 15
plot_one_metric_of_different_datasets_per_feature_engineering_outliers_with_plotly(
    experiment_df,
    problem_type="binary_classification",
    metric="F1 Score",
    metric_agg="max",
    print_stats=True,
    filename_prefix="step_4",
)

# Cell 16
plot_one_metric_of_different_datasets_per_feature_engineering_outliers_with_plotly(
    experiment_df,
    problem_type="multi_class_classification",
    metric="F1 Score",
    metric_agg="max",
    print_stats=True,
        filename_prefix="step_4"

)

# Cell 17
plot_one_metric_of_different_datasets_per_feature_engineering_outliers_with_plotly(
    experiment_df, problem_type="regression", metric="MAPE", metric_agg="min",
    print_stats=True,
        filename_prefix="step_4"

)

# Cell 18
plot_one_metric_of_different_datasets_per_feature_engineering_outliers_with_plotly(
    experiment_df, problem_type="regression", metric="R2", metric_agg="max",
    print_stats=True,
        filename_prefix="step_4"

)

# Cell 19
plot_one_metric_of_different_models_per_dataset_with_plotly(
    experiment_df,
    problem_type="regression",
    metric="R2",
    metric_agg="max",
    benchmark_model="dummy_regressor",
    print_stats=True,
        filename_prefix="step_4"

)

# Cell 20
plot_one_metric_of_different_models_per_dataset_with_plotly(
    experiment_df,
    problem_type="regression",
    metric="MAPE",
    metric_agg="min",
    benchmark_model="dummy_regressor",
    print_stats=True,
        filename_prefix="step_4"

)

# Cell 21
plot_one_metric_of_different_models_per_dataset_with_plotly(
    experiment_df,
    problem_type="binary_classification",
    metric="F1 Score",
    metric_agg="max",
    benchmark_model="dummy_classifier",
    print_stats=True,
        filename_prefix="step_4"

)

# Cell 22
# Load the data

# Example usage
problem_types = ["binary_classification", "multi_class_classification"]
groupby_columns = ["dataset_name", "has_outliers_removed", "feature_engineering"]
metrics = {
    "binary_classification": {"metric": "F1 Score", "operation": "max"},
    "multi_class_classification": {"metric": "F1 Score", "operation": "max"},
}

plot_best_confusion_matrices_from_metadata(
    experiment_df=experiment_df,
    conf_matrix_df=conf_matrix_df,
    problem_types=problem_types,
    groupby_columns=groupby_columns,
    metrics=metrics,
    display_chart=True,
    print_stats=True,
    filename_prefix="step_4"
)

# Cell 23
plot_best_regression_results_from_metadata(
    experiment_df=experiment_df,
    display_chart=True,
    print_stats=True,
    filename_prefix="step_4"
)

# Cell 24

