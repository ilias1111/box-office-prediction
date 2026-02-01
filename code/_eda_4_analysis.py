import pandas as pd
import numpy as np
import json
import os
import plotly.io as pio
from eda_support_functions import *
import joblib
from sklearn.model_selection import train_test_split
import sys
import re
from glob import glob

# Add 4_machine_learning to path to allow unpickling models that reference utils
# Assuming this script is run from Code/ directory
sys.path.append(os.path.abspath("4_machine_learning"))

# Set pandas options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_info_columns", 1000)

# Ensure charts directory exists
OUTPUT_DIR = "thesis_assets"
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# Set plotly renderer to output images
pio.renderers.default = "notebook" 

# --- CONFIGURATION ---
# If RUNS_IDS is populated, the analysis will focus ONLY on these runs.
# If empty, it will consider ALL runs.
RUNS_IDS = [
    # "20240905_183321",
    # "20240905_231518",
]

# --- LOAD METADATA ---
metadata_path = "../metadata" # Adjust path if running from Code/
if not os.path.exists(metadata_path):
    metadata_path = "metadata" # Try local if running from root
    
if not os.path.exists(metadata_path):
    # Fallback to absolute path based on user environment
    metadata_path = "/Users/iliasx/Documents/GitHub/box-office-prediction/metadata"

print(f"Loading metadata from: {metadata_path}")

parameters_df = load_metadata("parameters", metadata_path)
experiment_df = load_metadata("metrics", metadata_path)
conf_matrix_df = load_metadata("confusion_matrix", metadata_path)

print(f"Total experiments loaded: {len(experiment_df)}")

# --- FILTERING ---
if RUNS_IDS:
    print(f"\nFiltering for {len(RUNS_IDS)} specific runs: {RUNS_IDS}")
    parameters_df = parameters_df[parameters_df["run_id"].isin(RUNS_IDS)]
    experiment_df = experiment_df[experiment_df["run_id"].isin(RUNS_IDS)]
    conf_matrix_df = conf_matrix_df[conf_matrix_df["run_id"].isin(RUNS_IDS)]
    print(f"Filtered to {len(experiment_df)} experiments.")

# --- SUMMARY STATS (Notebook Cell 6) ---
print("\n--- Summary Statistics ---")
try:
    summary_stats = experiment_df.groupby("run_id").aggregate(
        {
            "run_id": "count",
            "dataset_name": "nunique",
            "model_type": "nunique",
            "timestamp": "max",
            "duration": "sum"
        }
    ).rename(columns={"run_id": "count"}).sort_values("timestamp", ascending=False)
    print(summary_stats)
except Exception as e:
    print(f"Could not generate summary stats: {e}")

# --- DETAILED PARAMETERS (Notebook Cell 8) ---
print("\n--- Detailed Parameters ---")
param_cols = [
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
# Only show columns that exist in the dataframe
valid_cols = [c for c in param_cols if c in parameters_df.columns]
if not parameters_df.empty:
    print(parameters_df[valid_cols].sort_values(by=["model_type", "problem_type", "dataset_name"]))

# --- XGBOOST DEEP DIVE (Notebook Cell 9) ---
if "xgboost_regressor" in parameters_df["model_type"].values:
    print("\n--- XGBoost Details ---")
    xgb_cols = [
        "timestamp", "problem_type", "dataset_name", "has_outliers_removed",
        "feature_engineering", "model_type", "scaler", "variance_threshold",
        "n_estimators", "max_depth", "learning_rate"
    ]
    valid_xgb_cols = [c for c in xgb_cols if c in parameters_df.columns]
    print(parameters_df[parameters_df["model_type"] == "xgboost_regressor"][valid_xgb_cols])


# --- BEST MODEL SELECTION (Notebook Cell 11) ---
print("\n--- Selecting Best Models ---")
problem_types = ["regression", "binary_classification", "multi_class_classification"]
groupby_columns = ["dataset_name", "has_outliers_removed", "feature_engineering"]

metrics = {
    "regression": {"metric": "MAPE", "operation": "min"},
    "binary_classification": {"metric": "F1 Score", "operation": "max"},
    "multi_class_classification": {"metric": "F1 Score", "operation": "max"},
}

display_columns = {
    "regression": [
        "model_type", "scaler", "MAPE", "Threshold Probability Accuracy", "R2", "duration", "run_id"
    ],
    "binary_classification": [
        "model_type", "scaler", "F1 Score", "Accuracy", "Precision", "Recall", "duration", "run_id"
    ],
    "multi_class_classification": [
        "model_type", "scaler", "F1 Score", "Accuracy", "Precision", "Recall", "duration", "run_id"
    ],
}

best_models = select_best_models(
    experiment_df, problem_types, groupby_columns, metrics, display_columns
)

result_reg = best_models["regression"]
result_class = best_models["binary_classification"]
result_multi_class = best_models["multi_class_classification"]

print("\nBest Regression Models:")
if not result_reg.empty:
    print(result_reg.to_string(index=False, justify="center", float_format="%.2f"))

print("\nBest Binary Classification Models:")
if not result_class.empty:
    print(result_class.to_string(index=False, justify="center", float_format="%.2f"))

print("\nBest Multi-class Classification Models:")
if not result_multi_class.empty:
    print(result_multi_class.to_string(index=False, justify="center", float_format="%.2f"))


# --- EXPORT TABLES TO TEX ---
def save_latex(df, filename, caption, label):
    if df.empty:
        return
    path = os.path.join(TABLES_DIR, filename)
    latex = generate_latex_table(df, caption=caption, label=label)
    with open(path, "w") as f:
        f.write(latex)
    print(f"Saved LaTeX table to {path}")

save_latex(result_reg, "table_best_reg.tex", "Best Regression Models", "tab:best_reg")
save_latex(result_class, "table_best_bin_class.tex", "Best Binary Classification Models", "tab:best_bin_class")


# --- PLOTS (Notebook Cells 15-23) ---
print("\n--- Generating Plots ---")

# 1. Dataset Comparison Plots
print("Generating Dataset Comparison Plots...")
try:
    plot_one_metric_of_different_datasets_per_feature_engineering_outliers_with_plotly(
        experiment_df,
        problem_type="binary_classification",
        metric="F1 Score",
        metric_agg="max",
        print_stats=True,
        output_dir=CHARTS_DIR,
        filename_prefix="step_4",
        display_chart=False 
    )

    plot_one_metric_of_different_datasets_per_feature_engineering_outliers_with_plotly(
        experiment_df,
        problem_type="multi_class_classification",
        metric="F1 Score",
        metric_agg="max",
        print_stats=True,
        output_dir=CHARTS_DIR,
        filename_prefix="step_4",
        display_chart=False 
    )

    plot_one_metric_of_different_datasets_per_feature_engineering_outliers_with_plotly(
        experiment_df, 
        problem_type="regression", 
        metric="MAPE", 
        metric_agg="min", # MAPE should be minimized
        print_stats=True,
        output_dir=CHARTS_DIR,
        filename_prefix="step_4",
        display_chart=False 
    )

    plot_one_metric_of_different_datasets_per_feature_engineering_outliers_with_plotly(
        experiment_df, 
        problem_type="regression", 
        metric="R2", 
        metric_agg="max",
        print_stats=True,
        output_dir=CHARTS_DIR,
        filename_prefix="step_4",
        display_chart=False 
    )
except Exception as e:
    print(f"Error generating dataset comparison plots: {e}")

# 2. Model Comparison Plots (with Benchmarks)
print("Generating Model Comparison Plots...")
try:
    plot_one_metric_of_different_models_per_dataset_with_plotly(
        experiment_df,
        problem_type="regression",
        metric="R2",
        metric_agg="max",
        benchmark_model="dummy_regressor",
        print_stats=True,
        output_dir=CHARTS_DIR,
        filename_prefix="step_4",
        display_chart=False 
    )

    plot_one_metric_of_different_models_per_dataset_with_plotly(
        experiment_df,
        problem_type="regression",
        metric="MAPE",
        metric_agg="min",
        benchmark_model="dummy_regressor",
        print_stats=True,
        output_dir=CHARTS_DIR,
        filename_prefix="step_4",
        display_chart=False 
    )

    plot_one_metric_of_different_models_per_dataset_with_plotly(
        experiment_df,
        problem_type="binary_classification",
        metric="F1 Score",
        metric_agg="max",
        benchmark_model="dummy_classifier",
        print_stats=True,
        output_dir=CHARTS_DIR,
        filename_prefix="step_4",
        display_chart=False 
    )
except Exception as e:
    print(f"Error generating model comparison plots: {e}")

# 3. Best Results Detailed Visualization
print("Generating Detailed Results...")
# Confusion Matrices
try:
    plot_best_confusion_matrices_from_metadata(
        metadata_path=metadata_path, 
        display_chart=False,
        output_dir=CHARTS_DIR,
        print_stats=True,
        filename_prefix="step_4"
    )
except Exception as e:
    print(f"Error generating confusion matrices: {e}")

# Best Regression Results (Predicted vs Actual, Residuals from Metadata)
try:
    plot_best_regression_results_from_metadata(
        experiment_df=experiment_df,
        display_chart=False, # Don't open browser
        output_dir=CHARTS_DIR,
        print_stats=True,
        filename_prefix="step_4"
    )
except Exception as e:
    print(f"Error generating best regression results: {e}")


# --- REGENERATION & FEATURE IMPORTANCE ---
# Kept from original script for deep-dive charts if needed

def load_specific_run_data(run_id, metadata_path):
    files = glob(os.path.join(metadata_path, f"*{run_id}.json"))
    if not files: return {}
    try:
        with open(files[0], 'r') as f: return json.load(f)
    except: return {}

# Combined list of best models to iterate over for advanced plots
all_best_models = pd.concat([result_reg, result_class, result_multi_class])

print("\n--- Generating Feature Importance & Extra Plots ---")
for idx, row in all_best_models.iterrows():
    run_id = row['run_id']
    dataset = row['dataset_name']
    model_type = row['model_type']
    
    # Try to load full data for this run from metadata first (fastest)
    run_data_full = load_specific_run_data(run_id, metadata_path)

    # 1. Feature Importance
    if run_data_full.get("feature_importance"):
        print(f"Plotting Feature Importance for {model_type} ({dataset}) - {run_id}")
        plot_feature_importance(
            run_data_full["feature_importance"],
            top_n=20,
            output_dir=CHARTS_DIR,
            filename_prefix=f"imp_{run_id}",
            title=f"Feature Importance - {model_type} ({dataset})",
            display_chart=False
        )
    else:
        # Could skip or try load model. Skipping for speed/consistency with notebook.
        pass
