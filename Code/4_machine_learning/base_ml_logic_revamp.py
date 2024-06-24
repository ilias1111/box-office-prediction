import os
import json
import logging
from sklearn.utils import class_weight
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
import joblib

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/model_performance_{timestamp}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CV_FOLDS = 5
GRID_TYPE = 'grid_search'
DATA_FILE_PATH = "./data/ml_ready_data/full_dataset.csv"
TARGET_COLUMN_NAME = "binary_classification"
ID_COLUMN_NAME = "movie_id"
OPTIMIZE_FOR_METRIC = "roc_auc"
MODELS = ["logistic_regression", "random_forest", "support_vector_machine", "decision_tree"]

def load_param_grids(file_path, model_name, grid_type):
    logging.info(f"Loading parameter grids for {model_name} with {grid_type}")
    with open(file_path, 'r') as file:
        all_grids = json.load(file)
    param_grid = all_grids[model_name].get(grid_type, {})

    # Replace scaler names with actual scaler objects
    if 'preprocessor__num__scaler' in param_grid:
        scaler_mapping = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler()
        }
        param_grid['preprocessor__num__scaler'] = [scaler_mapping[scaler] for scaler in param_grid['preprocessor__num__scaler']]
    
    return param_grid

def model_factory(model_type):
    logging.info(f"Creating model for {model_type}")
    models = {
        "logistic_regression": LogisticRegression(random_state=42, n_jobs=-1),
        "random_forest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "support_vector_machine": SVC(probability=True, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42)
        ##"balanced_random_forest": BalancedRandomForestClassifier(random_state=42, n_jobs=-1)
    }
    return models[model_type]

def create_preprocessor(numeric_columns, categorical_columns):
    logging.info("Creating preprocessor")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numeric_columns),
            ("cat", Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_columns)
        ]
    )
    return preprocessor

def load_and_preprocess_data(file_path, target_column_name, id_column_name):
    logging.info(f"Loading and preprocessing data from {file_path}")
    data = pd.read_csv(file_path)
    data.drop(id_column_name, axis=1, inplace=True)
    y = data[target_column_name].values
    X = data.drop(target_column_name, axis=1)

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    return X, y, numerical_features, categorical_features

def evaluate_model(model, X_valid, y_valid):
    logging.info("Evaluating model")
    pred_classes = model.predict(X_valid)
    pred_probs = model.predict_proba(X_valid)[:, 1]

    metrics = {
        "ROC AUC Score": roc_auc_score(y_valid, pred_probs),
        "Accuracy": accuracy_score(y_valid, pred_classes),
        "Precision": precision_score(y_valid, pred_classes, zero_division=0, pos_label='Success'),
        "Recall": recall_score(y_valid, pred_classes, zero_division=0, pos_label='Success'),
        "F1 Score": f1_score(y_valid, pred_classes, zero_division=0, pos_label='Success'),
    }

    conf_matrix = pd.DataFrame(
        confusion_matrix(y_valid, pred_classes),
        index=["Actual: No", "Actual: Yes"],
        columns=["Predicted: No", "Predicted: Yes"],
    )
    return metrics, conf_matrix

def get_feature_importance(model, preprocessor):
    """Generate and return a DataFrame containing feature importances."""
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame(
            {"Feature": feature_names, "Importance": model.feature_importances_}
        ).sort_values(by="Importance", ascending=False)
    elif hasattr(model, "coef_"):
        feature_importance = pd.DataFrame(
            {"Feature": feature_names, "Importance (coef_)": model.coef_[0]}
        ).sort_values(by="Importance (coef_)", ascending=False)
    else:
        return pd.DataFrame()  # Return empty DataFrame if conditions not met
    return feature_importance

def save_model_and_metadata(model, metrics, conf_matrix, model_type, grid_type, feature_importance):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{timestamp}_{model_type}.pkl"
    metadata_filename = f"{timestamp}_{model_type}.json"

    model_directory = "models"
    metadata_directory = "metadata"

    os.makedirs(model_directory, exist_ok=True)
    os.makedirs(metadata_directory, exist_ok=True)

    joblib.dump(model, os.path.join(model_directory, model_filename))
    scaler = str(model.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler'])
    

    metadata = {
        "timestamp": timestamp,
        "model_type": model_type,
        "grid_type": grid_type,
        "scaler" : scaler.__class__.__name__,
        "model_parameters": model.named_steps["classifier"].get_params(),
        "performance_metrics": metrics,
        "conf_matrix" : conf_matrix.to_dict(orient='records'),
        "feature_importance": feature_importance.to_dict(orient='records')
    }
    
    logging.info(f"Saving model parameters \n {model.named_steps["classifier"].get_params()}")

    with open(os.path.join(metadata_directory, metadata_filename), "w") as f:
        json.dump(metadata, f, indent=4)

    logging.info(f"Model and metadata saved. Model: {model_filename}, Metadata: {metadata_filename}")

def main(file_path, target_column_name, id_column_name, model_type, grid_type='grid_search'):
    logging.info(f"Running main for model {model_type} with grid type {grid_type}")
    X, y, numerical_features, categorical_features =  load_and_preprocess_data(file_path, target_column_name, id_column_name)
    param_grid = load_param_grids('param_grids.json', model_type, grid_type)

    model = Pipeline([
        ("preprocessor", create_preprocessor(numerical_features, categorical_features)),
        ("classifier", model_factory(model_type))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

# Create a dictionary with class names as keys
    class_weight_dict = {label: weight for label, weight in zip(np.unique(y_train), class_weights)}

    param_grid['classifier__class_weight'] = [class_weight_dict]

    if grid_type != 'non_grid':
        grid_search = GridSearchCV(model, param_grid, cv=CV_FOLDS, scoring=OPTIMIZE_FOR_METRIC, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = model
        classifier_params = {k: v[0] for k, v in param_grid.items() if k.startswith('classifier__')}
        best_model.set_params(**classifier_params)
        best_model.fit(X_train, y_train)

    metrics, conf_matrix = evaluate_model(best_model, X_test, y_test)
    feature_importance = get_feature_importance(best_model.named_steps['classifier'], best_model.named_steps['preprocessor'])

    
    logging.info('For model:', model_type)
    print(model_type)
    logging.info(metrics)
    logging.info(conf_matrix)
    logging.info(feature_importance)


    save_model_and_metadata(best_model, metrics, conf_matrix, model_type, grid_type, feature_importance)

    return model_type, metrics

if __name__ == "__main__":
    results = []

    logging.info("Starting model training script")
    for model in MODELS:
        model, metrics = main(DATA_FILE_PATH, TARGET_COLUMN_NAME, ID_COLUMN_NAME, model, GRID_TYPE)
        results.append(
            {
                "Model": model,
                **metrics,  # Unpack the metrics dictionary directly into the row
            }
        )
    
    logging.info("All models trained. Final results:")
    logging.info(pd.DataFrame(results).to_string())
    print(pd.DataFrame(results))
