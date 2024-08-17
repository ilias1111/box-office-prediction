import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import math
import pandas as pd
import joblib
from sklearn.utils import class_weight
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from datetime import datetime
from sklearn.base import is_classifier, is_regressor, BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler, LabelEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
# dummy regressor
from sklearn.dummy import DummyRegressor
# dummy classifier
from sklearn.dummy import DummyClassifier

from random_search import perform_random_search
from utils import threshold_mape, threshold_probability_accuracy, log10_threshold_probability_accuracy
import shap
from sklearn.metrics import (
    make_scorer,
    confusion_matrix, classification_report,
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, root_mean_squared_error, mean_squared_log_error, root_mean_squared_log_error,
    mean_absolute_error, mean_absolute_percentage_error, r2_score
)
from scikeras.wrappers import KerasClassifier, KerasRegressor

# suppress warnings UndefinedMetricWarning
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

class MOTR:
    # MOTR : Model Optimization and Training Rig
    def __init__(self,run_id, file_path, target_column_name, id_column_name, task_type, grid_type, positive_class='Success'):
        self.file_path = file_path
        self.dataset_name = os.path.basename(file_path).split('.')[0].split('__')[0]
        self.outliers = os.path.basename(file_path).split('.')[0].split('__')[2]
        self.feature_engineering = os.path.basename(file_path).split('.')[0].split('__')[3]
        self.target_column_name = target_column_name
        self.id_column_name = id_column_name
        self.task_type = task_type
        self.grid_type = grid_type
        self.positive_class = positive_class
        self.models = self.init_models(task_type)
        self.run_id = run_id
        self.setup_logging()

    def init_models(self, task_type):
        if task_type in ['binary_classification', 'multi_class_classification']:
            base_models = {
                "dummy_classifier" : DummyClassifier(strategy='stratified'),
                "logistic_regression": LogisticRegression(random_state=42, n_jobs=-1, max_iter=1000),
                "random_forest_classifier": RandomForestClassifier(random_state=42, n_jobs=-1),
                "decision_tree_classifier": DecisionTreeClassifier(random_state=42),
                # "xgboost_classifier": XGBClassifier(random_state=42, n_jobs=-1),
                # "lightgbm_classifier": LGBMClassifier(random_state=42, n_jobs=-1, verbosity=-1)
            }

            if task_type == 'multi_class_classification':
                if 'nn_classifier' in base_models:
                    base_models["nn_classifier"].set_params(model__target_type_='multiclass')
                if "xgboost_classifier" in base_models:
                    base_models["xgboost_classifier"].set_params(objective='multi:softprob', num_class='auto')

            return base_models
        elif task_type == 'regression':
            return {
                # "dummy_regressor" : DummyRegressor(strategy='mean'),
                "random_forest_regressor": RandomForestRegressor(random_state=42, n_jobs=-1),
                # "decision_tree_regressor": DecisionTreeRegressor(random_state=42),
                # "xgboost_regressor": XGBRegressor(random_state=42, n_jobs=-1),
                # "lightgbm_regressor": LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)
            }
        else:
            raise ValueError("Invalid task type specified. Choose 'binary_classification', 'multi_class_classification', or 'regression'.")

    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("logs", exist_ok=True)
        log_filename = f"logs/model_performance_{timestamp}_.log"
        logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    def load_data(self):
        logging.info(f"Loading data from {self.file_path}")
        print(f"Loading data from {self.file_path}")
        data = pd.read_csv(self.file_path)

        data = data.convert_dtypes(infer_objects=True)

        label_encoder_classes = None



        # Keep the target column in X
        X = data.copy()
        y = data[self.target_column_name].values

        temp_id = data[self.id_column_name]

        X = X.drop(self.target_column_name, axis=1)
        X = X.drop(self.id_column_name, axis=1)
        if self.task_type == 'binary_classification':
            y = np.where(y == self.positive_class, 1, 0)

        if self.task_type == 'multi_class_classification':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            label_encoder_classes = label_encoder.classes_

        categorical_features = X.select_dtypes(include=["string"]).columns.tolist()
        binary_features = X.select_dtypes(include=["boolean"]).columns.tolist()
        numerical_features = X.select_dtypes(include=["number"]).columns.tolist()

        X[binary_features] = X[binary_features].astype('Int8')
        X[categorical_features] = X[categorical_features].astype('object')
        X[numerical_features] = X[numerical_features].astype('float64')

        # # List all the columns in the dataset tha contain "usd"
        # usd_columns = [col for col in X.columns if 'usd' in col]
        # # Log10 transform all the columns that contain "usd"
        # X[usd_columns] = np.log10(X[usd_columns])

        X[self.id_column_name] = temp_id

        return X, y, numerical_features, categorical_features, binary_features, label_encoder_classes

    def create_preprocessor(self, numeric_columns, categorical_columns, binary_features):
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
        ])

        binary_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('variance_threshold', VarianceThreshold(threshold=0))
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=0.1))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", numerical_pipeline, numeric_columns),
                ("binary", binary_pipeline, binary_features),
                ("categorical", categorical_pipeline, categorical_columns),
            ],
            remainder='passthrough'
        )

        return preprocessor
    

    def train_model(self, X, y, model_name, numerical_features, categorical_features, binary_features, label_encoder=None):
        logging.info(f"Training model: {model_name}")

        self.filename = f"{self.dataset_name}__{self.task_type}__{self.outliers}__{self.feature_engineering}__{model_name}__{self.run_id}"
        start_time = datetime.now()
        preprocessor = self.create_preprocessor(numerical_features, categorical_features, binary_features)
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.models[model_name])
        ])

        param_grid = self.load_param_grids('param_grids.json', model_name)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_test_id = X_test[self.id_column_name]
        X_train = X_train.drop(self.id_column_name, axis=1)
        X_test = X_test.drop(self.id_column_name, axis=1)

        if self.task_type == 'regression':
            y_train = np.log10(y_train)
              
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )

        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

        if (self.task_type != "regression") and (model_name not in ('mlp_classifier','nn_classifier')):
            param_grid['model__class_weight'] = [class_weight_dict]

        scaler_mapping = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
            "PowerTransformer": PowerTransformer(method='yeo-johnson')
        }
        # Example usage in param_grid:
        param_grid = { **param_grid,
            'preprocessor__numerical__scaler': [scaler_mapping[scaler] for scaler in param_grid.get('preprocessor__numerical__scaler',["StandardScaler"])]
        }
        if self.grid_type == 'random_search':
            model_with_parameters, number_of_combinations = perform_random_search(model, model_name, X_train, y_train, cv=5, n_iter=100, scoring=self.select_scoring(), random_state=42, task_type = self.task_type)
        elif self.grid_type != 'non_grid':
            number_of_combinations = len(list(ParameterGrid(param_grid)))
            model_with_parameters = GridSearchCV(model, param_grid, cv=5, scoring=self.select_scoring(), n_jobs=-1, verbose=0, pre_dispatch='4*n_jobs', error_score='raise')
            model_with_parameters.fit(X_train, y_train)
            model_with_parameters = model_with_parameters.best_estimator_

        else:
            number_of_combinations = 1
            model_params = {k: v[0] for k, v in param_grid.items() if k.startswith('model__')}
            model.set_params(**model_params)
            model_with_parameters =  model.fit(X_train, y_train)

        stop_time = datetime.now()
        duration = stop_time - start_time
        metrics, conf_matrix, class_report, predicted_vs_actual = self.evaluate_model(model_with_parameters, X_test, y_test, label_encoder, X_test_id, X)
        feature_importance = self.get_feature_importance(model_with_parameters.named_steps['model'], model_with_parameters.named_steps['preprocessor'])
        self.save_model_and_metadata(model_with_parameters, metrics, conf_matrix, class_report, model_name, feature_importance,duration,number_of_combinations, predicted_vs_actual)
        return metrics

    def get_feature_importance(self, model, preprocessor):
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
            return pd.DataFrame()
        return feature_importance

    def load_param_grids(self, file_path, model_name):
        logging.info(f"Loading parameter grids for {model_name}")
        with open(file_path, 'r') as file:
            all_grids = json.load(file)
        return all_grids[model_name].get(self.grid_type, {})

    def select_scoring(self):
        if self.task_type in ['binary_classification']:
            return 'f1_weighted'
        elif self.task_type == 'multi_class_classification':
            return 'f1_weighted'
        elif self.task_type == 'regression':
            # return 'neg_mean_squared_error'
            # return 'neg_mean_absolute_percentage_error'
            # return make_scorer(threshold_mape, greater_is_better=False)
            return make_scorer(log10_threshold_probability_accuracy, greater_is_better=True, threshold=0.17609125905)
        else:
            raise ValueError("Unsupported task type for scoring")

    def evaluate_model(self, model, X_test, y_test,label_encoder, X_test_id, X):
        logging.info("Evaluating model")

        transformer = model.named_steps['preprocessor']
        estimator = model.named_steps['model']

        X_test_transformed = transformer.transform(X_test)

        pred = estimator.predict(X_test_transformed)
        predicted_vs_actual_describe = None

        if is_classifier(model):
            if self.task_type == 'multi_class_classification':
                pred_proba = model.predict_proba(X_test)

                metrics = {
                    #"ROC AUC Score": roc_auc_score(y_test, pred_proba, average='weighted', multi_class='ovr'),
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision": precision_score(y_test, pred, zero_division=0, average='weighted'),
                    "Recall": recall_score(y_test, pred, zero_division=0, average='weighted'),
                    "F1 Score": f1_score(y_test, pred, zero_division=0, average='weighted'),
                }
                conf_matrix = pd.DataFrame(confusion_matrix(y_test, pred), columns=label_encoder)
                class_report = classification_report(y_test, pred, target_names=label_encoder)

            else:
                metrics = {
                    "ROC AUC Score": roc_auc_score(y_test, pred),
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision": precision_score(y_test, pred, zero_division=0),
                    "Recall": recall_score(y_test, pred, zero_division=0),
                    "F1 Score": f1_score(y_test, pred, zero_division=0)
                }
                conf_matrix = pd.DataFrame(confusion_matrix(y_test, pred))
                class_report = classification_report(y_test, pred)

        elif is_regressor(model):
            pred_raw = model.predict(X_test)

            pred = np.power(10,  np.where(abs(pred_raw) >= 12, 12, abs(pred_raw)))
            #pred = abs(pred_raw)


            conf_matrix = None
            class_report = None

            metrics = {
                "MSE": mean_squared_error(y_test, pred),
                "MAPE": mean_absolute_percentage_error(y_test, pred),
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": root_mean_squared_error(y_test, pred),
                "RMSLE": root_mean_squared_log_error(y_test, pred),
                "MSLE" : mean_squared_log_error(y_test, pred),
                "R2": r2_score(y_test, pred),
                "Threshold Probability Accuracy": threshold_probability_accuracy(y_test, pred, threshold=0.2),
                "Threshold MAPE": threshold_mape(y_test, pred),
                "Threshold MAPE (25%)": threshold_mape(y_test, pred, threshold=0.25),

            }

            conf_matrix = None
            class_report = None

            predicted_vs_actual = pd.DataFrame({self.id_column_name: X_test_id.values, 'actual': y_test, 'predicted': pred})
            predicted_vs_actual['year'] = 10 *  np.floor(predicted_vs_actual.merge(X[['year',self.id_column_name]], on=self.id_column_name, how='left')['year']/10).astype(int)
            predicted_vs_actual['absolute_error'] = (predicted_vs_actual['predicted'] - predicted_vs_actual['actual']).abs()
            predicted_vs_actual['squared_error'] = (predicted_vs_actual['predicted'] - predicted_vs_actual['actual']) ** 2
            predicted_vs_actual['percentage_error'] = ((predicted_vs_actual['predicted'] - predicted_vs_actual['actual']) / predicted_vs_actual['actual'])
            predicted_vs_actual['absolute_percentage_error'] = abs(predicted_vs_actual['predicted'] - predicted_vs_actual['actual']) / predicted_vs_actual['actual']

            predicted_vs_actual.sort_values(by='absolute_percentage_error', ascending=False, inplace=True)


            # Assuming 'predicted_vs_actual' is your DataFrame with 'actual' and 'predicted' columns
            predicted_vs_actual['squared_error'] = (predicted_vs_actual['predicted'] - predicted_vs_actual['actual']) ** 2
            predicted_vs_actual['residuals'] = predicted_vs_actual['predicted'] - predicted_vs_actual['actual']
            predicted_vs_actual['threshold_mape'] = threshold_mape(predicted_vs_actual['actual'], predicted_vs_actual['predicted'])
            predicted_vs_actual['threshold_mape_25'] = threshold_mape(predicted_vs_actual['actual'], predicted_vs_actual['predicted'], threshold=0.25)

            # fig = px.scatter(predicted_vs_actual, x=predicted_vs_actual.index, y='squared_error',
            #                 title=f'Squared Errors for Each Prediction',
            #                 labels={'x': 'Index', 'y': 'Squared Error'},
            #                 log_y=True,  # Applying logarithmic scale on y-axis
            #                 hover_data=['movie_id'],  # Adding movie_id to hover information
            #                 width=1000, height=500)
            # fig.show()

            # fig = px.scatter(predicted_vs_actual, x='predicted', y='residuals',
            #                 title=f'Residual Plot',
            #                 labels={'x': 'Predicted Values', 'y': 'Residuals'},
            #                 hover_data=['movie_id'],  # Adding movie_id to hover information
            #                 width=1000, height=500)

            # # Adding a horizontal line at y=0
            # fig.add_hline(y=0, line_dash="dash", line_color="black")

            # fig.show()

            predicted_vs_actual_describe =  predicted_vs_actual.describe()
            
            # Create a dataframe styler object

            # sfromater = {
            #     'actual': "{:,.2f}",
            #     'predicted': "{:,.2f}",
            #     'absolute_error': "{:,.2f}",
            #     'squared_error': "{:,.2f}",
            #     'percentage_error': "{:.2f}",
            #     'absolute_percentage_error': "{:.2f}",
            #     'residuals': "{:,.2f}"
            # }


            print("Top 10 predictions with highest absolute percentage error")
            print(predicted_vs_actual.head(40).to_markdown(floatfmt=",.2f"))

            print("Top 10 predictions with lowest absolute percentage error")
            print(predicted_vs_actual.tail(25).to_markdown(floatfmt=",.2f"))

            print("Summary statistics for predictions")
            # I also want to format the thousand separator
            
            print(predicted_vs_actual_describe.to_markdown(floatfmt=",.2f"))
            
            # print(predicted_vs_actual.groupby('year')['absolute_percentage_error'].describe().sort_values(by='year').to_markdown(floatfmt=",.2f"))

                  
        return metrics, conf_matrix, class_report, predicted_vs_actual_describe

    def save_model_and_metadata(self, model, metrics, conf_matrix, class_report, model_type, feature_importance,duration, number_of_combinations, predicted_vs_actual_describe):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{self.filename}.pkl"
        metadata_filename = f"{self.filename}.json"
        os.makedirs("models", exist_ok=True)
        os.makedirs("metadata", exist_ok=True)
        if model_type not in ['nn_classifier', 'nn_regression']:
            joblib.dump(model, os.path.join("models", model_filename))
        feature_importance.to_dict(orient='records')
        model_params = model.named_steps["model"].get_params()
        model_params.pop('model',None)
        model_params.pop('estimator',None)
        metadata = {
            "run_id": self.run_id,
            "timestamp": timestamp,
            "model_type": model_type,
            "problem_type": self.task_type,
            "dataset_name": self.dataset_name,
            "has_outliers_removed": True if self.outliers == 'no_outliers' else False,
            'feature_engineering': self.feature_engineering,
            "grid_type": self.grid_type,
            "duration": duration.total_seconds(),
            "number_of_combinations": number_of_combinations,
            "metrics": metrics,
            "conf_matrix": conf_matrix.to_dict(orient='records') if conf_matrix is not None else None,
            "class_report": class_report,
            "model_parameters": model_params,
            "scaler": type(model.named_steps["preprocessor"].named_transformers_['numerical'].named_steps['scaler']).__name__,
            "variance_threshold": model.named_steps["preprocessor"].named_transformers_['binary'].named_steps['variance_threshold'].threshold,
            "feature_importance": feature_importance.to_dict(orient='records'),
            "predicted_vs_actual": predicted_vs_actual_describe.to_json(orient="columns") if predicted_vs_actual_describe is not None else None
        }
        with open(os.path.join("metadata", metadata_filename), "w") as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"Saved model and metadata for {model_type}")

    def run(self):
        results = []
        X, y, numerical_features, categorical_features, binary_features, label_encoder = self.load_data()
        for model_name in self.models.keys():
            ##print(f"Training model: {model_name}\n")
            metrics = self.train_model(X, y, model_name, numerical_features, categorical_features, binary_features, label_encoder)
            results.append({"Model": model_name, **metrics})
        return pd.DataFrame(results)

if __name__ == "__main__":

    GRID_TYPE = "random_search"
    ID_COLUMN_NAME = "movie_id"

    DATA_FILES_LIST = os.listdir("./data/ml_ready_data")
    # DATA_FILES_LIST = [i for i in DATA_FILES_LIST if i.split("__")[1] == "binary_classification"]
    DATA_FILES_LIST = [
                       "full__regression__no_outliers__complex.csv",
                        # "small_productions__regression__no_outliers__complex.csv",
                        # "medium_productions__regression__no_outliers__complex.csv",
                        # "large_productions__regression__no_outliers__complex.csv"
                       ]
    TASK_TYPE_LIST = [i.split("__")[1] for i in DATA_FILES_LIST]
    TARGET_COLUMN_NAME_LIST = [
        "revenue_usd_adj" if i == "regression" else i for i in TASK_TYPE_LIST
    ]

    # create a hash for each run
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    cases = len(
        [
            (i, j, k)
            for i, j, k in zip(DATA_FILES_LIST, TASK_TYPE_LIST, TARGET_COLUMN_NAME_LIST)
        ]
    )
    counter = 1
    print(f"Running {RUN_ID} with {cases} cases")
    for data_file, task_type, target_column_name in zip(
        DATA_FILES_LIST, TASK_TYPE_LIST, TARGET_COLUMN_NAME_LIST
    ):
        print(f"Running model {counter} from {cases}")
        counter += 1
        trainer = MOTR(
            RUN_ID,
            f"./data/ml_ready_data/{data_file}",
            target_column_name,
            ID_COLUMN_NAME,
            task_type=task_type,
            grid_type=GRID_TYPE,
            positive_class="Success",
        )
        results = trainer.run()
        print(f"Results for {data_file}")
        print(results)
