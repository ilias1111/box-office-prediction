import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.utils import class_weight
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from datetime import datetime
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, root_mean_squared_error, mean_squared_log_error, root_mean_squared_log_error,
    mean_absolute_error, mean_absolute_percentage_error, r2_score
)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_info_columns', 99999)

class ModelTrainer:
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
                "logistic_regression": LogisticRegression(random_state=42, n_jobs=-1, multi_class='multinomial', max_iter=1000),
                "random_forest_classifier": RandomForestClassifier(random_state=42, n_jobs=-1),
                "support_vector_machine_classifier": SVC(probability=True, random_state=42, decision_function_shape='ovo'),
                "decision_tree_classifier": DecisionTreeClassifier(random_state=42),
                "xgboost_classifier": XGBClassifier(random_state=42, n_jobs=-1),
                "lightgbm_classifier": LGBMClassifier(random_state=42, n_jobs=-1)
            }
            if task_type == 'multi_class_classification':
                base_models["xgboost_classifier"].set_params(objective='multi:softprob', num_class='auto')
            return base_models
        elif task_type == 'regression':
            return {
                "linear_regression": LinearRegression(n_jobs=-1),
                "random_forest_regressor": RandomForestRegressor(random_state=42, n_jobs=-1),
                "decision_tree_regressor": DecisionTreeRegressor(random_state=42),
                "xgboost_regressor": XGBRegressor(random_state=42, n_jobs=-1, base_score =0),
                "lightgbm_regressor": LGBMRegressor(random_state=42, n_jobs=-1, base_score=0)
            }
        else:
            raise ValueError("Invalid task type specified. Choose 'binary_classification', 'multi_class_classification', or 'regression'.")


    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/model_performance_{timestamp}_.log"
        logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    def load_data(self):

        logging.info(f"Loading data from {self.file_path}")
        print(f"Loading data from {self.file_path}")
        data = pd.read_csv(self.file_path)
        data.drop(self.id_column_name, axis=1, inplace=True)
    
        data = data.convert_dtypes(infer_objects=True)

        # # Dynamicly identify columns to convert to boolean
        # convert_to_bool_columns = [i for i in data.columns if data[i].nunique() <= 2 and set(data[i].unique()) in set(["True","False", "TRUE", "FALSE", "true", "false"]) and data[i].dtype in ['object','string']]
        # print(convert_to_bool_columns)
        # data[convert_to_bool_columns] = data[convert_to_bool_columns].astype('boolean')

        # for i in data.columns:
        #     if data[i].nunique() <= 2 and data[i].dtype in ['object','string']:
        #         print(i)
        #         print(data[i].unique())


        # Transform binary to 1 and 0

        label_encoder_classes = None



        y = data[self.target_column_name].values
        X = data.drop(self.target_column_name, axis=1)
        if self.task_type == 'binary_classification':
            y = np.where(y == self.positive_class, 1, 0)

            # Handle multi-class labels if the task type is multi-class classification
        if self.task_type == 'multi_class_classification':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)  # Convert categorical labels to integers
            label_encoder_classes = label_encoder.classes_
        categorical_features = X.select_dtypes(include=["string"]).columns.tolist()
        binary_features = X.select_dtypes(include=["boolean"]).columns.tolist()
        numerical_features = X.select_dtypes(include=["number"]).columns.tolist()

        
        X[binary_features] = X[binary_features].astype('Int8')
        X[categorical_features] = X[categorical_features].astype('object')
        X[numerical_features] = X[numerical_features].astype('float64')


        return X, y, numerical_features, categorical_features, binary_features, label_encoder_classes

    def create_preprocessor(self, numeric_columns, categorical_columns, binary_features):
        logging.info("Creating preprocessor")
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numeric_columns),
                ("binary", Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ]), binary_features),
                ("cat", Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_columns)
            ],
            remainder='passthrough'
        )
        return preprocessor

    def train_model(self, X, y, model_name, numerical_features, categorical_features, binary_features, label_encoder=None):
        logging.info(f"Training model: {model_name}")
        preprocessor = self.create_preprocessor(numerical_features, categorical_features, binary_features)
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.models[model_name])
        ])


        param_grid = self.load_param_grids('param_grids.json', model_name)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )

        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

        param_grid['classifier__class_weight'] = [class_weight_dict]

        scaler_mapping = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler()
        }
        # Example usage in param_grid:
        param_grid = {
            'preprocessor__num__scaler': [scaler_mapping[scaler] for scaler in param_grid.get('preprocessor__num__scaler',["StandardScaler"])]
        }

        
        if self.grid_type != 'non_grid':
            model_with_parameters = GridSearchCV(model, param_grid, cv=5, scoring=self.select_scoring(), n_jobs=-1, verbose=2)
            model_with_parameters.fit(X_train, y_train)
            model_with_parameters = model_with_parameters.best_estimator_


        else:
            model_params = {k: v[0] for k, v in param_grid.items() if k.startswith('model__')}
            model.set_params(**model_params)
            model_with_parameters =  model.fit(X_train, y_train)

        
        metrics, conf_matrix, class_report = self.evaluate_model(model_with_parameters, X_test, y_test,label_encoder)
        feature_importance = self.get_feature_importance(model_with_parameters.named_steps['model'], model_with_parameters.named_steps['preprocessor'])
        self.save_model_and_metadata(model_with_parameters, metrics, conf_matrix, class_report, model_name, feature_importance)
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
            return pd.DataFrame()  # Return empty DataFrame if conditions not met
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
            return 'neg_mean_absolute_percentage_error'
        else:
            raise ValueError("Unsupported task type for scoring")
    
    def evaluate_model(self, model, X_test, y_test,label_encoder):
        logging.info("Evaluating model")
        pred = model.predict(X_test)
        if is_classifier(model):
            if self.task_type == 'multi_class_classification':
                pred_proba = model.predict_proba(X_test)
                metrics = {
                    "ROC AUC Score": roc_auc_score(y_test, pred_proba, average='weighted', multi_class='ovr'),
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
            pred = np.abs(pred)
            metrics = {
                "MSE": mean_squared_error(y_test, pred),
                "MAPE": mean_absolute_percentage_error(y_test, pred),
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": root_mean_squared_error(y_test, pred),
                "RMSLE": root_mean_squared_log_error(y_test, pred),
                "MSLE" : mean_squared_log_error(y_test, pred),
                "R2": r2_score(y_test, pred)
            }
            conf_matrix = None
            class_report = None
        return metrics, conf_matrix, class_report

    def save_model_and_metadata(self, model, metrics, conf_matrix, class_report, model_type, feature_importance):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{timestamp}_{self.dataset_name}_{model_type}.pkl"
        metadata_filename = f"{timestamp}_{self.dataset_name}_{model_type}.json"
        os.makedirs("models", exist_ok=True)
        os.makedirs("metadata", exist_ok=True)
        joblib.dump(model, os.path.join("models", model_filename))
        feature_importance.to_dict(orient='records')
        metadata = {
            "run_id" : self.run_id,
            "timestamp": timestamp,
            "model_type": model_type,
            "problem_type": self.task_type,
            "dataset_name": self.dataset_name,
            "has_outliers_removed": True if self.outliers == 'no_outliers' else False,
            'feature_engineering' : self.feature_engineering,
            "grid_type": self.grid_type,
            "metrics": metrics,
            "conf_matrix": conf_matrix.to_dict(orient='records') if conf_matrix is not None else None,
            "class_report": class_report,
            "model_parameters": model.named_steps["model"].get_params(),
            "feature_importance": feature_importance.to_dict(orient='records')
        }
        with open(os.path.join("metadata", metadata_filename), "w") as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"Saved model and metadata for {model_type}")

    def run(self):
        results = []
        X, y, numerical_features, categorical_features, binary_features, label_encoder = self.load_data()
        for model_name in self.models.keys():
            metrics = self.train_model(X, y, model_name, numerical_features, categorical_features, binary_features, label_encoder)
            results.append({"Model": model_name, **metrics})
        return pd.DataFrame(results)

if __name__ == "__main__":

    GRID_TYPE = 'grid_search'
    ID_COLUMN_NAME = "movie_id"

    DATA_FILES_LIST = os.listdir('./data/ml_ready_data')
    TASK_TYPE_LIST = [i.split('__')[1] for i in DATA_FILES_LIST]
    TARGET_COLUMN_NAME_LIST = ['revenue_usd_adj' if i == 'regression' else i for i in TASK_TYPE_LIST]

    #create a hash for each run
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

    for data_file, task_type, target_column_name in zip(DATA_FILES_LIST, TASK_TYPE_LIST, TARGET_COLUMN_NAME_LIST):
        trainer = ModelTrainer(RUN_ID, f'./data/ml_ready_data/{data_file}', target_column_name, ID_COLUMN_NAME, task_type=task_type, grid_type=GRID_TYPE, positive_class='Success')
        results = trainer.run()
        # print(f"Results for {data_file}")
        # print(results)
