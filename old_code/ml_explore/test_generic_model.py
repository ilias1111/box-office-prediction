import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from ml_support_functions import load_data, clean_data, plot_feature_importance

class GenericMLModel:
    def __init__(self, dataset_name, columns_to_drop, target_column, model_type):
        self.dataset_name = dataset_name
        self.columns_to_drop = columns_to_drop
        self.target_column = target_column
        self.model_type = model_type
        self.model = None

    def load_and_preprocess_data(self):
        data = load_data(self.dataset_name)
        data = clean_data(data, columns_to_drop=self.columns_to_drop)
        return data

    def train_model(self, data, params, num_boost_round=100, early_stopping_rounds=10):
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.model_type == 'lightgbm':
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test)
            self.model = lgb.train(params,
                                   train_data,
                                   num_boost_round=num_boost_round,
                                   valid_sets=[train_data, valid_data],
                                   early_stopping_rounds=early_stopping_rounds)
        elif self.model_type == 'xgboost':
            train_data = xgb.DMatrix(X_train, label=y_train)
            valid_data = xgb.DMatrix(X_test, label=y_test)
            self.model = xgb.train(params,
                                   train_data,
                                   num_boost_round=num_boost_round,
                                   evals=[(train_data, 'train'), (valid_data, 'eval')],
                                   early_stopping_rounds=early_stopping_rounds)
        elif self.model_type == 'catboost':
            self.model = cb.CatBoostClassifier(**params) if params.get('objective') == 'Logloss' else cb.CatBoostRegressor(**params)
            self.model.fit(X_train, y_train,
                           eval_set=(X_test, y_test),
                           early_stopping_rounds=early_stopping_rounds,
                           verbose=False)
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(**params)
            self.model.fit(X_train, y_train)
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(**params)
            self.model.fit(X_train, y_train)
        else:
            raise NotImplementedError(f"Model type {self.model_type} is not supported.")

        return X_test, y_test

    def predict_and_evaluate(self, X_test, y_test):
        if self.model_type in ['lightgbm', 'xgboost']:
            y_pred = self.model.predict(X_test)
        elif self.model_type == 'catboost':
            y_pred = self.model.predict(X_test)
        elif self.model_type in ['logistic_regression', 'random_forest']:
            y_pred = self.model.predict(X_test)
        else:
            raise NotImplementedError(f"Model type {self.model_type} is not supported for prediction.")

        if self.model_type in ['logistic_regression', 'random_forest', 'catboost'] and params.get('objective') == 'Logloss':
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy}')
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5
            print(f'RMSE: {rmse}')

    def plot_importances(self):
        if self.model_type in ['lightgbm', 'xgboost', 'catboost']:
            feature_importances = self.model.get_feature_importance()
            feature_names = X.columns.tolist()
            plot_feature_importance(feature_importances, feature_names, 'Feature importance')
        elif self.model_type == 'random_forest':
            feature_importances = self.model.feature_importances_
            feature_names = X.columns.tolist()
            plot_feature_importance(feature_importances, feature_names, 'Feature importance')
        else:
            print(f"Model type {self.model_type} does not support feature importance plotting.")

# Example usage:
params = {
    'objective': 'regression',
    # other parameters as needed for the specific model
}

model = GenericMLModel(dataset_name='your_dataset.csv',
                       columns_to_drop=['unnecessary_column'],
                       target_column='target',
                       model_type='lightgbm')
data = model.load_and_preprocess_data()
X_test, y_test = model.train_model(data, params)
model.predict_and_evaluate(X_test, y_test)
model.plot_importances()
