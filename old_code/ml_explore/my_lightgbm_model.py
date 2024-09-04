import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from ml_support_functions import (
    load_data,
    clean_data,
    split_dataset,
    plot_feature_importance,
)


class LightGBMRegressor:
    def __init__(self, dataset_name, columns_to_drop, target_column):
        self.dataset_name = dataset_name
        self.columns_to_drop = columns_to_drop
        self.target_column = target_column
        self.model = None
        self.feature_importances = None
        self.feature_importances_gain = None

    def load_and_preprocess_data(self):
        data = load_data(self.dataset_name)
        data = clean_data(data, columns_to_drop=self.columns_to_drop)
        return data

    def train_model(self, data, params, num_boost_round, early_stopping_rounds):
        feature_names = data.columns.drop(self.target_column)
        X_train, X_test, y_train, y_test = split_dataset(
            data, target_column=self.target_column
        )
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test)

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, test_data],
            valid_names=["train", "eval"],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(50),
            ],
        )
        return X_test, y_test

    def predict_and_evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse**0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results = {"MAPE": mape, "MSE": mse, "RMSE": rmse, "MAE": mae, "R^2": r2}
        return results

    def save_model(self, filename):
        self.model.save_model(filename)

    def calculate_feature_importances(self, feature_names, importance_type="split"):
        feature_importance_values = self.model.feature_importance(
            importance_type=importance_type
        )
        feature_importances = pd.DataFrame(
            {"feature": feature_names, "importance": feature_importance_values}
        )
        feature_importances = feature_importances.sort_values(
            by="importance", ascending=False
        ).reset_index(drop=True)
        return feature_importances

    def plot_importances(self, feature_importances, title, max_num_features=30):
        plot_feature_importance(
            feature_importances["importance"],
            feature_importances["feature"],
            title,
            max_num_features,
        )


if __name__ == "__main__":
    # Parameters for LightGBM
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",  # Added 'regression' as objective
        "num_leaves": 31,
        "learning_rate": 0.1,
        "feature_fraction": 1.0,  # Equivalent to 'colsample_bytree'
        "bagging_fraction": 1.0,  # Equivalent to 'subsample'
        "min_data_in_leaf": 20,  # Equivalent to 'min_child_samples'
        "min_sum_hessian_in_leaf": 0.001,  # Might be used as 'min_child_weight'
        "min_gain_to_split": 0.0,  # Equivalent to 'min_split_gain'
        "n_estimators": 100,
        "num_threads": -1,  # Equivalent to 'n_jobs'
        "random_state": 1053608,
        "lambda_l1": 0.0,  # Equivalent to 'reg_alpha'
        "lambda_l2": 0.0,  # Equivalent to 'reg_lambda'
        "verbosity": -1,  # Equivalent to 'silent'
        "bagging_freq": 0,
        "bagging_seed": 1053608,  # Use the same seed for bagging as the random state
        "metric": "mape",
    }

    columns_to_drop = [
        "revenue_usd_adj_log",
        "budget_usd_adj_log",
        "movie_id",
        "revenue_world",
        "revenue_dom",
        "revenue_int",
        "revenue_open",
        "surplus",
        "surplus",
        "metascore",
        "rating_count_imdb",
        "rating_value_imdb",
        "rating_count_tmdb",
        "rating_value_tmdb",
        "tag",
        "ratio_adj",
    ]

    # iterate for ml_data_reg_X ( where X being small, medium, large, full)
    for i in ["small", "medium", "large", "full"]:
        # Initialize the LightGBM regressor class
        lgb_regressor = LightGBMRegressor(
            f"code/data_pipeline/data/ml_data_reg_{i}.csv",
            columns_to_drop=columns_to_drop,
            target_column="revenue_usd_adj",
        )

        # Load and preprocess data
        data = lgb_regressor.load_and_preprocess_data()

        # Train the model
        X_test, y_test = lgb_regressor.train_model(
            data, params, num_boost_round=2000, early_stopping_rounds=50
        )

        print(i)
        # Predict and evaluate
        evaluation_results = lgb_regressor.predict_and_evaluate(X_test, y_test)
        print(evaluation_results)

    # # Save model
    # lgb_regressor.save_model(f'lightgbm_model_{i}.txt')

    # # Calculate feature importances
    # feature_importances = lgb_regressor.calculate_feature_importances(data.columns.drop('revenue_usd_adj'))
    # print(feature_importances)

    # # Plot feature importances
    # lgb_regressor.plot_importances(feature_importances, 'Feature Importance (split)')
