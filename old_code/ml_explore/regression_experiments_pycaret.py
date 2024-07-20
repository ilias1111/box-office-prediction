import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, explained_variance_score, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, ParameterGrid
from ml_support_functions import load_data, clean_data, split_dataset, plot_feature_importance


# Load and preprocess the data
data = load_data('code/data_pipeline/ml_data_reg.csv')
data = clean_data(data, columns_to_drop=["revenue_usd_adj_log", "budget_usd_adj_log", 'movie_id',
                                            'revenue_world','revenue_dom', 'revenue_int', 'revenue_open','surplus','surplus',
                                            'metascore', 'rating_count_imdb', 'rating_value_imdb',
                                            'rating_count_tmdb', 'rating_value_tmdb', 'tag','ratio_adj'])


feature_names = data.columns.drop('revenue_usd_adj')
X_train, X_test, y_train, y_test = split_dataset(data, target_column='revenue_usd_adj')  # Set your actual target column


# Assuming X_train, y_train are already defined and preprocessed

# Define a wider parameter grid
param_grid = {
    'num_leaves': [2**3, 2**5 ,2**15],  # Lower or higher to adjust model complexity
    'max_depth': [3,5,15],  # -1 means no limit
    'learning_rate': [0.01],
    'n_estimators': [500],
    'subsample': [1.0],  # Disable subsampling by setting to 1.0
    'colsample_bytree': [0.5]  # Disable col subsampling by setting to 1.0,
    # You can include other parameters such as 'min_child_weight', 'reg_alpha', and 'reg_lambda'
}

# Initialize the LGBMRegressor
lgbm = lgb.LGBMRegressor()

# Initialize the GridSearchCV
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_percentage_error',
    n_jobs=-1,
    verbose=2
)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Get the best parameters and score
best_params = grid_search.best_params_
best_score = (-1) * grid_search.best_score_  # Multiply by -1 because sklearn "neg" scores are negatives

# Print the results
print(f"Best Score: {best_score}")
print(f"Best Parameters: {best_params}")

# Best model is automatically refitted on the whole training set with best parameters
best_model = grid_search.best_estimator_
print(best_model)

# Predict with the best model
y_pred = best_model.predict(X_test)

# Evaluation metrics
print('RMSE:', mean_squared_error(y_test, y_pred, squared=True))
print('MSE:', mean_squared_error(y_test, y_pred, squared=False))
print('MAE:', mean_absolute_error(y_test, y_pred))
print('MAPE:', mean_absolute_percentage_error(y_test, y_pred))
print('Max Error:', max_error(y_test, y_pred))
print('Explained Variance Score:', explained_variance_score(y_test, y_pred))
