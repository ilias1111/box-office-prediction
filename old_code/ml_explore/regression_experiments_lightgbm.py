import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
from ml_support_functions import load_data, clean_data, split_dataset, plot_feature_importance

if __name__ == '__main__':
    # Load and preprocess the data
    data = load_data('ml_data_reg_large.csv')
    data = clean_data(data, columns_to_drop=["revenue_usd_adj_log", "budget_usd_adj_log", 'movie_id',
                                             'revenue_world','revenue_dom', 'revenue_int', 'revenue_open','surplus','surplus',
                                             'metascore', 'rating_count_imdb', 'rating_value_imdb',
                                             'rating_count_tmdb', 'rating_value_tmdb', 'tag','ratio_adj'])
    
    feature_names = data.columns.drop('revenue_usd_adj')

    # Split the data
    X_train, X_test, y_train, y_test = split_dataset(data, target_column='revenue_usd_adj')

    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    # Parameters for the model
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',  # Added 'regression' as objective
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 1.0,  # Equivalent to 'colsample_bytree'
        'bagging_fraction': 1.0,  # Equivalent to 'subsample'
        'min_data_in_leaf': 20,   # Equivalent to 'min_child_samples'
        'min_sum_hessian_in_leaf': 0.001,  # Might be used as 'min_child_weight'
        'min_gain_to_split': 0.0,  # Equivalent to 'min_split_gain'
        'n_estimators': 100,
        'num_threads': -1,  # Equivalent to 'n_jobs'
        'random_state': 1053608,
        'lambda_l1': 0.0,  # Equivalent to 'reg_alpha'
        'lambda_l2': 0.0,  # Equivalent to 'reg_lambda'
        'verbosity': -1,  # Equivalent to 'silent'
        'bagging_freq': 0,
        'bagging_seed': 1053608,  # Use the same seed for bagging as the random state
        'metric' : 'mape'
    }

# You can then use these parameters with LightGBM's train function or cv function.
# lgb.train(params, train_set, num_boost_round=100)


    # Training the model
    print("Starting training...")
    gbm = lgb.train(params,
                    train_data,
                    num_boost_round=2000,
                    valid_sets=[train_data, test_data],
                    valid_names=['train', 'eval'],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])

    # Save the model
    gbm.save_model('lightgbm_model.txt')

    # Predict on test data
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    # Evaluate the model
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'MAPE: {mape}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    print(f'R^2: {r2}')


    # Get feature importances
    feature_importance_values = gbm.feature_importance(importance_type='split')
    #feature_names = X_train.columns

    # Create a DataFrame for feature importances
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    # Sort features by importance
    feature_importances = feature_importances.sort_values(by='importance', ascending=False).reset_index(drop=True)

    # Display feature importances
    print(feature_importances)

    # For gain importance, change importance_type to 'gain'
    feature_importance_values_gain = gbm.feature_importance(importance_type='gain')
    feature_importances_gain = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values_gain})
    feature_importances_gain = feature_importances_gain.sort_values(by='importance', ascending=False).reset_index(drop=True)
    print(feature_importances_gain)


    # Plot feature importances for split
    plot_feature_importance(feature_importance_values, feature_names, 'Feature Importance (split)', max_num_features=30)

    # Plot feature importances for gain
    plot_feature_importance(feature_importance_values_gain, feature_names, 'Feature Importance (gain)', max_num_features=30)


    # plt.figure(figsize=(20, 10))
    # ax = lgb.plot_tree(gbm, tree_index=0, figsize=(20, 10), show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
    # plt.show()