import numpy as np

def adaptive_rmsle_scorer(estimator, X, y_true):
    y_pred = estimator.predict(X)
    
    # Check if log transform is being used
    log_transform_used = estimator.named_steps['preprocessor'].named_transformers_['target'].named_steps['log_transform'].use_log
    
    if log_transform_used:
        # Data is already log-transformed, calculate RMSLE directly
        return -np.sqrt(np.mean(np.power(y_true - y_pred, 2)))
    else:
        # Data is in original scale, apply log1p before calculating RMSLE
        return -np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))