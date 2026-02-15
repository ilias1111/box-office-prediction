import numpy as np


def threshold_mape(y_true, y_pred, threshold=0.1):
    """
    Calculate Mean Absolute Percentage Error (MAPE) for errors exceeding the threshold.

    Args:
    y_true (array-like): True values
    y_pred (array-like): Predicted values
    threshold (float): Error threshold (default: 0.1, i.e., 10%)

    Returns:
    float: TMAPE score
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Avoid division by zero
    mask = y_true != 0

    # Calculate percentage errors
    percentage_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])

    # Only consider errors above the threshold
    significant_errors = percentage_errors[percentage_errors > threshold]

    if len(significant_errors) == 0:
        return 0.0
    else:
        return np.mean(significant_errors)


def threshold_probability_accuracy(y_true, y_pred, threshold=0.1):
    """
    Calculate the proportion of predictions with error less than the threshold.

    Args:
    y_true (array-like): True values
    y_pred (array-like): Predicted values
    threshold (float): Error threshold (default: 0.1, i.e., 10%)

    Returns:
    float: TPA score (proportion of predictions within the threshold)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Avoid division by zero
    mask = y_true != 0
    y_true, y_pred = y_true[mask], y_pred[mask]

    # Calculate absolute percentage errors
    percentage_errors = np.abs((y_true - y_pred) / y_true)

    # Calculate proportion of errors below the threshold
    accuracy = np.mean(percentage_errors <= threshold)

    return accuracy


def log10_threshold_probability_accuracy(y_true, y_pred, threshold=0.0414):
    """
    Calculate the proportion of predictions with log10 error less than the threshold.

    Args:
    y_true (array-like): True log10 values
    y_pred (array-like): Predicted log10 values
    threshold (float): Error threshold in log10 space (default: 0.0414, i.e., ~10% error in linear space)
    #20% error in linear space is 0.1 in log10 space (log10(1.2) - log10(0.8) = 0.0414)

    Returns:
    float: Log10 TPA score (proportion of predictions within the threshold)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Calculate absolute errors in log space
    log_errors = np.abs(y_true - y_pred)

    # Calculate proportion of errors below the threshold
    accuracy = np.mean(log_errors <= threshold)

    return accuracy
