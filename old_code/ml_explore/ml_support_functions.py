import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def remove_outlier_ratio(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes outliers from the given DataFrame based on the ratio_adj column.

    Parameters:
    data (pd.DataFrame): The DataFrame to remove outliers from.
    """

    return data[(data.ratio_adj > 0.1) & (data.ratio_adj < 100)]


# Helper function to plot feature importance
def plot_feature_importance(importance, names, title, max_num_features=None):
    # Create tuples of feature names and importance
    feature_importance = zip(names, importance)
    # Create a list and sort it based on importance
    feature_importance = sorted(
        list(feature_importance), key=lambda x: x[1], reverse=True
    )
    if max_num_features:
        feature_importance = feature_importance[:max_num_features]
    # Unzip the feature names and importance into separate lists
    sorted_features, sorted_importance = zip(*feature_importance)

    # Create a barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(y=list(sorted_features), x=list(sorted_importance), palette="viridis")
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.grid(True, which="major", linestyle="--", linewidth="0.5", color="grey")
    plt.show()


def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a given file path.

    Parameters:
    file_path (str): The file path to load the data from.
    """
    return pd.read_csv(file_path)


def clean_data(
    data: pd.DataFrame, columns_to_drop: list[str] = None, remove_outliers: bool = False
) -> pd.DataFrame:
    """Performs data cleaning.

    Parameters:
    data (pd.DataFrame): The dataframe to clean.
    columns_to_drop (list): The list of columns to drop.
    remove_outliers (bool): Whether to remove outliers or not.
    """
    # Identify non-numerical columns
    non_numerical_columns = data.select_dtypes(exclude=["number"]).columns.tolist()

    if remove_outliers:
        # Remove outliers
        data = remove_outlier_ratio(data)

    # Drop non-numerical columns
    data = data.drop(columns=non_numerical_columns)
    data = data.drop(columns=columns_to_drop)

    return data


def split_dataset(data, target_column, test_size=0.2):
    """Preprocesses the data and splits it into train and test sets."""
    X = data.drop(columns=target_column)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    return X_train, X_test, y_train, y_test


def plot_training_history(history):
    """Plots the training history."""
    plt.plot(history.history["mean_absolute_percentage_error"])
    plt.plot(history.history["val_mean_absolute_percentage_error"])
    plt.title("Model MAE")
    plt.ylabel("Mean Absolute Error")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()

    plt.plot(history.history["mean_squared_error"])
    plt.plot(history.history["val_mean_squared_error"])
    plt.title("Model MSE")
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()
