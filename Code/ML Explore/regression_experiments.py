from math import log
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
import pandas as pd
from pycaret.regression import setup, compare_models
import logging
import re
import unicodedata


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def clean_feature_name(name):
    # Remove accents
    name = ''.join(c for c in unicodedata.normalize('NFKD', name) if unicodedata.category(c) != 'Mn')
    
    # Replace spaces and non-alphanumeric characters with underscores
    name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    
    # Remove leading or trailing underscores
    name = name.strip('_')
    
    # Convert to lowercase
    name = name.lower()
    
    return name

def clean_feature_names(data):
    cleaned_data = data.rename(columns={col: clean_feature_name(col) for col in data.columns})
    return cleaned_data


def load_data(file_path):
    """Loads data from a given file path."""
    return pd.read_csv(file_path)


def clean_data(data, columns_to_drop=None):
    """Performs data cleaning."""
    # Identify non-numerical columns
    non_numerical_columns = data.select_dtypes(
        exclude=['number']).columns.tolist()

    # Drop non-numerical columns
    data = data.drop(columns=non_numerical_columns)
    data = data.drop(columns=columns_to_drop)

    data = clean_feature_names(data)
    # Print the dropped columns
    print("Dropped Non-Numerical Columns:", non_numerical_columns)

    print("Feature names:", list(data.columns))

    ## Return any duplicate columns names and drop them
    duplicate_columns = data.columns[data.columns.duplicated()]
    print("Duplicate Columns:", duplicate_columns)
    data = data.drop(columns=duplicate_columns)

    return data


def split_dataset(data, target_column='revenue_usd_adj', test_size=0.2):
    """Preprocesses the data."""
    X = data.drop(columns=target_column)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def build_and_train_model(X_train, y_train, X_test, y_test, optimizer='Adamax', nLayers=3, act="elu", neurons=512, dropout=0.8):
    """
    Build and train a deep neural networks model.

    :param X_train: Training data features
    :param y_train: Training data labels
    :param X_test: Testing data features
    :param y_test: Testing data labels
    :param optimizer: Optimizer to use for training
    :param nLayers: Number of hidden layers
    :param act: Activation function
    :param neurons: Number of neurons in hidden layers
    :param dropout: Dropout rate
    :return: Trained model and training history
    """
    shape = X_train.shape[1]  # Extracting the shape from the training data

    # Building the model
    model = Sequential()
    model.add(Dense(4096, input_dim=shape, activation=act))
    model.add(Dropout(0.8))
    model.add(Dense(2048, activation=act))
    model.add(Dropout(0.8))

    for _ in range(nLayers):
        model.add(Dense(neurons, activation=act))
        model.add(Dropout(dropout))

    model.add(Dense(64, activation=act))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="relu"))

    # Compiling the model
    model.compile(optimizer=optimizer,
                  loss='mape',
                  metrics=[tf.keras.metrics.RootMeanSquaredError(),
                           tf.keras.metrics.MAPE,
                           tf.keras.metrics.MSE,
                           tf.keras.metrics.MAE])

    # Training the model
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=400,
                        batch_size=1024,
                        verbose=2,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_percentage_error', mode="min", patience=40)])

    return model, history


def plot_training_history(history):
    metrics = ['root_mean_squared_error', 'mean_absolute_error',
               'loss', 'mean_absolute_percentage_error', 'mean_squared_error']
    for metric in metrics:
        plt.plot(history.history[metric])
        plt.plot(history.history[f'val_{metric}'])
        plt.title(f'Model {metric}')
        plt.ylabel(metric)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


def setup_pycaret(data, target):
    try:
        reg_s = setup(data, target=target, session_id=1053608,
                      log_profile=True,
                      log_experiment=True, experiment_name='reg_exp_1',
                      normalize=True, normalize_method='minmax',
                      feature_selection=True
                      )
        return reg_s
    except Exception as e:
        logging.error(f"Error setting up PyCaret: {e}")
        return None


def compare_models_pycaret():
    try:
        best_models = compare_models(n_select=3, sort='MAPE', turbo=True,
                                     exclude=['par', 'et', 'rf', 'ada','lightgbm'])
        return best_models
    except Exception as e:
        logging.error(f"Error comparing models in PyCaret: {e}")
        return None


if __name__ == '__main__':
    # data = load_data('Code/Data Pipeline/ml_data_reg.csv')
    # data = clean_data(data)
    # X_train, X_test, y_train, y_test = preprocess_data(data, columns_to_drop=['revenue_usd_adj', 'revenue_usd_adj_log', 'budget_usd_adj', 'budget_usd_adj_log'])
    # model, history = build_and_train_model(X_train, y_train, X_test, y_test)
    # plot_training_history(history)
    # model.save('model.h5')
    # print(model.summary())

    # Load and preprocess data
    ml_data_reg = load_data('Code/Data Pipeline/ml_data_reg.csv')
    if ml_data_reg is None:
        logging.error("Error loading data!")
    data = clean_data(ml_data_reg, columns_to_drop=[
                      'revenue_usd_adj_log', 'budget_usd_adj'])
    if data is None:
        logging.error("Error preprocessing data!")
    # Setup PyCaret
    setup_pycaret(data, target='revenue_usd_adj')
    # if reg_data is None:
    #     logging.error("Error setting up PyCaret!")
    # Compare models
    best_models = compare_models_pycaret()
    for model in best_models:
        logging.info(f"Model: {model}")
