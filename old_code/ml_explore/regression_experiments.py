from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import logging
from ml_support_functions import (
    load_data,
    clean_data,
    split_dataset,
    plot_training_history,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def build_neural_network_model(
    shape, optimizer="Adamax", layers=3, act="elu", neurons=512, dropout=0.8
):
    """
    Build_Model_DNN_Tex(shape, nClasses,dropout)
    Build Deep neural networks Model
    Shape is input feature space
    """
    model = Sequential()
    # number of  hidden layer
    model.add(Dense(4096, input_dim=shape, activation=act))
    model.add(Dropout(0.8))

    for i in range(layers):
        model.add(Dense(neurons, activation=act))
        model.add(Dropout(dropout))

    model.add(Dense(64, activation=act))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="relu"))
    model.compile(
        optimizer=optimizer,
        loss="mape",
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(),
            tf.keras.metrics.MAPE,
            tf.keras.metrics.MSE,
            tf.keras.metrics.MAE,
        ],
    )
    return model


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    epochs,
    batch_size,
    optimizer,
    layers,
    activation,
    neurons,
    dropout,
):
    model_DNN = build_neural_network_model(
        X_train.shape[1], optimizer, layers, activation, neurons, dropout
    )
    # Confirm creation of model, and some details
    logging.info(f"Created model with {model_DNN.count_params()} parameters.")
    logging.info(
        f"Optimizer: {optimizer}, Activation: {activation}, Layers: {layers}, Neurons: {neurons}, Dropout: {dropout}"
    )
    history = model_DNN.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_mean_absolute_percentage_error", mode="min", patience=40
            )
        ],
    )

    return model_DNN, history


if __name__ == "__main__":
    data_init = load_data("code/data_pipeline/data/ml_data_reg_large.csv")
    # Checkpoint for successful loading of data, just a message
    logging.info(f"Loaded {len(data_init)} records from the dataset.")
    data = clean_data(
        data_init,
        columns_to_drop=[
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
        ],
        remove_outliers=True,
    )
    # checkpoint for successful cleaning of data, just a message
    logging.info(
        f"Cleaned the dataset and removed {len(data) - len(data_init)} records."
    )
    X_train, X_test, y_train, y_test = split_dataset(
        data, target_column="revenue_usd_adj"
    )
    # checkpoint for successful splitting of data, just a message
    logging.info(
        f"Split the dataset into train and test sets with {len(X_train)} and {len(X_test)} records respectively."
    )

    model, history = train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=400,
        batch_size=1024,
        optimizer="Adamax",
        layers=4,
        activation="elu",
        neurons=1024,
        dropout=0.8,
    )

    plot_training_history(history)

    model.save("model.h5")
    print(model.summary())
