from typing import Dict
import keras
from keras import layers


def build_neural_network_model(
    meta: Dict[str, int],
    units: int,
    num_layers: int,
    dropout: float = 0.2,
    layers_activation: str = "relu",
    output_activation: str = "linear",
    loss: str = "mean_absolute_percentage_error",
):
    n_features_in_ = meta["n_features_in_"]

    model = keras.models.Sequential()
    model.add(layers.Input(shape=(n_features_in_,)))

    for _ in range(num_layers):
        model.add(layers.Dense(units, activation=layers_activation))
        model.add(layers.Dropout(dropout))

    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation=output_activation))
    model.compile(optimizer="adam", loss=loss, metrics=[loss])
    return model
