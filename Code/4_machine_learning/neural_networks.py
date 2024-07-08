# import the libraries 
import keras

def build_clf(meta, unit, dropout=0.2):
    n_features_in_ = meta["n_features_in_"]

    ann = keras.models.Sequential([
        keras.layers.Input(shape=(n_features_in_,)),
        keras.layers.Dense(units=unit, activation='relu'),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(units=unit, activation='relu'),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return ann

def build_regressor(meta, unit, dropout=0.2):
    n_features_in_ = meta["n_features_in_"]

    ann = keras.models.Sequential([
        keras.layers.Input(shape=(n_features_in_,)),
        keras.layers.Dense(units=unit, activation='relu'),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(units=unit, activation='relu'),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(units=1, activation='linear')
    ])
    ann.compile(optimizer='adam', loss='mean_absolute_percentage_error', metrics=['mean_absolute_percentage_error'])
    return ann
