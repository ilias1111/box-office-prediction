from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# Define general parameter distributions
PARAM_DISTRIBUTIONS = {
    "C": loguniform(1e-4, 1e2),
    "max_depth": randint(2, 32),
    "n_estimators": randint(100, 2000),
    "learning_rate": uniform(0.001, 0.2),
    "num_leaves": randint(16, 256),
    "min_samples_split": randint(2, 32),
    "min_samples_leaf": randint(1, 20),
    "variance_threshold" : uniform(0, 0.2),
    "max_iter": randint(100, 1000),
    "max_samples": uniform(0.1, 1),
    "max_features": uniform(0.1, 1),
    "bootstrap": [True, False],
    "estimator" : [DecisionTreeRegressor(random_state=42) , RandomForestRegressor(random_state=42), LogisticRegression(random_state=42)],
    "units": randint(4, 2048),
    "dropout": uniform(0.01, 0.8),
    "num_layers": randint(2, 8),
    "epochs": [50, 100, 200],
    "optimizer": ["adam"],
    "layers_activation": ["relu", "tanh", "sigmoid"],
    "output_activation": ["linear", "relu"],
    "scaler": [StandardScaler(),MinMaxScaler(),RobustScaler()],
    "solver": ["saga"],
    "penalty": ["l1", "l2"],
    "kernel": ["rbf"],
    "gamma": ['scale', 'auto'],
    "fit_intercept" : [True, False],
    "classifier__strategy": ['most_frequent', 'prior', 'stratified', 'uniform'],
    "regressor__strategy":  ['mean', 'median'],
    "use_log_transform": [True, False]

}

# Define model-specific parameter distributions using the general distributions
MODEL_PARAM_DISTRIBUTIONS = {
    "logistic_regression": {
        "model__C": PARAM_DISTRIBUTIONS["C"],
        "model__solver": PARAM_DISTRIBUTIONS["solver"],
        "model__penalty": PARAM_DISTRIBUTIONS["penalty"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS["variance_threshold"],
        "preprocessor__numerical__scaler" : PARAM_DISTRIBUTIONS["scaler"]

    },

    "linear_regression" : {
        "model__fit_intercept" : PARAM_DISTRIBUTIONS["fit_intercept"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS["variance_threshold"],
        "preprocessor__numerical__scaler" : PARAM_DISTRIBUTIONS["scaler"]
    },

    "random_forest_classifier": {
        "model__n_estimators": PARAM_DISTRIBUTIONS["n_estimators"],
        "model__max_depth": PARAM_DISTRIBUTIONS["max_depth"],
        "model__min_samples_split": PARAM_DISTRIBUTIONS["min_samples_split"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS["variance_threshold"],
        "preprocessor__numerical__scaler" : PARAM_DISTRIBUTIONS["scaler"]
    },

    "random_forest_regressor":{
        "model__n_estimators": PARAM_DISTRIBUTIONS["n_estimators"],
        "model__max_depth": PARAM_DISTRIBUTIONS["max_depth"],
        "model__min_samples_split": PARAM_DISTRIBUTIONS["min_samples_split"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS["variance_threshold"],
        "preprocessor__numerical__scaler" : PARAM_DISTRIBUTIONS["scaler"]
    },

    "decision_tree_classifier": {
        "model__max_depth": PARAM_DISTRIBUTIONS["max_depth"],
        "model__min_samples_split": PARAM_DISTRIBUTIONS["min_samples_split"],
        "model__min_samples_leaf": PARAM_DISTRIBUTIONS["min_samples_leaf"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS["variance_threshold"],
        "preprocessor__numerical__scaler" : PARAM_DISTRIBUTIONS["scaler"]
    },

    "decision_tree_regressor": {
        "model__max_depth": PARAM_DISTRIBUTIONS["max_depth"],
        "model__min_samples_split": PARAM_DISTRIBUTIONS["min_samples_split"],
        "model__min_samples_leaf": PARAM_DISTRIBUTIONS["min_samples_leaf"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS["variance_threshold"],
        "preprocessor__numerical__scaler" : PARAM_DISTRIBUTIONS["scaler"]
    },

    "xgboost_classifier": {
        "model__n_estimators": PARAM_DISTRIBUTIONS["n_estimators"],
        "model__max_depth": PARAM_DISTRIBUTIONS["max_depth"],
        "model__learning_rate": PARAM_DISTRIBUTIONS["learning_rate"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS["variance_threshold"],
        "preprocessor__numerical__scaler" : PARAM_DISTRIBUTIONS["scaler"]
    },
    "xgboost_regressor": {
        "model__n_estimators": PARAM_DISTRIBUTIONS["n_estimators"],
        "model__max_depth": PARAM_DISTRIBUTIONS["max_depth"],
        "model__learning_rate": PARAM_DISTRIBUTIONS["learning_rate"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS["variance_threshold"],
        "preprocessor__numerical__scaler" : PARAM_DISTRIBUTIONS["scaler"]
    },

    "lightgbm_regressor" : {
        "model__num_leaves" :  PARAM_DISTRIBUTIONS["num_leaves"],
         "model__max_depth" : PARAM_DISTRIBUTIONS["max_depth"],
          "model__learning_rate": PARAM_DISTRIBUTIONS["learning_rate"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS["variance_threshold"],
        "preprocessor__numerical__scaler" : PARAM_DISTRIBUTIONS["scaler"]
    },

    "lightgbm_classifier" : {
        "model__num_leaves" :  PARAM_DISTRIBUTIONS["num_leaves"],
         "model__max_depth" : PARAM_DISTRIBUTIONS["max_depth"],
          "model__learning_rate": PARAM_DISTRIBUTIONS["learning_rate"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS["variance_threshold"],
        "preprocessor__numerical__scaler" : PARAM_DISTRIBUTIONS["scaler"]
    },

    "svm_classifier": {
        "model__C": PARAM_DISTRIBUTIONS["C"],
        "model__kernel": PARAM_DISTRIBUTIONS["kernel"],
        "model__gamma": PARAM_DISTRIBUTIONS["gamma"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS["variance_threshold"],
        "preprocessor__numerical__scaler" : PARAM_DISTRIBUTIONS["scaler"]
    },

    "nn_classifier" : {
        "model__model__units": PARAM_DISTRIBUTIONS["units"],
        "model__model__dropout": PARAM_DISTRIBUTIONS["dropout"],
        "model__model__num_layers": PARAM_DISTRIBUTIONS["num_layers"],
        "model__epochs": PARAM_DISTRIBUTIONS["epochs"],
        "model__optimizer": PARAM_DISTRIBUTIONS["optimizer"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS["variance_threshold"],
        "preprocessor__numerical__scaler" : PARAM_DISTRIBUTIONS["scaler"]
    },

    "nn_regression" : {
        "model__model__units": PARAM_DISTRIBUTIONS["units"],
        "model__model__dropout": PARAM_DISTRIBUTIONS["dropout"],
        "model__model__num_layers": PARAM_DISTRIBUTIONS["num_layers"],
        "model__epochs": PARAM_DISTRIBUTIONS["epochs"],
        "model__optimizer": PARAM_DISTRIBUTIONS["optimizer"],
        "model__model__layers_activation": PARAM_DISTRIBUTIONS["layers_activation"],
        "model__model__output_activation": PARAM_DISTRIBUTIONS["output_activation"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS["variance_threshold"],
        "preprocessor__numerical__scaler" : PARAM_DISTRIBUTIONS["scaler"]
    },

    "dummy_classifier": {
        "model__strategy": PARAM_DISTRIBUTIONS["classifier__strategy"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS["variance_threshold"],
        "preprocessor__numerical__scaler" : PARAM_DISTRIBUTIONS["scaler"]
    },

    "dummy_regressor": {
        "model__strategy": PARAM_DISTRIBUTIONS["regressor__strategy"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS["variance_threshold"],
        "preprocessor__numerical__scaler" : PARAM_DISTRIBUTIONS["scaler"]
    },

    "bagging_regressor" : {
        "model__estimator": PARAM_DISTRIBUTIONS["estimator"],
        "model__n_estimators": PARAM_DISTRIBUTIONS["n_estimators"],
        "model__max_samples": PARAM_DISTRIBUTIONS["max_samples"],
        "model__max_features": PARAM_DISTRIBUTIONS["max_features"],
        "model__bootstrap": PARAM_DISTRIBUTIONS["bootstrap"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS["variance_threshold"],
        "preprocessor__numerical__scaler" : PARAM_DISTRIBUTIONS["scaler"]
    }

}


def perform_random_search(estimator, model_name, X, y, cv, n_iter, scoring, random_state, task_type):

    param_distributions = MODEL_PARAM_DISTRIBUTIONS.get(model_name, {})

    if (task_type in ('binary_classification', 'multiclass_classification')) and (model_name not in ['dummy_classifier']):
        param_distributions['model__class_weight'] = ['balanced', None]
    
    random_search = RandomizedSearchCV(estimator, param_distributions, n_iter=n_iter, cv=cv, scoring=scoring, random_state=random_state, verbose=1, n_jobs=-1)
    random_search.fit(X, y)
    return random_search.best_estimator_, None