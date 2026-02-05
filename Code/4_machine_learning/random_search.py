# Suppress all warnings (important for parallel workers)
import warnings
import os
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# Define general parameter distributions
PARAM_DISTRIBUTIONS = {
    "C": loguniform(1e-4, 1e2),
    "max_depth_tree": randint(2, 32),
    "max_depth_gbm": randint(2, 11),
    "n_estimators_rf": randint(200, 1200),
    "n_estimators_gbm": randint(200, 1400),
    "learning_rate_gbm": loguniform(5e-3, 2e-1),
    "num_leaves": [31, 63, 127, 255],
    "min_samples_split": randint(2, 40),
    "min_samples_leaf": randint(1, 20),
    "variance_threshold": uniform(0, 0.2),
    "max_iter": randint(200, 2000),
    "max_samples": uniform(0.5, 0.5),  # 0.5..1.0, only used when bootstrap=True
    "max_features": [None, "sqrt", "log2", 0.5, 0.8, 1.0],
    "bootstrap": [True, False],
    "estimator": [
        DecisionTreeRegressor(random_state=42),
        RandomForestRegressor(random_state=42),
        LogisticRegression(random_state=42),
    ],
    "units": randint(4, 2048),
    "dropout": uniform(0.01, 0.8),
    "num_layers": randint(2, 8),
    "epochs": [50, 100, 200],
    "optimizer": ["adam"],
    "layers_activation": ["relu", "tanh", "sigmoid"],
    "output_activation": ["linear", "relu"],
    "scaler": [
        StandardScaler(),
        MinMaxScaler(),
        RobustScaler(),
        PowerTransformer(method="yeo-johnson"),
    ],
    "solver": ["saga"],
    "l1_ratio": uniform(0, 1),  # 0=l2, 1=l1, in between=elasticnet
    "kernel": ["rbf"],
    "gamma_rbf": loguniform(1e-4, 1e0),
    "fit_intercept": [True, False],
    "classifier__strategy": ["most_frequent", "prior", "stratified", "uniform"],
    "regressor__strategy": ["mean", "median"],
    "use_log_transform": [True, False],
    # Common regularization/sampling knobs for boosted trees
    "subsample": uniform(0.7, 0.3),  # 0.7..1.0
    "colsample_bytree": uniform(0.7, 0.3),  # 0.7..1.0
    "min_child_weight": loguniform(0.5, 10.0),
    "reg_alpha": loguniform(1e-3, 10.0),
    "reg_lambda": loguniform(1e-3, 10.0),
    "min_child_samples": randint(5, 80),
    "ccp_alpha": loguniform(1e-4, 1e-1),
    "scale_pos_weight": loguniform(0.5, 20.0),
    # Bagging expects floats/ints for max_samples/max_features (no 'sqrt'/'log2')
    "bagging_n_estimators": randint(10, 200),
    "bagging_max_samples": uniform(0.3, 0.7),  # 0.3..1.0
    "bagging_max_features": uniform(0.3, 0.7),  # 0.3..1.0
}

# Define model-specific parameter distributions using the general distributions
MODEL_PARAM_DISTRIBUTIONS = {
    "logistic_regression": {
        "model__C": PARAM_DISTRIBUTIONS["C"],
        "model__solver": PARAM_DISTRIBUTIONS["solver"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS[
            "variance_threshold"
        ],
        "preprocessor__numerical__scaler": PARAM_DISTRIBUTIONS["scaler"],
    },
    "linear_regression": {
        "model__fit_intercept": PARAM_DISTRIBUTIONS["fit_intercept"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS[
            "variance_threshold"
        ],
        "preprocessor__numerical__scaler": PARAM_DISTRIBUTIONS["scaler"],
    },
    "random_forest_classifier": {
        "model__n_estimators": PARAM_DISTRIBUTIONS["n_estimators_rf"],
        "model__max_depth": [None],
        "model__min_samples_split": PARAM_DISTRIBUTIONS["min_samples_split"],
        "model__min_samples_leaf": PARAM_DISTRIBUTIONS["min_samples_leaf"],
        "model__max_features": PARAM_DISTRIBUTIONS["max_features"],
        "model__bootstrap": PARAM_DISTRIBUTIONS["bootstrap"],
    },
    "random_forest_regressor": {
        "model__n_estimators": PARAM_DISTRIBUTIONS["n_estimators_rf"],
        "model__max_depth": [None],
        "model__min_samples_split": PARAM_DISTRIBUTIONS["min_samples_split"],
        "model__min_samples_leaf": PARAM_DISTRIBUTIONS["min_samples_leaf"],
        "model__max_features": PARAM_DISTRIBUTIONS["max_features"],
        "model__bootstrap": PARAM_DISTRIBUTIONS["bootstrap"],
    },
    "decision_tree_classifier": {
        "model__max_depth": [None],
        "model__min_samples_split": PARAM_DISTRIBUTIONS["min_samples_split"],
        "model__min_samples_leaf": PARAM_DISTRIBUTIONS["min_samples_leaf"],
        "model__max_features": PARAM_DISTRIBUTIONS["max_features"],
        "model__ccp_alpha": PARAM_DISTRIBUTIONS["ccp_alpha"],
    },
    "decision_tree_regressor": {
        "model__max_depth": [None],
        "model__min_samples_split": PARAM_DISTRIBUTIONS["min_samples_split"],
        "model__min_samples_leaf": PARAM_DISTRIBUTIONS["min_samples_leaf"],
        "model__max_features": PARAM_DISTRIBUTIONS["max_features"],
        "model__ccp_alpha": PARAM_DISTRIBUTIONS["ccp_alpha"],
    },
    "xgboost_classifier": {
        "model__n_estimators": PARAM_DISTRIBUTIONS["n_estimators_gbm"],
        "model__max_depth": PARAM_DISTRIBUTIONS["max_depth_gbm"],
        "model__learning_rate": PARAM_DISTRIBUTIONS["learning_rate_gbm"],
        "model__subsample": PARAM_DISTRIBUTIONS["subsample"],
        "model__colsample_bytree": PARAM_DISTRIBUTIONS["colsample_bytree"],
        "model__min_child_weight": PARAM_DISTRIBUTIONS["min_child_weight"],
        "model__reg_alpha": PARAM_DISTRIBUTIONS["reg_alpha"],
        "model__reg_lambda": PARAM_DISTRIBUTIONS["reg_lambda"],
    },
    "xgboost_regressor": {
        "model__n_estimators": PARAM_DISTRIBUTIONS["n_estimators_gbm"],
        "model__max_depth": PARAM_DISTRIBUTIONS["max_depth_gbm"],
        "model__learning_rate": PARAM_DISTRIBUTIONS["learning_rate_gbm"],
        "model__subsample": PARAM_DISTRIBUTIONS["subsample"],
        "model__colsample_bytree": PARAM_DISTRIBUTIONS["colsample_bytree"],
        "model__min_child_weight": PARAM_DISTRIBUTIONS["min_child_weight"],
        "model__reg_alpha": PARAM_DISTRIBUTIONS["reg_alpha"],
        "model__reg_lambda": PARAM_DISTRIBUTIONS["reg_lambda"],
    },
    "lightgbm_regressor": {
        "model__num_leaves": PARAM_DISTRIBUTIONS["num_leaves"],
        "model__learning_rate": PARAM_DISTRIBUTIONS["learning_rate_gbm"],
        "model__n_estimators": PARAM_DISTRIBUTIONS["n_estimators_gbm"],
        "model__subsample": PARAM_DISTRIBUTIONS["subsample"],
        "model__subsample_freq": [1],
        "model__colsample_bytree": PARAM_DISTRIBUTIONS["colsample_bytree"],
        "model__min_child_samples": PARAM_DISTRIBUTIONS["min_child_samples"],
        "model__reg_alpha": PARAM_DISTRIBUTIONS["reg_alpha"],
        "model__reg_lambda": PARAM_DISTRIBUTIONS["reg_lambda"],
    },
    "lightgbm_classifier": {
        "model__num_leaves": PARAM_DISTRIBUTIONS["num_leaves"],
        "model__learning_rate": PARAM_DISTRIBUTIONS["learning_rate_gbm"],
        "model__n_estimators": PARAM_DISTRIBUTIONS["n_estimators_gbm"],
        "model__subsample": PARAM_DISTRIBUTIONS["subsample"],
        "model__subsample_freq": [1],
        "model__colsample_bytree": PARAM_DISTRIBUTIONS["colsample_bytree"],
        "model__min_child_samples": PARAM_DISTRIBUTIONS["min_child_samples"],
        "model__reg_alpha": PARAM_DISTRIBUTIONS["reg_alpha"],
        "model__reg_lambda": PARAM_DISTRIBUTIONS["reg_lambda"],
    },
    "svm_classifier": {
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS[
            "variance_threshold"
        ],
        "preprocessor__numerical__scaler": PARAM_DISTRIBUTIONS["scaler"],
    },
    "nn_classifier": {
        "model__model__units": PARAM_DISTRIBUTIONS["units"],
        "model__model__dropout": PARAM_DISTRIBUTIONS["dropout"],
        "model__model__num_layers": PARAM_DISTRIBUTIONS["num_layers"],
        "model__epochs": PARAM_DISTRIBUTIONS["epochs"],
        "model__optimizer": PARAM_DISTRIBUTIONS["optimizer"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS[
            "variance_threshold"
        ],
        "preprocessor__numerical__scaler": PARAM_DISTRIBUTIONS["scaler"],
    },
    "nn_regression": {
        "model__model__units": PARAM_DISTRIBUTIONS["units"],
        "model__model__dropout": PARAM_DISTRIBUTIONS["dropout"],
        "model__model__num_layers": PARAM_DISTRIBUTIONS["num_layers"],
        "model__epochs": PARAM_DISTRIBUTIONS["epochs"],
        "model__optimizer": PARAM_DISTRIBUTIONS["optimizer"],
        "model__model__layers_activation": PARAM_DISTRIBUTIONS["layers_activation"],
        "model__model__output_activation": PARAM_DISTRIBUTIONS["output_activation"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS[
            "variance_threshold"
        ],
        "preprocessor__numerical__scaler": PARAM_DISTRIBUTIONS["scaler"],
    },
    "dummy_classifier": {
        "model__strategy": PARAM_DISTRIBUTIONS["classifier__strategy"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS[
            "variance_threshold"
        ],
        "preprocessor__numerical__scaler": PARAM_DISTRIBUTIONS["scaler"],
    },
    "dummy_regressor": {
        "model__strategy": PARAM_DISTRIBUTIONS["regressor__strategy"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS[
            "variance_threshold"
        ],
        "preprocessor__numerical__scaler": PARAM_DISTRIBUTIONS["scaler"],
    },
    "bagging_regressor": {
        "model__estimator": PARAM_DISTRIBUTIONS["estimator"],
        "model__n_estimators": PARAM_DISTRIBUTIONS["bagging_n_estimators"],
        "model__max_samples": PARAM_DISTRIBUTIONS["bagging_max_samples"],
        "model__max_features": PARAM_DISTRIBUTIONS["bagging_max_features"],
        "model__bootstrap": PARAM_DISTRIBUTIONS["bootstrap"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS[
            "variance_threshold"
        ],
        "preprocessor__numerical__scaler": PARAM_DISTRIBUTIONS["scaler"],
    },
}


def perform_random_search(
    estimator,
    model_name,
    X,
    y,
    cv,
    n_iter,
    scoring,
    random_state,
    task_type,
    n_jobs=-1,
):
    param_distributions = MODEL_PARAM_DISTRIBUTIONS.get(model_name, {})

    # Some models need conditional parameter spaces.
    # RandomizedSearchCV supports a list of dicts to express such constraints.
    if model_name == "logistic_regression":
        base_space = {
            **param_distributions,
            "model__penalty": ["l1", "l2"],
        }
        elasticnet_space = {
            **param_distributions,
            "model__penalty": ["elasticnet"],
            "model__l1_ratio": PARAM_DISTRIBUTIONS["l1_ratio"],
        }
        param_distributions = [base_space, elasticnet_space]
    elif model_name == "svm_classifier":
        linear_space = {
            **param_distributions,
            "model__kernel": ["linear"],
            "model__C": PARAM_DISTRIBUTIONS["C"],
        }
        rbf_space = {
            **param_distributions,
            "model__kernel": ["rbf"],
            "model__C": PARAM_DISTRIBUTIONS["C"],
            "model__gamma": PARAM_DISTRIBUTIONS["gamma_rbf"],
        }
        param_distributions = [linear_space, rbf_space]
    elif model_name in ("random_forest_classifier", "random_forest_regressor"):
        # Only sample max_samples when bootstrap=True.
        base_space = {**param_distributions, "model__max_depth": [None]}
        boot_space = {**base_space, "model__bootstrap": [True], "model__max_samples": PARAM_DISTRIBUTIONS["max_samples"]}
        no_boot_space = {**base_space, "model__bootstrap": [False]}
        # Allow bounded depths too (metadata shows good depths ~10-25).
        depth_space = {**param_distributions, "model__max_depth": PARAM_DISTRIBUTIONS["max_depth_tree"]}
        boot_depth_space = {**depth_space, "model__bootstrap": [True], "model__max_samples": PARAM_DISTRIBUTIONS["max_samples"]}
        no_boot_depth_space = {**depth_space, "model__bootstrap": [False]}
        param_distributions = [boot_space, no_boot_space, boot_depth_space, no_boot_depth_space]
    elif model_name in ("decision_tree_classifier", "decision_tree_regressor"):
        # Allow either unlimited depth or a bounded depth search.
        none_depth = {**param_distributions, "model__max_depth": [None]}
        bounded_depth = {**param_distributions, "model__max_depth": PARAM_DISTRIBUTIONS["max_depth_tree"]}
        param_distributions = [none_depth, bounded_depth]

    class_weight_supported = {
        "logistic_regression",
        "random_forest_classifier",
        "decision_tree_classifier",
        "svm_classifier",
        "lightgbm_classifier",
    }
    if (
        task_type in ("binary_classification", "multi_class_classification")
        and model_name in class_weight_supported
    ):
        if isinstance(param_distributions, list):
            for space in param_distributions:
                space["model__class_weight"] = ["balanced", None]
        else:
            param_distributions["model__class_weight"] = ["balanced", None]

    # XGBoost doesn't accept `class_weight`; use `scale_pos_weight` for binary tasks.
    if task_type == "binary_classification" and model_name == "xgboost_classifier":
        param_distributions["model__scale_pos_weight"] = PARAM_DISTRIBUTIONS["scale_pos_weight"]

    random_search = RandomizedSearchCV(
        estimator,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        verbose=1,
        n_jobs=n_jobs,
    )
    random_search.fit(X, y)
    return random_search.best_estimator_, n_iter
