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
from sklearn.linear_model import LogisticRegression, LinearRegression

# Define GENERAL "EXTREME" parameter distributions
# Wider ranges, more granularity.
PARAM_DISTRIBUTIONS = {
    "C": loguniform(1e-5, 1e3), # Extended range
    "max_depth_tree": randint(2, 64), # Deeper trees
    "max_depth_gbm": randint(2, 25), # Deeper GBMs allowed on GPU
    "n_estimators_rf": randint(500, 5000), # More trees
    "n_estimators_gbm": randint(500, 10000), # MUCH more trees (GPU can handle it)
    "learning_rate_gbm": loguniform(1e-4, 0.3), # Slower learning rates allowed
    
    # Correlated spaces (learning-rate vs number of trees)
    # Low LR needs MANY trees
    "n_estimators_gbm_low_lr": randint(3000, 10000),
    "learning_rate_gbm_low": loguniform(1e-4, 1e-2),
    
    # Mid LR
    "n_estimators_gbm_mid_lr": randint(1000, 5000),
    "learning_rate_gbm_mid": loguniform(1e-2, 0.1),
    
    # High LR (Fast convergence, fewer trees)
    "n_estimators_gbm_high_lr": randint(100, 1000),
    "learning_rate_gbm_high": loguniform(0.1, 0.5),

    "num_leaves": randint(20, 1024), # More leaves for complex patterns
    "min_samples_split": randint(2, 100),
    "min_samples_leaf": randint(1, 50),
    "variance_threshold": uniform(0, 0.15), # Tighter threshold
    "max_iter": randint(500, 5000),
    "max_samples": uniform(0.4, 0.6),
    "max_features": [None, "sqrt", "log2", 0.3, 0.5, 0.7, 0.9],
    "bootstrap": [True, False],
    "estimator": [
        DecisionTreeRegressor(random_state=42),
        RandomForestRegressor(random_state=42),
        LinearRegression(),
    ],
    "units": randint(16, 4096), # Larger NN layers
    "dropout": uniform(0.0, 0.7),
    "num_layers": randint(2, 10),
    "epochs": [100, 200, 500, 1000], # More epochs
    "optimizer": ["adam", "rmsprop"], 
    "layers_activation": ["relu", "leaky_relu", "tanh", "swish"],
    "output_activation": ["linear", "relu"],
    "scaler": [
        StandardScaler(),
        MinMaxScaler(),
        RobustScaler(),
        PowerTransformer(method="yeo-johnson"),
    ],
    "solver": ["saga", "lbfgs"],
    "l1_ratio": uniform(0, 1),
    "kernel": ["rbf", "poly", "sigmoid"],
    "gamma_rbf": loguniform(1e-5, 1e1),
    "fit_intercept": [True, False],
    "classifier__strategy": ["most_frequent", "prior", "stratified", "uniform"],
    "regressor__strategy": ["mean", "median"],
    "use_log_transform": [True, False],
    
    # Boosting Regularization
    "subsample": uniform(0.5, 0.5),  # 0.5..1.0
    "colsample_bytree": uniform(0.5, 0.5),  # 0.5..1.0
    "min_child_weight": loguniform(0.1, 100.0),
    "reg_alpha": loguniform(1e-4, 100.0),
    "reg_lambda": loguniform(1e-4, 100.0),
    "min_child_samples": randint(5, 200),
    "ccp_alpha": loguniform(1e-5, 1e-1),
    
    "scale_pos_weight": loguniform(0.1, 50.0),
    "gamma_xgb": loguniform(1e-5, 10.0),
    "min_split_gain_lgbm": uniform(0.0, 0.5),
    
    # Bagging
    "bagging_n_estimators": randint(50, 500),
    "bagging_max_samples": uniform(0.4, 0.6),
    "bagging_max_features": uniform(0.4, 0.6),
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
        "model__max_depth": PARAM_DISTRIBUTIONS["max_depth_gbm"],
        "model__subsample": PARAM_DISTRIBUTIONS["subsample"],
        "model__colsample_bytree": PARAM_DISTRIBUTIONS["colsample_bytree"],
        "model__min_child_weight": PARAM_DISTRIBUTIONS["min_child_weight"],
        "model__gamma": PARAM_DISTRIBUTIONS["gamma_xgb"],
        "model__reg_alpha": PARAM_DISTRIBUTIONS["reg_alpha"],
        "model__reg_lambda": PARAM_DISTRIBUTIONS["reg_lambda"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS[
            "variance_threshold"
        ],
    },
    "xgboost_regressor": {
        "model__max_depth": PARAM_DISTRIBUTIONS["max_depth_gbm"],
        "model__subsample": PARAM_DISTRIBUTIONS["subsample"],
        "model__colsample_bytree": PARAM_DISTRIBUTIONS["colsample_bytree"],
        "model__min_child_weight": PARAM_DISTRIBUTIONS["min_child_weight"],
        "model__gamma": PARAM_DISTRIBUTIONS["gamma_xgb"],
        "model__reg_alpha": PARAM_DISTRIBUTIONS["reg_alpha"],
        "model__reg_lambda": PARAM_DISTRIBUTIONS["reg_lambda"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS[
            "variance_threshold"
        ],
    },
    "lightgbm_regressor": {
        "model__num_leaves": PARAM_DISTRIBUTIONS["num_leaves"],
        "model__max_depth": [-1, 10, 20, 30],
        "model__subsample": PARAM_DISTRIBUTIONS["subsample"],
        "model__subsample_freq": [1, 5],
        "model__colsample_bytree": PARAM_DISTRIBUTIONS["colsample_bytree"],
        "model__min_child_samples": PARAM_DISTRIBUTIONS["min_child_samples"],
        "model__min_split_gain": PARAM_DISTRIBUTIONS["min_split_gain_lgbm"],
        "model__reg_alpha": PARAM_DISTRIBUTIONS["reg_alpha"],
        "model__reg_lambda": PARAM_DISTRIBUTIONS["reg_lambda"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS[
            "variance_threshold"
        ],
    },
    "lightgbm_classifier": {
        "model__num_leaves": PARAM_DISTRIBUTIONS["num_leaves"],
        "model__max_depth": [-1, 10, 20, 30],
        "model__subsample": PARAM_DISTRIBUTIONS["subsample"],
        "model__subsample_freq": [1, 5],
        "model__colsample_bytree": PARAM_DISTRIBUTIONS["colsample_bytree"],
        "model__min_child_samples": PARAM_DISTRIBUTIONS["min_child_samples"],
        "model__min_split_gain": PARAM_DISTRIBUTIONS["min_split_gain_lgbm"],
        "model__reg_alpha": PARAM_DISTRIBUTIONS["reg_alpha"],
        "model__reg_lambda": PARAM_DISTRIBUTIONS["reg_lambda"],
        "preprocessor__binary__variance_threshold__threshold": PARAM_DISTRIBUTIONS[
            "variance_threshold"
        ],
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
    elif model_name in ("xgboost_classifier", "xgboost_regressor"):
        base_space = param_distributions
        low_lr = {
            **base_space,
            "model__learning_rate": PARAM_DISTRIBUTIONS["learning_rate_gbm_low"],
            "model__n_estimators": PARAM_DISTRIBUTIONS["n_estimators_gbm_low_lr"],
        }
        mid_lr = {
            **base_space,
            "model__learning_rate": PARAM_DISTRIBUTIONS["learning_rate_gbm_mid"],
            "model__n_estimators": PARAM_DISTRIBUTIONS["n_estimators_gbm_mid_lr"],
        }
        high_lr = {
            **base_space,
            "model__learning_rate": PARAM_DISTRIBUTIONS["learning_rate_gbm_high"],
            "model__n_estimators": PARAM_DISTRIBUTIONS["n_estimators_gbm_high_lr"],
        }
        param_distributions = [low_lr, mid_lr, high_lr]
    elif model_name in ("lightgbm_classifier", "lightgbm_regressor"):
        base_space = param_distributions
        low_lr = {
            **base_space,
            "model__learning_rate": PARAM_DISTRIBUTIONS["learning_rate_gbm_low"],
            "model__n_estimators": PARAM_DISTRIBUTIONS["n_estimators_gbm_low_lr"],
        }
        mid_lr = {
            **base_space,
            "model__learning_rate": PARAM_DISTRIBUTIONS["learning_rate_gbm_mid"],
            "model__n_estimators": PARAM_DISTRIBUTIONS["n_estimators_gbm_mid_lr"],
        }
        high_lr = {
            **base_space,
            "model__learning_rate": PARAM_DISTRIBUTIONS["learning_rate_gbm_high"],
            "model__n_estimators": PARAM_DISTRIBUTIONS["n_estimators_gbm_high_lr"],
        }
        param_distributions = [low_lr, mid_lr, high_lr]

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
        if isinstance(param_distributions, list):
            for space in param_distributions:
                space["model__scale_pos_weight"] = PARAM_DISTRIBUTIONS["scale_pos_weight"]
        else:
            param_distributions["model__scale_pos_weight"] = PARAM_DISTRIBUTIONS["scale_pos_weight"]

    random_search = RandomizedSearchCV(
        estimator,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        verbose=2,
        n_jobs=n_jobs,
    )
    random_search.fit(X, y)
    return random_search.best_estimator_, n_iter
