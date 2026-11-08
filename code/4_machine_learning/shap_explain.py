"""SHAP helpers for fitted sklearn Pipelines used by MOTR.

Expected pipeline step names: ``preprocessor``, ``model``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

TREE_MODEL_PREFIXES = (
    "xgboost_",
    "lightgbm_",
    "random_forest_",
    "decision_tree_",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShapResult:
    summary: pd.DataFrame
    values_sample: np.ndarray
    feature_names: tuple[str, ...]
    n_rows: int


def _to_dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


def _mean_abs_shap(shap_values, *, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (per-feature mean |SHAP|, sample matrix for optional NPZ dump)."""
    if isinstance(shap_values, list):
        stacked = np.stack([np.asarray(sv) for sv in shap_values], axis=0)
        abs_mean = np.abs(stacked).mean(axis=(0, 1))
        sample_matrix = np.abs(stacked).mean(axis=0)
        return np.asarray(abs_mean).ravel(), np.asarray(sample_matrix)

    values = np.asarray(shap_values)
    if values.ndim == 3:
        # (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        if values.shape[1] == n_features:
            abs_mean = np.abs(values).mean(axis=(0, 2))
            sample_matrix = np.abs(values).mean(axis=2)
        else:
            abs_mean = np.abs(values).mean(axis=(0, 1))
            sample_matrix = np.abs(values).mean(axis=0)
        return np.asarray(abs_mean).ravel(), np.asarray(sample_matrix)

    if values.ndim != 2:
        raise ValueError(f"Unsupported SHAP value shape: {values.shape}")
    return np.abs(values).mean(axis=0), values


def _build_explainer(estimator, model_name: str, X_transformed: np.ndarray):
    import shap

    if model_name.startswith(TREE_MODEL_PREFIXES):
        return shap.TreeExplainer(estimator)
    if hasattr(estimator, "coef_"):
        masker = shap.maskers.Independent(
            X_transformed, max_samples=min(100, len(X_transformed))
        )
        return shap.LinearExplainer(estimator, masker)
    return None


def explain_pipeline(
    pipeline_model,
    X_explain: pd.DataFrame,
    model_name: str,
    *,
    max_samples: int = 500,
    random_state: int = 42,
) -> Optional[ShapResult]:
    """Compute mean |SHAP| for a fitted MOTR pipeline. No disk writes."""
    if "dummy" in model_name:
        return None
    try:
        import shap  # noqa: F401
    except ImportError:
        logger.warning("shap is not installed; skipping SHAP for %s", model_name)
        return None

    if not hasattr(pipeline_model, "named_steps"):
        raise TypeError("pipeline_model must be a fitted sklearn Pipeline")
    missing = {"preprocessor", "model"} - set(pipeline_model.named_steps)
    if missing:
        raise ValueError(f"Pipeline missing required steps: {sorted(missing)}")

    preprocessor = pipeline_model.named_steps["preprocessor"]
    estimator = pipeline_model.named_steps["model"]
    feature_names = tuple(map(str, preprocessor.get_feature_names_out()))

    explain_frame = X_explain
    if len(explain_frame) > max_samples:
        explain_frame = explain_frame.sample(max_samples, random_state=random_state)
    if explain_frame.empty:
        logger.warning("Empty explain frame for %s; skipping SHAP", model_name)
        return None

    X_transformed = _to_dense(preprocessor.transform(explain_frame))
    try:
        explainer = _build_explainer(estimator, model_name, X_transformed)
        if explainer is None:
            logger.info("No SHAP explainer wired for %s; skipping", model_name)
            return None
        shap_values = explainer.shap_values(X_transformed)
    except Exception:
        logger.exception("SHAP failed for %s", model_name)
        return None

    abs_mean, sample_matrix = _mean_abs_shap(
        shap_values, n_features=len(feature_names)
    )
    summary = (
        pd.DataFrame(
            {
                "feature": list(feature_names),
                "mean_abs_shap": abs_mean[: len(feature_names)],
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return ShapResult(
        summary=summary,
        values_sample=np.asarray(sample_matrix, dtype=np.float32),
        feature_names=feature_names,
        n_rows=int(len(explain_frame)),
    )


def save_shap_artifacts(
    result: ShapResult,
    output_dir: str | Path,
    *,
    top_k: int = 20,
) -> dict[str, Any]:
    """Persist CSV + NPZ and return a metadata snippet for run JSON."""
    shap_dir = Path(output_dir)
    shap_dir.mkdir(parents=True, exist_ok=True)
    summary_path = shap_dir / "mean_abs_shap.csv"
    values_path = shap_dir / "shap_values_sample.npz"
    result.summary.to_csv(summary_path, index=False)
    np.savez_compressed(
        values_path,
        shap_values=result.values_sample,
        feature_names=np.asarray(result.feature_names),
        n_rows=np.asarray(result.n_rows),
    )
    logger.info("Saved SHAP artifacts to %s", shap_dir)
    return {
        "dir": str(shap_dir),
        "summary_csv": str(summary_path),
        "values_npz": str(values_path),
        "n_rows": result.n_rows,
        "top_features": result.summary.head(top_k).to_dict(orient="records"),
    }


def compute_shap_summary(
    pipeline_model,
    X_explain: pd.DataFrame,
    model_name: str,
    *,
    output_dir: str | Path,
    max_samples: int = 500,
    random_state: int = 42,
    top_k: int = 20,
) -> Optional[dict[str, Any]]:
    """Explain + save convenience wrapper used by MOTR."""
    result = explain_pipeline(
        pipeline_model,
        X_explain,
        model_name,
        max_samples=max_samples,
        random_state=random_state,
    )
    if result is None:
        return None
    return save_shap_artifacts(result, output_dir, top_k=top_k)
