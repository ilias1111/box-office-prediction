"""Replay MOTR with frozen hyperparameters from prior metadata JSON(s).

Thesis protocol:
  --split-strategy temporal            → metrics-only forward-time check
  --split-strategy random --compute-shap → SHAP next to impurity FI
  SHAP is hard-gated: refused with temporal.

Examples:
  .venv/bin/python code/4_machine_learning/replay_from_metadata.py \\
      --run-id 20260207_190503 --split-strategy temporal

  .venv/bin/python code/4_machine_learning/replay_from_metadata.py \\
      --metadata metadata/full__binary_classification__no_outliers__complex__xgboost_classifier__20260207_190503.json \\
      --split-strategy random --compute-shap
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml_logic import MOTR  # noqa: E402

_DROP = {
    "model",
    "estimator",
    "callbacks",
    "kwargs",
    "n_jobs",
    "nthread",
    "thread_count",
    "device",
    "gpu_id",
    "verbosity",
    "verbose",
    "silent",
}
_SCALERS = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
    "PowerTransformer": lambda: PowerTransformer(method="yeo-johnson"),
}
_METRICS = (
    "ROC AUC Score",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "R2",
    "MAPE",
    "MAE",
    "RMSE",
)


def dataset_csv(meta: dict, data_dir: Path) -> Path:
    """Resolve ML-ready CSV from metadata fields (not filename parsing)."""
    outliers = {True: "no_outliers", False: "with_outliers"}[meta["has_outliers_removed"]]
    stem = (
        f"{meta['dataset_name']}__{meta['problem_type']}__"
        f"{outliers}__{meta['feature_engineering']}"
    )
    path = data_dir / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return path


def fixed_params(meta: dict) -> dict:
    """Map saved estimator parameters, scaler and variance threshold onto the Pipeline."""
    fixed: dict = {}
    for key, value in dict(meta.get("model_parameters") or {}).items():
        bare = key.split("model__", 1)[-1]
        if bare in _DROP:
            continue
        pipe_key = key if key.startswith("model__") else f"model__{key}"
        if bare == "class_weight" and isinstance(value, dict):
            value = {
                int(k) if isinstance(k, str) and k.isdigit() else k: v
                for k, v in value.items()
            }
        fixed[pipe_key] = value

    scaler = meta.get("scaler")
    if scaler in _SCALERS:
        fixed["preprocessor__numerical__scaler"] = _SCALERS[scaler]()
    if meta.get("variance_threshold") is not None:
        fixed["preprocessor__binary__variance_threshold__threshold"] = meta[
            "variance_threshold"
        ]
    return fixed


def resolve_sources(
    metadata: list[Path], run_id: str | None, metadata_dir: Path
) -> list[Path]:
    paths = [p.resolve() for p in metadata]
    if run_id:
        paths.extend(sorted(metadata_dir.glob(f"*__{run_id}.json")))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        if path in seen or not path.is_file() or path.name.startswith("_"):
            continue
        seen.add(path)
        out.append(path)
    if not out:
        raise FileNotFoundError("No metadata files found.")
    return out


def replay(
    meta_path: Path,
    *,
    split: str,
    data_dir: Path,
    run_id: str,
    compute_shap: bool,
    test_start_year: int | None,
) -> dict:
    meta = json.loads(meta_path.read_text())
    csv_path = dataset_csv(meta, data_dir)
    model_name = meta["model_type"]
    task = meta["problem_type"]
    target = "revenue_usd_adj" if task == "regression" else task
    # Hard gate: SHAP never runs on temporal.
    do_shap = bool(compute_shap and split == "random")
    cv_folds = int((meta.get("split_info") or {}).get("cv_folds") or 3)

    trainer = MOTR(
        f"{run_id}_{split}",
        str(csv_path),
        target,
        "movie_id",
        task_type=task,
        grid_type="non_grid",
        positive_class="Success",
        cv_folds=cv_folds,
        search_n_jobs=1,
        model_n_jobs=-1,
        enable_pipeline_cache=True,
        onehot_sparse=True,
        compute_feature_importance=True,
        compute_predicted_vs_actual=True,
        compute_shap=do_shap,
        shap_max_samples=400,
        save_model=False,
        save_metadata=True,
        split_strategy=split,
        test_size=0.2,
        test_start_year=test_start_year,
        fixed_model_params=fixed_params(meta),
        model_filter=[model_name],
    )
    trainer.split_info["replayed_from"] = str(meta_path)
    row = trainer.run(parallel=False).iloc[0].to_dict()

    out = {
        "source_metadata": str(meta_path),
        "model": model_name,
        "task_type": task,
        "split_strategy": split,
        "run_id": run_id,
        "shap_dir": (trainer.last_shap_summary or {}).get("dir"),
    }
    for key in _METRICS:
        if key in row and row[key] is not None:
            out[key] = row[key]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        action="append",
        default=[],
        help="Explicit metadata JSON path (repeatable).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Replay all metadata/*__{run_id}.json (prefer --metadata to be selective).",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("random", "temporal"),
        required=True,
    )
    parser.add_argument(
        "--compute-shap",
        action="store_true",
        help="Compute SHAP (random only; refused with temporal).",
    )
    parser.add_argument("--test-start-year", type=int, default=None)
    args = parser.parse_args()

    if not args.metadata and not args.run_id:
        parser.error("Provide --metadata and/or --run-id")
    if args.compute_shap and args.split_strategy == "temporal":
        parser.error("SHAP is not allowed with --split-strategy temporal")

    os.chdir(ROOT)
    sources = resolve_sources(args.metadata, args.run_id, ROOT / "metadata")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = ROOT / "metadata" / f"_replay_summary__{run_id}.csv"

    print(
        f"Replay run_id={run_id} n={len(sources)} "
        f"split={args.split_strategy} shap={args.compute_shap}",
        flush=True,
    )
    rows = []
    for meta_path in sources:
        print(f"\n=== {meta_path.name} ===", flush=True)
        row = replay(
            meta_path,
            split=args.split_strategy,
            data_dir=ROOT / "data/ml_ready_data",
            run_id=run_id,
            compute_shap=args.compute_shap,
            test_start_year=args.test_start_year,
        )
        rows.append(row)
        print({k: row[k] for k in _METRICS if k in row}, flush=True)

    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
