#!/usr/bin/env python3
"""Export saved experiment results as thesis tables. Optional; never trains models."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "docs/thesis_results/results_metadata.json"
TABLES = ROOT.parent / "box-office-prediction-thesis/latex/tables"
RUNS = ("20260207_215143", "20260213_084518")
SEGMENTS = {"full": "Full", "large_productions": "Large",
            "medium_productions": "Medium", "small_productions": "Small"}
MODELS = {"dummy": "Dummy", "logistic_regression": "Logistic Regression",
          "decision_tree": "Decision Tree", "random_forest": "Random Forest",
          "xgboost": "XGBoost", "lightgbm": "LightGBM"}


def selection_score(record, metric):
    """Crown regression on log-space R² once a run has recorded it."""
    metrics = record["metrics"]
    if metric == "R2" and "R2 (log10)" in metrics:
        return metrics["R2 (log10)"]
    return metrics[metric]


def model_name(value):
    for prefix, label in MODELS.items():
        if value.startswith(prefix):
            return label
    raise ValueError(value)


def write_table(name, caption, label, headings, rows, resize=False):
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{" + caption + "}", r"\label{" + label + "}"]
    if resize:
        lines.append(r"\resizebox{\linewidth}{!}{%")
    columns = "l" * len(headings)
    if name == "table_best_reg.tex":
        columns = r"l@{\hspace{1.5em}}" + "l" * (len(headings) - 1)
    lines += [r"\begin{tabular}{" + columns + "}", r"\toprule",
              " & ".join(headings) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if resize:
        lines.append("}")
    lines += [r"\end{table}", ""]
    (TABLES / name).write_text("\n".join(lines))


def archive(directory):
    records = []
    keys = ("run_id", "timestamp", "model_type", "problem_type", "dataset_name",
            "has_outliers_removed", "feature_engineering", "grid_type", "scaler",
            "metrics", "feature_importance", "class_report", "conf_matrix",
            "predicted_vs_actual")
    for path in sorted(directory.glob("*.json")):
        raw = path.read_bytes()
        data = json.loads(raw)
        if data["run_id"] in RUNS:
            record = {key: data.get(key) for key in keys}
            record.update(source_file=path.name, source_sha256=hashlib.sha256(raw).hexdigest())
            records.append(record)
    if len(records) != 320:
        raise ValueError(f"Expected 320 experiments in the selected runs, found {len(records)}")
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps({
        "status": "Historical artifacts; unfiltered results include a revenue-derived flag and are provisional.",
        "run_ids": RUNS,
        "reporting_selection": {"regression": "max held-out R2", "classification": "max reported held-out F1"},
        "experiments": records,
    }, indent=2) + "\n")


def main():
    global TABLES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, help="Refresh the snapshot from original JSON files")
    parser.add_argument("--tables-dir", type=Path, default=TABLES,
                        help="Destination (default: sibling thesis repo's latex/tables)")
    parser.add_argument("--charts", action="store_true",
                        help="Also regenerate all data-derived charts referenced by the thesis")
    args = parser.parse_args()
    TABLES = args.tables_dir.resolve()
    TABLES.mkdir(parents=True, exist_ok=True)
    if args.metadata_dir:
        archive(args.metadata_dir)
    records = json.loads(SNAPSHOT.read_text())["experiments"]
    provenance = []
    for task, suffix in [("regression", "reg"), ("binary_classification", "bin_class"),
                         ("multi_class_classification", "multi_class")]:
        rows = []
        metric = "R2" if task == "regression" else "F1 Score"
        for dataset, display in SEGMENTS.items():
            for removed in (False, True):
                for features in ("complex", "none"):
                    group = [d for d in records if d["problem_type"] == task
                             and d["dataset_name"] == dataset
                             and d["has_outliers_removed"] == removed
                             and d["feature_engineering"] == features]
                    winner = max(group, key=lambda d: selection_score(d, metric))
                    base = [display, "Yes" if removed else "No", features.title(),
                            model_name(winner["model_type"])]
                    keys = ["MAPE", "Threshold Probability Accuracy (log10)", "R2"] if task == "regression" else ["F1 Score", "Accuracy"]
                    rows.append(base + [f'{winner["metrics"][k]:.3f}' for k in keys])
                    provenance.append({k: winner[k] for k in ("problem_type", "dataset_name",
                                      "has_outliers_removed", "feature_engineering", "model_type",
                                      "run_id", "source_file", "source_sha256")})
        if task == "regression":
            ranks_log_r2 = any(
                d["problem_type"] == "regression" and "R2 (log10)" in d["metrics"]
                for d in records
            )
            if ranks_log_r2:
                caption = r"Best regression models (highest held-out log10 $R^2$ per setting; dollar $R^2$ and MAPE shown; MAPE is a ratio)"
            else:
                caption = r"Best historical regression models (highest held-out $R^2$ per setting; MAPE is a ratio; unfiltered results provisional)"
            headings = ["Dataset", "Outliers removed", "Features", "Model", "MAPE", r"TP Acc.\ (log)", r"$R^2$"]
        else:
            metric_label = "Success F1" if task == "binary_classification" else "weighted F1"
            caption = f"Best {'binary' if task == 'binary_classification' else 'multi-class'} classification models ({metric_label}; unfiltered results provisional)"
            headings = ["Dataset", "Outliers removed", "Features", "Model", "F1", "Accuracy"]
        write_table(f"table_best_{suffix}.tex", caption, f"tab:best_{suffix}", headings, rows, resize=True)

    for suffix, task, metric, operation in [
        ("r2_reg", "regression", "R2", max), ("mape_reg", "regression", "MAPE", min),
        ("f1_bin", "binary_classification", "F1 Score", max),
        ("f1_multi", "multi_class_classification", "F1 Score", max)]:
        rows = []
        for dataset, display in SEGMENTS.items():
            for removed in (False, True):
                for features in ("complex", "none"):
                    values = [selection_score(d, metric) for d in records if d["problem_type"] == task
                              and d["dataset_name"] == dataset and d["has_outliers_removed"] == removed
                              and d["feature_engineering"] == features]
                    rows.append([display, "Yes" if removed else "No", features.title(), f"{operation(values):.3f}"])
        title = {"r2_reg": r"Maximum regression $R^2$", "mape_reg": "Minimum regression MAPE (ratio)",
                 "f1_bin": "Maximum binary Success F1", "f1_multi": "Maximum multi-class weighted F1"}[suffix]
        if suffix == "r2_reg" and any(
            d["problem_type"] == "regression" and "R2 (log10)" in d["metrics"] for d in records
        ):
            title = r"Maximum regression log10 $R^2$"
        write_table(f"table_dataset_comp_{suffix}.tex", title + "; unfiltered results provisional",
                    f"tab:dataset_comp_{suffix}", ["Dataset", "Outliers removed", "Features", metric], rows)

    # Keep each detailed model row internally consistent: choose its highest-R2
    # run, retaining missing per-film residual statistics as unavailable.
    for dataset, display in SEGMENTS.items():
        for removed in (False, True):
            for features in ("complex", "none"):
                group = [d for d in records if d["problem_type"] == "regression"
                         and d["dataset_name"] == dataset and d["has_outliers_removed"] == removed
                         and d["feature_engineering"] == features]
                winners = [max([d for d in group if d["model_type"] == model], key=lambda d: selection_score(d, "R2"))
                           for model in sorted({d["model_type"] for d in group})]
                rows = []
                for d in sorted(winners, key=lambda d: selection_score(d, "R2"), reverse=True):
                    pva = d.get("predicted_vs_actual")
                    while isinstance(pva, str):
                        pva = json.loads(pva)
                    pva = pva or {}
                    def statistic(column, stat):
                        value = (pva.get(column) or {}).get(stat)
                        return f"{value:,.0f}" if value is not None else "---"
                    actual = (pva.get("actual") or {}).get("mean")
                    predicted = (pva.get("predicted") or {}).get("mean")
                    bias = f"{actual-predicted:,.0f}" if actual is not None and predicted is not None else "---"
                    m = d["metrics"]
                    rows.append([model_name(d["model_type"]), f'{m["R2"]:.3f}', f'{m["MAE"]:,.0f}',
                                 statistic("absolute_error", "50%"), f'{m["RMSE"]:,.0f}',
                                 f'{m["MAPE"] * 100:.2f}' + r"\%",
                                 f'{m["Threshold Probability Accuracy (log10)"]:.3%}'.replace("%", r"\%"),
                                 bias, statistic("absolute_error", "max")])
                tag = "no_outliers" if removed else "with_outliers"
                rank_label = r"Highest log10-$R^2$ run per model; dollar $R^2$ shown" if any(
                    "R2 (log10)" in d["metrics"] for d in group
                ) else r"Highest-$R^2$ run per model"
                caption = f"Historical regression results: {display}, {tag.replace('_', ' ')}, {features} features. {rank_label}; --- denotes unavailable residual statistics."
                if not removed:
                    caption += " Results are provisional."
                write_table(f"table_reg_results_{dataset}_{tag}_{features}.tex", caption,
                            f"tab:reg_{dataset}_{tag}_{features}",
                            ["Model", r"$R^2$", "MAE", "Median AE", "RMSE", r"MAPE (\%)",
                             "TP Acc. (log)", "Bias", "Max error"], rows, resize=True)
    (SNAPSHOT.parent / "selected_models.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Rebuilt 23 tables from {len(records)} archived experiments; no models trained.")
    if args.charts:
        subprocess.run([sys.executable, str(ROOT / "code/_export_thesis_charts.py"),
                        "--all", "--thesis-dir", str(TABLES.parent.parent)], check=True)


if __name__ == "__main__":
    main()
