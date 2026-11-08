#!/usr/bin/env python3
"""Export existing thesis chart definitions using Plotly/Kaleido and Chrome.

Stages native PNG exports and checks LaTeX-reference coverage before publishing.
"""
import argparse
import json
import importlib.util
from pathlib import Path
import re
import sys
import shutil
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
THESIS_ROOT = ROOT.parent / "box-office-prediction-thesis"
import plotly.io as pio
import numpy as np
import pandas as pd

FIGURES = {}


def collect_chart(chart, filename):
    """Keep the original Plotly figure, with space for its native labels."""
    bars = [trace for trace in chart.data if trace.type == "bar"]
    if bars:
        horizontal = bars[0].orientation == "h"
        axis = chart.layout.xaxis if horizontal else chart.layout.yaxis
        values = np.concatenate([np.asarray(t.x if horizontal else t.y, dtype=float)
                                 for t in chart.data if t.type in ("bar", "scatter")])
        values = values[np.isfinite(values)]
        if chart.layout.barmode == "stack":
            totals = {}
            for trace in bars:
                for category, value in zip(trace.y if horizontal else trace.x,
                                           trace.x if horizontal else trace.y):
                    totals[category] = totals.get(category, 0.0) + float(value)
            values = np.asarray(list(totals.values()), dtype=float)
        extra = [a.y for a in chart.layout.annotations or ()
                 if not horizontal and isinstance(a.y, (int, float)) and a.yref != "paper"]
        if extra:
            values = np.append(values, extra)
        if len(values) and not axis.range:
            low, high = min(0.0, float(values.min())), max(0.0, float(values.max()))
            if axis.type == "log" and values.min() > 0:
                bounds = [np.log10(values.min()) - .15, np.log10(high) + .25]
            else:
                axis.type = "linear"
                padding = max(high-low, 1e-9) * .18
                bounds = [low-padding if low < 0 else 0, high+padding]
            axis.range = bounds
        chart.update_traces(cliponaxis=False, selector=dict(type="bar"))
        chart.update_xaxes(automargin=True)
        chart.update_yaxes(automargin=True)
    FIGURES[filename] = chart
    print("Prepared", filename)


def render_legacy_data_charts(experiments, snapshot):
    """Reuse existing plotting functions; do not execute retrieval notebooks."""
    import eda_support_functions as support
    with patch.object(support, "save_figure", side_effect=lambda fig, name, *a, **k: collect_chart(fig, name)), \
         patch.object(support, "save_markdown_table", return_value=None):
        # Same inputs, thresholds and ordering as the retrieval notebook.
        for file, column, nickname, threshold in [
            ("production_companies", "company_name", "company", 300),
            ("cast", "name", "actor", 100), ("genres", "genre_name", "genre", 300),
            ("keywords", "keyword_name", "keyword", 300),
            ("collections", "collection_name", "collection", 10),
            ("spoken_languages", "english_name", "language", 100),
            ("production_countries", "name", "country", 500)]:
            frame = pd.read_csv(experiments / f"data/retrieved_data/tmdb/{file}.csv", usecols=[column])
            support.plot_and_export_categorical_distribution(
                frame, column, nickname, threshold, display_chart=False,
                output_dir=None, filename_prefix="step_1",
                count_label="Number of Cast Credits" if nickname == "actor" else "Number of Movies")
        movie = pd.read_csv(experiments / "data/processed_data/movie.csv", low_memory=False)
        for column in ("year", "quarter", "month", "release_category", "runtime", "ageCert"):
            support.plot_and_export_categorical_distribution(
                movie[[column]].copy(), column, column, 0, sort_by_value=False,
                display_chart=False, output_dir=None, format_as_int=True, filename_prefix="step_2")

        feature_spec = importlib.util.spec_from_file_location(
            "feature_eda", experiments / "code/_eda_3_feature_eng_analysis.py")
        feature = importlib.util.module_from_spec(feature_spec)
        feature_spec.loader.exec_module(feature)
        feature.ensure_output_dirs = lambda: None
        feature.save_step3_table = lambda *a, **k: None
        feature.save_step3_figure = collect_chart
        feature.run_feature_engineering_analysis(str(experiments / "data/ml_ready_data"))

        # Use the same archived records as the table exporter, not an implicit
        # newest-run selection. Each metric is aggregated within model/setting.
        records = json.loads(snapshot.read_text())["experiments"]
        metrics = pd.DataFrame([{**r, **r["metrics"]} for r in records])
        for task, metric, baseline in [
            ("regression", "R2", "dummy_regressor"),
            ("binary_classification", "F1 Score", "dummy_classifier"),
            ("multi_class_classification", "F1 Score", "dummy_classifier")]:
            support.plot_one_metric_of_different_models_per_dataset_with_plotly(
                metrics, problem_type=task, metric=metric, metric_agg="max",
                benchmark_model=baseline, display_chart=False, output_dir=None,
                print_stats=False, filename_prefix="step_4")
        # The two included matrix captions explicitly name Full/Complex/LightGBM.
        # Select the best matching run and retain that very record's matrix.
        for task, classes in [("binary_classification", 2), ("multi_class_classification", 4)]:
            candidates = [r for r in records if r["problem_type"] == task
                          and r["dataset_name"] == "full" and r["feature_engineering"] == "complex"
                          and not r["has_outliers_removed"] and r["model_type"] == "lightgbm_classifier"]
            winner = max(candidates, key=lambda r: r["metrics"]["F1 Score"])
            matrix = winner["conf_matrix"]
            if classes == 2:
                labels = {"0": "Bankrupt", "1": "Success"}
                matrix = [{labels.get(str(k), k): v for k, v in row.items()} for row in matrix]
            filename = f"step_4_matrix_classes_{classes}_dataset_full_model_lightgbm_classifier_fe_complex_outliers_false.png"
            support.plot_confusion_matrix(
                matrix, display_chart=False, output_dir=None, filename=filename,
                title=f"Confusion Matrix for Full\nLightGBM | Complex | Outliers retained\nProvisional historical results")


def required_charts(thesis):
    """The LaTeX references define the coverage requirement; static art stays put."""
    names = set()
    for path in (thesis / "latex/chapters").glob("*.tex"):
        text = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text())
        for ref in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
            name = Path(ref).name
            if name.startswith(("step_1_", "step_2_", "step_3_", "step_4_", "eda_")):
                names.add(name)
    if not names:
        raise ValueError("No thesis chart references found; check --thesis-dir")
    return names


def main():
    global THESIS_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=ROOT)
    parser.add_argument("--thesis-dir", type=Path, default=THESIS_ROOT)
    parser.add_argument("--all", action="store_true", help="Update all data charts referenced by the thesis")
    parser.add_argument("--export-csv", action="store_true", help="Also save optional EDA-summary CSVs")
    args = parser.parse_args()
    THESIS_ROOT = args.thesis_dir.resolve()
    (THESIS_ROOT / "latex/images").mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.experiments / "code"))

    def load_eda_module(module_name: str, relative_path: str):
        spec = importlib.util.spec_from_file_location(
            module_name, args.experiments / relative_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.META = args.experiments / "metadata"
        if hasattr(module, "ML"):
            module.ML = args.experiments / "data/ml_ready_data"
        if hasattr(module, "PROC"):
            module.PROC = args.experiments / "data/processed_data"
        if hasattr(module, "SHAP_ROOT"):
            module.SHAP_ROOT = args.experiments / "shap"
        module.OUT_CHARTS = ROOT / "thesis_assets/charts"
        module.OUT_TABLES = ROOT / "docs/thesis_results/eda"
        module.OUT_DOCS = ROOT / "docs"
        module.TEX_IMG = THESIS_ROOT / "latex/images"
        module.save_chart = collect_chart
        return module

    eda_modules = [
        load_eda_module("additional_eda", "code/_eda_5_additional_results.py"),
        load_eda_module("shap_temporal_eda", "code/_eda_6_shap_and_temporal.py"),
    ]
    required = required_charts(THESIS_ROOT) if args.all else set()
    FIGURES.clear()
    # Render into staging first. Missing inputs or export failures must
    # fail before any existing thesis PNG is replaced.
    with tempfile.TemporaryDirectory(prefix="bop-thesis-charts-") as temporary:
        for module in eda_modules:
            module.main(export_csv=args.export_csv)
        if args.all:
            render_legacy_data_charts(args.experiments, ROOT / "docs/thesis_results/results_metadata.json")
            missing = required - FIGURES.keys()
            if missing:
                raise RuntimeError(f"Missing chart coverage: {sorted(missing)}")
        publish = sorted(required if args.all else FIGURES)
        destinations = [Path(temporary) / name for name in publish]
        try:
            # Batch export otherwise uses global defaults rather than figure sizes.
            pio.write_images(
                fig=[FIGURES[name] for name in publish], file=destinations, scale=3,
                width=[FIGURES[name].layout.width for name in publish],
                height=[FIGURES[name].layout.height for name in publish],
            )
        except Exception as error:
            raise RuntimeError("Plotly/Chrome export failed; no thesis images were replaced.") from error
        if not all(path.is_file() and path.stat().st_size > 0 for path in destinations):
            raise RuntimeError("Incomplete PNG export; no thesis images were replaced.")
        for image in destinations:
            shutil.copy2(image, THESIS_ROOT / "latex/images" / image.name)
        print(f"Updated {len(publish)} thesis charts; no models trained; CSV summaries {'enabled' if args.export_csv else 'skipped'}.")


if __name__ == "__main__":
    main()
