# Box Office Prediction

This project predicts box office performance from information available before a film's release. The pipeline retrieves data, preprocesses it, builds features, and trains the models reported in the thesis.

Run every command below from the repository root.

## Project Structure

1. Data Retrieval — `code/1_data_retrieval/`
   - TMDb daily exports, IMDb id matching, and movie details
   - Wikipedia financial pages and currency conversion
2. Data Preprocessing — `code/2_data_preprocessing/`
   - Cleaning, merging, and the first derived fields
3. Feature Engineering — `code/3_feature_eng/`
   - Participant history, release timing, holidays, and socioeconomic indicators
   - Writes one CSV per budget segment, task, outlier setting, and feature setting (`none` or `complex`)
4. Machine Learning — `code/4_machine_learning/`
   - Training, random search, grid search, and evaluation
5. Infrastructure — `infra/`
   - `infra/compute_instance/`: Terraform for the training VM
   - `infra/mojo_scrapping/`: Terraform for the Box Office Mojo scraper

## Setup

1. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set the API tokens used by retrieval:
   ```
   export TMDB_API_TOKEN=your_tmdb_token_here
   export ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
   ```

Retrieved TMDb and Wikipedia files, plus the manual World Bank, OECD, Wikidata, and Box Office Mojo tables, are already in `data/`. Preprocessed and model-ready files are produced locally and are not in the repository.

## Usage

1. **IDs**
   ```
   python code/1_data_retrieval/tmdb_daily_exports_retrieval.py
   python code/1_data_retrieval/imdb_id_retrieval.py
   ```

2. **TMDb details**
   ```
   python code/1_data_retrieval/tmdb_retrieval.py
   ```

3. **Financial data**
   ```
   python code/1_data_retrieval/wikipedia_retrieval.py
   python code/1_data_retrieval/currency_converter.py
   ```

4. **Preprocessing** — writes `data/processed_data/`
   ```
   python code/2_data_preprocessing/main_preprocessing.py
   ```

5. **Feature engineering** — writes `data/ml_ready_data/`
   ```
   python code/3_feature_eng/ml_dataset_creation.py
   ```

6. **Training**
   ```
   python code/4_machine_learning/ml_logic.py
   ```
   Optional environment switches (defaults preserve the historical random-split protocol):

   | Variable | Default | Purpose |
   | --- | --- | --- |
   | `MOTR_SPLIT_STRATEGY` | `random` | `random` or `temporal` outer holdout / inner CV |
   | `MOTR_COMPUTE_SHAP` | `0` | Save mean-\|SHAP\| under `shap/` when `1` |
   | `MOTR_ONLY_FILE` | unset | Train a single ML-ready CSV |
   | `MOTR_MODELS` | unset | Comma-separated model filter |
   | `MOTR_TEST_START_YEAR` | unset | Optional temporal test-year cutoff |
   | `MOTR_FAST` | `1` | Fast mode (cache / sparse / skip model pickles) |
   | `MOTR_SEARCH_N_JOBS` | `1` | Parallel trials inside one model search |
   | `MOTR_MODEL_N_JOBS` | `4` | Threads used by one model |

   Example: random split with SHAP (main explanation protocol):

   ```sh
   MOTR_SPLIT_STRATEGY=random MOTR_COMPUTE_SHAP=1 \
   MOTR_ONLY_FILE=full__binary_classification__no_outliers__complex.csv \
   MOTR_MODELS=xgboost_classifier \
   .venv/bin/python code/4_machine_learning/ml_logic.py
   ```

7. **Replay a saved configuration**
   ```sh
   # Temporal sensitivity: fixed earlier non-grid configurations, metrics only
   .venv/bin/python code/4_machine_learning/replay_from_metadata.py \
     --run-id 20260207_190503 \
     --split-strategy temporal

   # Random protocol + SHAP (next to impurity FI). Refused with temporal.
   .venv/bin/python code/4_machine_learning/replay_from_metadata.py \
     --metadata metadata/full__binary_classification__no_outliers__complex__xgboost_classifier__20260207_190503.json \
     --split-strategy random \
     --compute-shap
   ```
   Pass `--metadata` for one file (repeatable) or `--run-id` for all
   `metadata/*__{run_id}.json`. Reuses frozen estimator + scaler from JSON;
   writes new run metadata, optional `shap/` artifacts, and a slim CSV summary.

## Data

- The Movie Database (TMDb)
- Wikipedia financial infoboxes
- Box Office Mojo, scraped with the Terraform stack in `infra/mojo_scrapping/`
- Wikidata, World Bank, and OECD

| Path | Contents |
| --- | --- |
| `data/retrieved_data/` | TMDb and Wikipedia extracts |
| `data/manual_data/` | Wikidata, Mojo, currency, World Bank, and OECD tables |
| `data/processed_data/` | Output of preprocessing |
| `data/ml_ready_data/` | Modeling CSVs |
| `logs/`, `metadata/` | Training logs and per-model results |
| `shap/` | Optional mean-\|SHAP\| CSVs / samples from explain runs |

## Models

MOTR trains these models for regression, binary classification, and multi-class classification:

- Dummy baseline
- Logistic regression (classification)
- Decision tree
- Random forest
- XGBoost
- LightGBM

## Evaluation

Regression reports MSE, RMSE, MAE, MAPE, and R2, plus threshold accuracy. Binary classification reports accuracy, positive-class precision/recall/F1, and ROC AUC. Multi-class classification reports accuracy and support-weighted precision/recall/F1, with per-class reports and confusion matrices; multi-class ROC AUC is not reported. Results are written under `logs/` and `metadata/`.

Historical unfiltered results are provisional: an audit found the revenue-derived
`is_outlier` flag among model inputs. The loader now excludes `is_outlier` and
`is_within_scope` from predictors while retaining audit flags in dataset CSVs.
This code correction does not repair existing models or metrics; affected
experiments require re-evaluation. Hyperparameter search optimizes log-space
R² for regression (the model is trained on `log10(revenue)`), ROC AUC for
binary classification, and weighted F1 for multi-class classification. Dollar
R², MAE, and the within-1.5× rate are still reported. The held-out regression
back-transform clips log predictions without flipping their sign. New runs
record the search score, its cross-validation value, the dataset SHA-256, the
feature columns, and the git revision.

## Thesis exports (optional)

The thesis repository only needs its LaTeX build tools. Analysis and result
generation stay here. To update its result tables and all data-derived charts
referenced by its chapters, run this from the experiment repository:

```sh
.venv/bin/python code/_export_thesis_tables.py --charts
```

This writes the 23 LaTeX result tables and replaces all 51 data-derived figures
currently referenced by the thesis: retrieval, preprocessing, feature-engineering,
model-comparison, confusion-matrix and additional EDA charts. It reuses the
existing datasets, chart definitions and selected historical experiment results
in `docs/thesis_results/results_metadata.json`. It does not retrieve data or
train models. Static illustrations and diagrams stay unchanged.

Figures are rendered into staging and checked against the LaTeX references
before replacing any existing PNGs. PNG export uses native Plotly/Kaleido with
the installed Google Chrome (no custom Matplotlib renderer). Chrome is already
available on this machine; chart export does not open a visible browser window.
Table export without `--charts` still updates
only tables. Future experiment runs require deliberately updating the selected
result snapshot; they are not silently selected as the newest results.

The six optional EDA-summary CSVs are skipped by default and were removed from
`docs/thesis_results/eda/`. If needed for review, they can be requested with
`code/_export_thesis_charts.py --export-csv`. They are never chart inputs.

`code/_eda_6_shap_and_temporal.py` (also wired into `_export_thesis_charts.py`)
follows the thesis story: SHAP charts for the **random** protocol next to
impurity FI, plus **temporal metric sensitivity** charts from
`docs/thesis_results/temporal_vs_random_compare.csv` (no temporal SHAP):

```sh
.venv/bin/python code/_eda_6_shap_and_temporal.py
```

The supplementary figures use nine earlier `non_grid` configurations, refitted
under paired random and chronological holdouts. They are representative models,
not the search winners from the main tables. Their source configurations, fitted
parameters, data hashes, split boundaries, package versions, and complete aggregate
SHAP summaries are pinned in `docs/thesis_results/supplementary_results.json`.
This tracked input lets SHAP figures regenerate without the ignored `shap/` run
folder. The two XGBoost summaries use 400 held-out films each (seed 42); their
sample movie IDs are recorded. Binary SHAP is in raw log-odds and regression SHAP
in log10 adjusted-revenue units. Temporal results remain conditional on the
retrospective revenue-based cohort filter. The historical main tables are unchanged.

After exporting results, run `./scripts/build-thesis.sh` in the thesis repository
to refresh the PDF. Building the PDF does not run analysis automatically.
