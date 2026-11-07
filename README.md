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
   Folds, search iterations, and the fast-mode switch are set in the `__main__` block of that file.

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

## Models

MOTR trains these models for regression, binary classification, and multi-class classification:

- Dummy baseline
- Logistic regression (classification)
- Decision tree
- Random forest
- XGBoost
- LightGBM

## Evaluation

Regression reports MSE, RMSE, MAE, MAPE, and R2, plus threshold accuracy. Classification reports accuracy, precision, recall, F1, and ROC AUC. Results are written under `logs/` and `metadata/`.
