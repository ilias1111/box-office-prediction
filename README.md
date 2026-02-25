# Box Office Prediction

This project aims to predict box office performance for movies using machine learning techniques. It involves data retrieval, preprocessing, feature engineering, and machine learning model training and evaluation.

## Project Structure

1. Data Retrieval
   - Located in `code/1_data_retrieval/`
   - Retrieves movie data from The Movie Database (TMDb) API
   - Handles rate limiting and error recovery
   - Stores data in CSV files

2. Data Preprocessing
   - Located in `code/2_data_preprocessing/`
   - Processes raw data into a format suitable for machine learning
   - Includes data cleaning, merging, and initial feature creation

3. Feature Engineering
   - Located in `code/3_feature_eng/`
   - Creates advanced features from the preprocessed data
   - Includes complex KPI features, socioeconomic features, and holiday-related features

4. Machine Learning
   - Located in `code/4_machine_learning/`
   - Implements various machine learning models for box office prediction
   - Includes model training, hyperparameter tuning, and evaluation

5. Infrastructure
   - Located in `infra/`
   - Contains Terraform configurations for compute instances and web scraping

## Key Features

- Asynchronous data retrieval from TMDb API
- Comprehensive feature engineering, including:
  - Production company performance
  - Cast and crew performance metrics
  - Genre and keyword analysis
  - Socioeconomic indicators
  - Release date analysis (including holidays)
- Support for both regression and classification tasks
- Multiple machine learning models, including XGBoost, LightGBM, and neural networks
- Hyperparameter tuning using random search and grid search
- Detailed model evaluation and logging

## Setup and Usage

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up your API tokens as environment variables:
   ```
   export TMDB_API_TOKEN=your_tmdb_token_here
   export ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
   ```

3. **Data Retrieval (Step 1: ID Retrieval)**:
   Run the daily export retrieval to get TMDB IDs and the IMDB ID retrieval script.
   ```
   python code/1_data_retrieval/tmdb_daily_exports_retrieval.py
   python code/1_data_retrieval/imdb_id_retrieval.py
   ```

4. **Data Retrieval (Step 2: TMDB Data)**:
   Retrieve movie details from TMDB.
   ```
   python code/1_data_retrieval/tmdb_retrieval.py
   ```

5. **Data Retrieval (Step 3: Financial Data)**:
   Retrieve financial data from Wikipedia and convert currencies.
   ```
   python code/1_data_retrieval/wikipedia_retrieval.py
   python code/1_data_retrieval/currency_converter.py
   ```

6. **Data Preprocessing**:
   Process the raw data into a clean structure.
   ```
   python code/2_data_preprocessing/main_preprocessing.py
   ```

7. **Feature Engineering & Dataset Creation**:
   Create the machine learning datasets (with/out outliers, simple/complex features).
   ```
   python code/3_feature_eng/ml_dataset_creation.py
   ```

8. **Machine Learning**:
   Run the machine learning models.
   ```
   python code/4_machine_learning/ml_logic.py
   ```

## Data

The project uses data from various sources:
- The Movie Database (TMDb)
- Box Office Mojo (via web scraping)
- World Bank and OECD for socioeconomic indicators

Data is stored in the `data/` directory, with subdirectories for raw, processed, and machine learning-ready data.

## Models

The project supports various machine learning models, including:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Neural Networks

Models can be configured for both regression (predicting box office revenue) and classification (predicting success categories) tasks.

## Evaluation

Model performance is evaluated using various metrics, including:
- For regression: MSE, MAPE, MAE, RMSE, R2
- For classification: Accuracy, Precision, Recall, F1 Score, ROC AUC

Results are logged and stored in the `logs/` and `metadata/` directories.