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

2. Set up your TMDb API token as an environment variable:
   ```
   export TMDB_API_TOKEN=your_token_here
   ```

3. Run the data retrieval script:
   ```
   python code/1_data_retrieval/tmdb_retrieval.py
   ```

4. Run the preprocessing script:
   ```
   python code/2_data_preprocessing/main_preprocessing.py
   ```

5. Run the feature engineering and machine learning scripts (modify as needed):
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