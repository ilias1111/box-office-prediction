import pandas as pd
import feature_eng
import os

# Constants for paths
DATA_PATH = './data/processed_data'
MANUAL_DATA_PATH = './data/manual_data'
ML_READY_DATA_PATH = './data/ml_ready_data'

def get_file_path(filename, path=DATA_PATH):
    """Construct the full path for a data file based on filename."""
    return f'{path}/{filename}.csv'

def load_files_into_dfs():
    """
    Load multiple CSV files into pandas DataFrames.
    Returns a tuple of DataFrames.
    """
    filenames = ['movie', 'keyword', 'production', 'collection', 'genre', 'movie_crew', 'movie_cast', 'spoken_languages', 'production_countries']
    return tuple(pd.read_csv(get_file_path(name)) for name in filenames)

def load_socioeconomic_data():

    world_bank = pd.read_csv(get_file_path('WORLD_BANK', MANUAL_DATA_PATH))
    oecd = pd.read_csv(get_file_path('OECD', MANUAL_DATA_PATH))

    return world_bank, oecd

def generate_file_name(production_size, task_type, remove_outliers, product_flag):
    """
    Generate a dynamic file name based on dataset configuration.
    """
    outlier_status = 'with_outliers' if not remove_outliers else 'no_outliers'
    return f"{production_size}__{task_type}__{outlier_status}__{product_flag}.csv"


def save_data(movie_df, task_type, remove_outliers, feature_flag):
    """
    Save subsets of data based on budget categories to separate CSV files.
    """

    # budget_categories = {
    #     'small_productions': (0, 3_000_000),          # Small budget
    #     'medium_productions': (3_000_000, 40_000_000), # Medium budget
    #     'large_productions': (40_000_000, 999_999_999), # Large budget
    #     'full' : (0, 999_999_999)          # Full range
    # }

    budget_categories = {
        'small_productions': (0, 5_000_000),          # Small budget
        'medium_productions': (5_000_001, 50_000_000), # Medium budget
        'large_productions': (50_000_001, 999_999_999), # Large budget
        'full' : (0, 999_999_999)          # Full range
    }

    os.makedirs(ML_READY_DATA_PATH, exist_ok=True)

    for production_size, (lower_bound, upper_bound) in budget_categories.items():
        subset = movie_df[movie_df.budget_usd_adj.between(lower_bound, upper_bound)]
        file_name = generate_file_name(production_size, task_type, remove_outliers, feature_flag)
        full_path = f'{ML_READY_DATA_PATH}/{file_name}'
        subset.to_csv(full_path, index=False)
        print(f"Data saved to {full_path}")


def construct_dataset(feature_flag, task_type, to_remove_outliers=False):

    movie, keyword, production, collection, genre, movie_crew, movie_cast, spoken_languages, production_countries  = load_files_into_dfs()

    world_bank, oecd = load_socioeconomic_data()

    # Filter necessary columns in movie DataFrame
    
    movie = feature_eng.remove_outliers(movie, to_remove_outliers)

    movie = feature_eng.remove_columns(movie)

    movie = feature_eng.add_features(feature_flag, movie, production, keyword, genre, collection, movie_crew, movie_cast, spoken_languages, production_countries, world_bank, oecd)
    
    movie = feature_eng.add_target_variable(movie, task_type)

    movie.drop(columns=['release_date'], inplace=True)

    save_data(movie, task_type, to_remove_outliers, feature_flag)

if __name__ == "__main__":

    TASK_TYPE = ['binary_classification', 'multi_class_classification', 'regression']
    REMOVE_OUTLIERS = [True, False]
    FEATURE_FLAG = ['complex'
                    ,'none'
                    ]

    for task in TASK_TYPE:
        for outlier_status in REMOVE_OUTLIERS:
            for feature_flag in FEATURE_FLAG:

                construct_dataset(feature_flag, task, outlier_status)