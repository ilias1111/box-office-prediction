import pandas as pd
import numpy as np
from math import sin, cos, pi
import logging
import duckdb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Classification thresholds
CLASSIFICATION_THRESHOLDS = {
    'Bankrupt': (None, -0.5),
    'Flop': (-0.5, -0.1),
    'Small Success': (-0.1, 0.1),
    'Huge Success': (0.1, None)
}

def remove_outliers(movie,remove_outliers):
    """
    Remove outliers from a DataFrame based on the Z-score of the 'ratio_adj' column.
    """
    movie['is_first_released_in_cinemas'] = np.where(movie['days_from_us_release'] > 0, 1, 0).astype(bool)
    movie['is_first_released_in_cinemas_safe'] = np.where(movie['days_from_us_release'] > 60, 1, 0).astype(bool)
    movie['is_released'] = movie['is_released'].astype(bool)

    conditions = [
    (movie['is_released']) & (~movie['is_first_released_in_cinemas']),
    (movie['is_released']) & (movie['is_first_released_in_cinemas_safe']),
    (movie['is_released']) & (movie['is_first_released_in_cinemas']) & (~movie['is_first_released_in_cinemas_safe']),
    ~movie['is_released']
    ]

    # Define the corresponding categories for each condition
    categories = [
    'Streaming release',
    'Far streaming release',
    'Close streaming release',
    'Not released in major markets'
    ]

    # Create the new categorical column
    movie['release_category'] = np.select(conditions, categories, default='Other')
    movie['is_within_scope'] = np.where((movie['budget_usd_adj'] > 10000) & (movie['revenue_usd_adj']>10000) , 1, 0).astype(bool)

    if remove_outliers:
        try:
            movie_clean = movie[(movie['is_within_scope']) & (movie['release_category'] == 'Far streaming release')]
            logging.info(f"{len(movie)- len(movie_clean)} outliers removed successfully.")
            return movie_clean
        except Exception as e:
            logging.error(f"Error removing outliers: {e}", exc_info=True)
            return movie  # Return the original DataFrame in case of error
    else:
        return movie

def sin_cos(n, k):
    """
    Calculate the sine and cosine of an angle derived from dividing the circle based on k partitions.
    """
    try:
        theta = (2 * pi * n) / k
        return sin(theta), cos(theta)
    except Exception as e:
        logging.error(f"Error in sin_cos calculation: {e}", exc_info=True)
        return np.nan, np.nan  # Return NaNs if calculation fails

def classify_fixed_buckets(roi):
    """
    Classify the surplus based on predefined budget ranges in CLASSIFICATION_THRESHOLDS.
    """
    for label, (low, high) in CLASSIFICATION_THRESHOLDS.items():
        if low is None and roi < high:
            return label
        elif high is None and roi >= low:
            return label
        elif low is not None and high is not None and low <= roi < high:
            return label
    logging.warning(f"Surplus value {roi} did not match any classification.")
    return 'Unknown'  # Default return if no conditions are met

def classify_binary(surplus):
    """
    Classify surplus as either 'Bankrupt' if negative or 'Success' if positive or zero.
    """
    return 'Bankrupt' if surplus < 0 else 'Success'


def remove_columns(df):
    """
    Filter DataFrame to keep only the specified columns.
    """

    necessary_columns = [
        'movie_id', 'original_language', 'release_date','runtime', 'ageCert', 'is_sequel_my', 'quarter', 'month', 'year',
        'is_released_US', 'is_released_CN', 'is_released_FR', 'is_released_GB', 'is_released_JP',
        'budget_usd_adj', 'revenue_usd_adj', 'surplus', 'ratio_adj', 'roi'
    ]
    return df[necessary_columns]


import pandas as pd

def pivot_one_hot(df, column_name, prefix):
    # Group by the column and count occurrences of movie_id, reset index to flatten the DataFrame
    df_count = df.groupby(by=column_name)['movie_id'].count().reset_index()
    
    # Filter for keys with more than 10 occurrences
    df_keys = df_count[df_count.movie_id > 10][column_name].values
    
    # Create a filtered DataFrame ensuring it's a copy with .copy()
    df_filtered = df[df[column_name].isin(df_keys)].copy()
    
    # Add a column 'v' set to True for all entries
    df_filtered['value'] = True
    
    # Create a new column 'act_column' by applying a format to 'column_name'
    df_filtered['act_column'] = df_filtered[column_name].apply(lambda x: f'{prefix}__{str(x)}')
    
    # Create a pivot table indexed by 'movie_id', with columns defined by 'act_column' and values set to 'value' and i want the value to be a boolean
    df_one_hot = pd.pivot_table(df_filtered, index="movie_id", columns="act_column", values='value', fill_value=False, aggfunc=any)

    return df_one_hot




def add_simple_features(movie_df):

    movie_df['month_sin'] = movie_df['month'].apply(lambda x: sin_cos(x, 12)[0])
    movie_df['month_cos'] = movie_df['month'].apply(lambda x: sin_cos(x, 12)[1])

    return movie_df




def add_complex_kpi_features(movie_df, production_df, keyword_df, genre_df, collection_df):

    past_production_companies_perfrormance_q = '''
    SELECT
        init_movie_df.movie_id,
        COUNT(post_movie_df.movie_id) as production_company_no_movies,
        AVG(post_movie_df.revenue_usd_adj) as production_company_avg_revenue,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as production_company_25th_percentile_revenue,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as production_company_75th_percentile_revenue
    FROM movie_df AS init_movie_df
    LEFT JOIN production_df AS init_prod ON init_movie_df.movie_id = init_prod.movie_id
    LEFT JOIN production_df AS post_prod ON init_prod.parent_id = post_prod.parent_id
    LEFT JOIN movie_df AS post_movie_df ON post_prod.movie_id = post_movie_df.movie_id
    WHERE post_movie_df.release_date < init_movie_df.release_date
    GROUP BY init_movie_df.movie_id
    '''

    past_keywords_perfrormance_q = '''
    SELECT
        init_movie_df.movie_id,
        COUNT(post_movie_df.movie_id) as keyword_no_movies,
        AVG(post_movie_df.revenue_usd_adj) as keyword_avg_revenue,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as keyword_25th_percentile_revenue,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as keyword_75th_percentile_revenue
    FROM movie_df AS init_movie_df
    LEFT JOIN keyword_df AS init_keyword ON init_movie_df.movie_id = init_keyword.movie_id
    LEFT JOIN keyword_df AS post_keyword ON init_keyword.keyword_id = post_keyword.keyword_id
    LEFT JOIN movie_df AS post_movie_df ON post_keyword.movie_id = post_movie_df.movie_id
    WHERE post_movie_df.release_date::date < init_movie_df.release_date::date
    GROUP BY init_movie_df.movie_id
    '''


    past_genres_perfrormance_q = '''
    SELECT
        init_movie_df.movie_id,
        COUNT(post_movie_df.movie_id) as genre_no_movies,
        AVG(post_movie_df.revenue_usd_adj) as genre_avg_revenue,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as genre_25th_percentile_revenue,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as genre_75th_percentile_revenue
    FROM movie_df AS init_movie_df
    LEFT JOIN genre_df AS init_genre ON init_movie_df.movie_id = init_genre.movie_id
    LEFT JOIN genre_df AS post_genre ON init_genre.genre_id = post_genre.genre_id
    LEFT JOIN movie_df AS post_movie_df ON post_genre.movie_id = post_movie_df.movie_id
    WHERE post_movie_df.release_date < init_movie_df.release_date
    GROUP BY init_movie_df.movie_id
    '''

    past_collection_perfrormance_q = '''
    SELECT
        init_movie_df.movie_id,
        COUNT(post_movie_df.movie_id) as collection_no_movies,
        AVG(post_movie_df.revenue_usd_adj) as collection_avg_revenue,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as collection_25th_percentile_revenue,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as collection_75th_percentile_revenue
    FROM movie_df AS init_movie_df
    LEFT JOIN collection_df AS init_collection ON init_movie_df.movie_id = init_collection.movie_id
    LEFT JOIN collection_df AS post_collection ON init_collection.collection_id = post_collection.collection_id
    LEFT JOIN movie_df AS post_movie_df ON post_collection.movie_id = post_movie_df.movie_id
    WHERE post_movie_df.release_date < init_movie_df.release_date
    GROUP BY init_movie_df.movie_id
    '''

    past_production_companies_perfrormance = duckdb.sql(past_production_companies_perfrormance_q).to_df()
    past_keywords_perfrormance = duckdb.sql(past_keywords_perfrormance_q).to_df()
    past_genres_perfrormance = duckdb.sql(past_genres_perfrormance_q).to_df()
    past_collection_perfrormance = duckdb.sql(past_collection_perfrormance_q).to_df()

    movie_df = movie_df.merge(past_production_companies_perfrormance, on='movie_id', how='left')
    movie_df = movie_df.merge(past_keywords_perfrormance, on='movie_id', how='left')
    movie_df = movie_df.merge(past_genres_perfrormance, on='movie_id', how='left')
    movie_df = movie_df.merge(past_collection_perfrormance, on='movie_id', how='left')

    genre_one_hot = pivot_one_hot(genre_df, 'name', 'genre')
    production_one_hot = pivot_one_hot(production_df, 'parent_name', 'prod_company')
    keyword_one_hot = pivot_one_hot(keyword_df, 'keyword_name', 'keyword')
    collection_one_hot = pivot_one_hot(collection_df, 'collection_name', 'collection')

    movie_df = movie_df.merge(genre_one_hot, on='movie_id', how='left')
    movie_df = movie_df.merge(production_one_hot, on='movie_id', how='left')
    movie_df = movie_df.merge(keyword_one_hot, on='movie_id', how='left')
    movie_df = movie_df.merge(collection_one_hot, on='movie_id', how='left')
    
    return movie_df

def add_features(feature_flag, movie_df, production_df, keyword_df, genre_df, collection_df):

    if feature_flag == 'complex':
        movie_df = add_complex_kpi_features(movie_df, production_df, keyword_df, genre_df, collection_df)
        movie_df = add_simple_features(movie_df)
    elif feature_flag == 'simple':
        movie_df = add_simple_features(movie_df)
    elif feature_flag == 'none':
        pass

    return movie_df

def add_target_variable(movie_df, task_type):

    # Classifying data
    if task_type == 'binary_classification':
        movie_df['binary_classification'] = movie_df['surplus'].apply(classify_binary)
        movie_df.drop(columns=['revenue_usd_adj'], inplace=True)
    elif task_type == 'multi_class_classification':
        movie_df['multi_class_classification'] = movie_df['roi'].apply(classify_fixed_buckets)
        movie_df.drop(columns=['revenue_usd_adj'], inplace=True)
    movie_df.drop(columns=['ratio_adj','roi','surplus'], inplace=True)

    return movie_df

# Example usage (if needed for debugging or operational script)
if __name__ == "__main__":
    # Example DataFrame creation (to demonstrate functionality)
    df = pd.DataFrame({
        'ratio_adj': np.random.normal(loc=0, scale=1, size=100)
    })

    df_clean = remove_outliers(df)
    month_sin, month_cos = sin_cos(1, 12)  # For January in a yearly cycle

    print("Sin and Cos values:", month_sin, month_cos)
    print("DataFrame after outlier removal:", df_clean.head())
