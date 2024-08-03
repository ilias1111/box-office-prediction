import pandas as pd
import numpy as np
from math import sin, cos, pi
import logging
import duckdb
from datetime import date, timedelta, datetime
import holidays


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

    roi_5 = movie['ratio_adj'].quantile(0.05)
    roi_95 = movie['ratio_adj'].quantile(0.95)
    movie['is_within_scope'] = np.where((movie['budget_usd_adj'] > 25_000) & (movie['revenue_usd_adj']> 25_000), 1, 0).astype(bool)


    if remove_outliers:
        try:
            movie_clean = movie[(movie['is_within_scope']) & (movie['release_category'] == 'Far streaming release')]
            movie_clean = movie_clean[(movie_clean['ratio_adj'] > roi_5) & (movie_clean['ratio_adj'] < roi_95)]
            logging.info(f"{len(movie)- len(movie_clean)} outliers removed successfully.")
            return movie_clean
        except Exception as e:
            logging.error(f"Error removing outliers: {e}", exc_info=True)
            return movie
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

def check_holiday_window(date, days=10, holidays=holidays.US()):
    
    for i in range(days+1):  # Check for the same day and the next X days
        if (date + timedelta(days=i)) in holidays:
            return True
    return False

def check_specific_holiday_delta(date, specific_holiday):

    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)

    date = date.date()
    year = date.year
    next_year = year + 1

    year_holidays = holidays.US(years=[year, next_year])

    future_holidays = {k: v for k, v in year_holidays.items() if v == specific_holiday and k >= date}
    
    if future_holidays:
        next_holiday_date = min(future_holidays.keys())
        return (next_holiday_date - date).days
    else:
        return None

def remove_columns(df):
    """
    Filter DataFrame to keep only the specified columns.
    """

    necessary_columns = [
        'movie_id', 'original_language', 'release_date','runtime', 'ageCert', 'quarter', 'month', 'year',
        'is_released_US', 'is_released_CN', 'is_released_FR', 'is_released_GB', 'is_released_JP',
        'budget_usd_adj', 'revenue_usd_adj', 'surplus', 'ratio_adj', 'roi'
    ]
    return df[necessary_columns]

def add_holiday_features(movie_df):

    us_holidays = holidays.US()

    movie_df['release_date'] = pd.to_datetime(movie_df['release_date'])

    # Create a new column 'is_holiday' with a boolean value
    movie_df['is_on_holiday_window'] = movie_df['release_date'].apply(check_holiday_window, days = 15, holidays=us_holidays)
    movie_df['days_to_xmas'] = movie_df['release_date'].apply(check_specific_holiday_delta, specific_holiday='Christmas Day')
    movie_df['days_to_labour_day'] = movie_df['release_date'].apply(check_specific_holiday_delta, specific_holiday='Labor Day')
    movie_df['days_to_thanksgiving'] = movie_df['release_date'].apply(check_specific_holiday_delta, specific_holiday='Thanksgiving')

    return movie_df

def pivot_one_hot(df, column_name, prefix, occurrences = 30):
    # Group by the column and count occurrences of movie_id, reset index to flatten the DataFrame
    df_count = df.groupby(by=column_name)['movie_id'].count().reset_index()
    
    # Filter for keys with more than 10 occurrences
    df_keys = df_count[df_count.movie_id > occurrences][column_name].values
    
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


def add_complex_socioeconomic_features(movie_df, world_bank, oecd):
    
    data_world_bank_q = '''
        SELECT
            movie_df.movie_id,
            world_bank_gdp_growth."OECD members [OED]"::float AS gdp_growth_oecd,
            world_bank_empl_rate."OECD members [OED]"::float AS empl_rate_oecd
        FROM movie_df
        LEFT JOIN world_bank AS world_bank_gdp_growth 
            ON date_part('year', CAST(release_date AS TIMESTAMP)) = world_bank_gdp_growth."Time" 
            AND world_bank_gdp_growth."Series Name" = 'GDP growth (annual %)'
        LEFT JOIN world_bank AS world_bank_empl_rate 
            ON date_part('year', CAST(release_date AS TIMESTAMP)) = world_bank_empl_rate."Time"
            AND world_bank_empl_rate."Series Name" = 'Employment to population ratio, 15+, total (%) (national estimate)'
    '''

    
    oecd_q = '''
        SELECT
            movie_df.movie_id,
            oecd_composite_consumer_confidence.OBS_VALUE AS composite_consumer_confidence_oecd,
            oecd_economic_situation.OBS_VALUE AS economic_situation_oecd,
            oecd_consumer_prices.OBS_VALUE AS consumer_prices_oecd
        FROM movie_df
        LEFT JOIN oecd AS oecd_composite_consumer_confidence
            ON strftime(CAST(release_date AS TIMESTAMP), '%Y-%m') = oecd_composite_consumer_confidence.TIME_PERIOD
            AND oecd_composite_consumer_confidence."Measure_1" = 'Composite consumer confidence'
        LEFT JOIN oecd AS oecd_economic_situation
            ON strftime(CAST(release_date AS TIMESTAMP), '%Y-%m') = oecd_economic_situation.TIME_PERIOD
            AND oecd_economic_situation."Measure_1" = 'Economic situation'
        LEFT JOIN oecd AS oecd_consumer_prices
            ON strftime(CAST(release_date AS TIMESTAMP), '%Y-%m') = oecd_consumer_prices.TIME_PERIOD
            AND oecd_consumer_prices."Measure_1" = 'Consumer prices'
    '''

    data_world_bank = duckdb.sql(data_world_bank_q).to_df()
    data_oecd = duckdb.sql(oecd_q).to_df()

    movie_df = movie_df.merge(data_world_bank, on='movie_id', how='left')
    movie_df = movie_df.merge(data_oecd, on='movie_id', how='left')



    return movie_df



def add_complex_kpi_features(movie_df, production_df, keyword_df, genre_df, collection_df, movie_crew, movie_cast, spoken_languages, production_countries):

    past_production_companies_perfrormance_q = '''
    SELECT
        init_movie_df.movie_id,
        COUNT(post_movie_df.movie_id) as production_company_no_movies,
        AVG(post_movie_df.revenue_usd_adj) as production_company_avg_revenue_usd,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as production_company_25th_percentile_revenue_usd,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as production_company_75th_percentile_revenue_usd
    FROM movie_df AS init_movie_df
    LEFT JOIN production_df AS init_prod ON init_movie_df.movie_id = init_prod.movie_id
    LEFT JOIN production_df AS post_prod ON init_prod.company_id = post_prod.company_id
    LEFT JOIN movie_df AS post_movie_df ON post_prod.movie_id = post_movie_df.movie_id
    WHERE post_movie_df.release_date < init_movie_df.release_date
    GROUP BY init_movie_df.movie_id
    '''

    past_director_perfrormance_q = '''
    SELECT
        init_movie_df.movie_id,
        COUNT(post_movie_df.movie_id) as director_no_movies,
        AVG(post_movie_df.revenue_usd_adj) as director_avg_revenue_usd,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as director_25th_percentile_revenue_usd,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as director_75th_percentile_revenue_usd
    FROM movie_df AS init_movie_df
    LEFT JOIN movie_crew AS init_movie_crew ON init_movie_df.movie_id = init_movie_crew.movie_id
    LEFT JOIN movie_crew AS post_movie_crew ON init_movie_crew.crew_id = post_movie_crew.crew_id
    LEFT JOIN movie_df AS post_movie_df ON post_movie_crew.movie_id = post_movie_df.movie_id
    WHERE post_movie_df.release_date < init_movie_df.release_date
        AND post_movie_crew.job = 'Director' AND init_movie_crew.job = 'Director'
    GROUP BY init_movie_df.movie_id
    '''

    past_writer_perfrormance_q = '''
    SELECT
        init_movie_df.movie_id,
        COUNT(post_movie_df.movie_id) as writer_no_movies,
        AVG(post_movie_df.revenue_usd_adj) as writer_avg_revenue_usd,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as writer_25th_percentile_revenue_usd,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as writer_75th_percentile_revenue_usd
    FROM movie_df AS init_movie_df
    LEFT JOIN movie_crew AS init_movie_crew ON init_movie_df.movie_id = init_movie_crew.movie_id
    LEFT JOIN movie_crew AS post_movie_crew ON init_movie_crew.crew_id = post_movie_crew.crew_id
    LEFT JOIN movie_df AS post_movie_df ON post_movie_crew.movie_id = post_movie_df.movie_id
    WHERE post_movie_df.release_date < init_movie_df.release_date
        AND post_movie_crew.job = 'Writer' AND init_movie_crew.job = 'Writer'
    GROUP BY init_movie_df.movie_id
    '''

    past_ex_prod_perfrormance_q = '''
    SELECT
        init_movie_df.movie_id,
        COUNT(post_movie_df.movie_id) as writer_no_movies,
        AVG(post_movie_df.revenue_usd_adj) as writer_avg_revenue_usd,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as writer_25th_percentile_revenue_usd,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as writer_75th_percentile_revenue_usd
    FROM movie_df AS init_movie_df
    LEFT JOIN movie_crew AS init_movie_crew ON init_movie_df.movie_id = init_movie_crew.movie_id
    LEFT JOIN movie_crew AS post_movie_crew ON init_movie_crew.crew_id = post_movie_crew.crew_id
    LEFT JOIN movie_df AS post_movie_df ON post_movie_crew.movie_id = post_movie_df.movie_id
    WHERE post_movie_df.release_date < init_movie_df.release_date
        AND post_movie_crew.job = 'Executive Producer' AND init_movie_crew.job = 'Executive Producer'
    GROUP BY init_movie_df.movie_id
    '''
    past_composer_perfrormance_q = '''
    SELECT
        init_movie_df.movie_id,
        COUNT(post_movie_df.movie_id) as writer_no_movies,
        AVG(post_movie_df.revenue_usd_adj) as writer_avg_revenue_usd,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as writer_25th_percentile_revenue_usd,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as writer_75th_percentile_revenue_usd
    FROM movie_df AS init_movie_df
    LEFT JOIN movie_crew AS init_movie_crew ON init_movie_df.movie_id = init_movie_crew.movie_id
    LEFT JOIN movie_crew AS post_movie_crew ON init_movie_crew.crew_id = post_movie_crew.crew_id
    LEFT JOIN movie_df AS post_movie_df ON post_movie_crew.movie_id = post_movie_df.movie_id
    WHERE post_movie_df.release_date < init_movie_df.release_date
        AND post_movie_crew.job ilike '%Composer%' AND init_movie_crew.job ilike '%Composer%'
    GROUP BY init_movie_df.movie_id
    '''

    past_keywords_perfrormance_q = '''
    SELECT
        init_movie_df.movie_id,
        COUNT(post_movie_df.movie_id) as keyword_no_movies,
        AVG(post_movie_df.revenue_usd_adj) as keyword_avg_revenue_usd,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as keyword_25th_percentile_revenue_usd,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as keyword_75th_percentile_revenue_usd
    FROM movie_df AS init_movie_df
    LEFT JOIN keyword_df AS init_keyword ON init_movie_df.movie_id = init_keyword.movie_id
    LEFT JOIN keyword_df AS post_keyword ON init_keyword.keyword_id = post_keyword.keyword_id
    LEFT JOIN movie_df AS post_movie_df ON post_keyword.movie_id = post_movie_df.movie_id
    WHERE post_movie_df.release_date < init_movie_df.release_date
    GROUP BY init_movie_df.movie_id
    '''


    past_genres_perfrormance_q = '''
    SELECT
        init_movie_df.movie_id,
        COUNT(post_movie_df.movie_id) as genre_no_movies,
        AVG(post_movie_df.revenue_usd_adj) as genre_avg_revenue_usd,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as genre_25th_percentile_revenue_usd,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as genre_75th_percentile_revenue_usd
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
        AVG(post_movie_df.revenue_usd_adj) as collection_avg_revenue_usd,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as collection_25th_percentile_revenue_usd,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as collection_75th_percentile_revenue_usd,
    FROM movie_df AS init_movie_df
    LEFT JOIN collection_df AS init_collection ON init_movie_df.movie_id = init_collection.movie_id
    LEFT JOIN collection_df AS post_collection ON init_collection.collection_id = post_collection.collection_id
    LEFT JOIN movie_df AS post_movie_df ON post_collection.movie_id = post_movie_df.movie_id
    WHERE post_movie_df.release_date < init_movie_df.release_date
    GROUP BY init_movie_df.movie_id
    '''


    movies_released_within_15_days_q = '''
        SELECT
            init_movie_df.movie_id,
            COUNT(post_movie_df.movie_id) as movies_released_within_15_days
        FROM movie_df AS init_movie_df
        CROSS JOIN movie_df AS post_movie_df
        WHERE CAST(post_movie_df.release_date AS DATE) BETWEEN 
            CAST(init_movie_df.release_date AS DATE) - INTERVAL '7' DAY AND 
            CAST(init_movie_df.release_date AS DATE) + INTERVAL '7' DAY
        GROUP BY init_movie_df.movie_id;
    '''

    # past_actor_perfrormance_q = '''

    # WITH temp AS (
    #     SELECT
    #         init_movie_df.movie_id,
    #         init_movie_cast.actor_id,
    #         init_movie_cast.order,
    #         sum(post_movie_df.revenue_usd_adj * ( 1/(post_movie_cast.order+1) ) / sum( 1/(post_movie_cast.order+1) ) as revenue_weighted_avg
    #     FROM movie_df AS init_movie_df
    #     LEFT JOIN movie_cast AS init_movie_cast ON init_movie_df.movie_id = init_movie_cast.movie_id
    #     LEFT JOIN movie_cast AS post_movie_cast ON init_movie_cast.actor_id = post_movie_cast.actor_id
    #     LEFT JOIN movie_df AS post_movie_df ON post_movie_cast.movie_id = post_movie_df.movie_id
    #     WHERE post_movie_df.release_date < init_movie_df.release_date
    #         AND init_movie_cast.order < 6
    #         AND post_movie_cast.order < 6
    #     GROUP BY init_movie_df.movie_id
    
    # )

    # SELECT
    #     movie_id,
    #     COUNT(revenue_weighted_avg) as actor_no_movies,
    #     AVG(revenue_weighted_avg) as actor_avg_revenue_usd,
    #     PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY revenue_weighted_avg) as actor_25th_percentile_revenue_usd,
    #     PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY revenue_weighted_avg) as actor_75th_percentile_revenue_usd
    # FROM temp
    # GROUP BY movie_id
    # '''
    past_actor_perfrormance_q = '''
        SELECT
            init_movie_df.movie_id,
            COUNT(post_movie_df.movie_id) as actor_no_movies,
            AVG(post_movie_df.revenue_usd_adj) as actor_avg_revenue_usd,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as actor_25th_percentile_revenue_usd,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY post_movie_df.revenue_usd_adj) as actor_75th_percentile_revenue_usd
        FROM movie_df AS init_movie_df
        LEFT JOIN movie_cast AS init_movie_cast ON init_movie_df.movie_id = init_movie_cast.movie_id
        LEFT JOIN movie_cast AS post_movie_cast ON init_movie_cast.actor_id = post_movie_cast.actor_id
        LEFT JOIN movie_df AS post_movie_df ON post_movie_cast.movie_id = post_movie_df.movie_id
        WHERE post_movie_df.release_date < init_movie_df.release_date
            AND init_movie_cast.order < 10
            AND post_movie_cast.order < 10
        GROUP BY init_movie_df.movie_id
    '''

    past_production_companies_perfrormance = duckdb.sql(past_production_companies_perfrormance_q).to_df()
    past_keywords_perfrormance = duckdb.sql(past_keywords_perfrormance_q).to_df()
    past_genres_perfrormance = duckdb.sql(past_genres_perfrormance_q).to_df()
    past_collection_perfrormance = duckdb.sql(past_collection_perfrormance_q).to_df()
    past_director_perfrormance = duckdb.sql(past_director_perfrormance_q).to_df()
    past_writer_perfrormance = duckdb.sql(past_writer_perfrormance_q).to_df()
    past_actor_perfrormance = duckdb.sql(past_actor_perfrormance_q).to_df()
    past_ex_prod_perfrormance = duckdb.sql(past_ex_prod_perfrormance_q).to_df()
    past_composer_perfrormance = duckdb.sql(past_composer_perfrormance_q).to_df()
    movies_released_within_15_days = duckdb.sql(movies_released_within_15_days_q).to_df()

    movie_df = movie_df.merge(past_production_companies_perfrormance, on='movie_id', how='left')
    movie_df = movie_df.merge(past_keywords_perfrormance, on='movie_id', how='left')
    movie_df = movie_df.merge(past_genres_perfrormance, on='movie_id', how='left')
    movie_df = movie_df.merge(past_collection_perfrormance, on='movie_id', how='left')
    movie_df = movie_df.merge(past_director_perfrormance, on='movie_id', how='left')
    movie_df = movie_df.merge(past_writer_perfrormance, on='movie_id', how='left')
    movie_df = movie_df.merge(past_ex_prod_perfrormance, on='movie_id', how='left')
    movie_df = movie_df.merge(past_composer_perfrormance, on='movie_id', how='left')


    movie_df = movie_df.merge(past_actor_perfrormance, on='movie_id', how='left')
    movie_df = movie_df.merge(movies_released_within_15_days, on='movie_id', how='left')

    genre_one_hot = pivot_one_hot(genre_df, 'genre_name', 'genre')
    production_one_hot = pivot_one_hot(production_df, 'company_name', 'prod_company')
    keyword_one_hot = pivot_one_hot(keyword_df, 'keyword_name', 'keyword', 60)
    collection_one_hot = pivot_one_hot(collection_df, 'collection_name', 'collection', 5)
    spoken_languages_one_hot = pivot_one_hot(spoken_languages, 'language', 'spoken_language', 30)
    production_countries_one_hot = pivot_one_hot(production_countries, 'country_code', 'prod_country', 30)

    movie_df = movie_df.merge(genre_one_hot, on='movie_id', how='left')
    movie_df = movie_df.merge(production_one_hot, on='movie_id', how='left')
    movie_df = movie_df.merge(keyword_one_hot, on='movie_id', how='left')
    movie_df = movie_df.merge(collection_one_hot, on='movie_id', how='left')
    movie_df = movie_df.merge(spoken_languages_one_hot, on='movie_id', how='left')
    movie_df = movie_df.merge(production_countries_one_hot, on='movie_id', how='left')
    
    return movie_df

def add_features(feature_flag, movie, production, keyword, genre, collection, movie_crew, movie_cast, spoken_languages, production_countries, world_bank, oecd):

    if feature_flag == 'complex':
        movie = add_complex_kpi_features(movie, production, keyword, genre, collection, movie_crew, movie_cast, spoken_languages, production_countries)
        movie = add_simple_features(movie)
        movie = add_complex_socioeconomic_features(movie, world_bank, oecd)
        movie = add_holiday_features(movie)
    elif feature_flag == 'simple':
        movie = add_simple_features(movie)
    elif feature_flag == 'none':
        pass

    return movie

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
