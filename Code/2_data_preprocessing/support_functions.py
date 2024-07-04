import pandas as pd
import logging
from typing import Dict, Tuple, Optional
import re
from unidecode import unidecode
import cpi
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

CERTIFICATE_MAPPINGS = {
    "0": "G",
    "6": "G",
    "G": "G",
    "R": "R",
    "12": "PG13",
    "16": "PG16",
    "18": "R",
    "PG": "PG",
    "NC-17": "R",
    "PG-13": "PG13",
    "TV-14": "PG16",
    "Unrated": "U",
    "(Banned)": "R",
    "Not Rated": "U",
    "BPjM Restricted": "R"
}


# Functions
def fix_certificates(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['ageCert'] = df['ageCert'].replace(CERTIFICATE_MAPPINGS).fillna("U")
    except Exception as e:
        logging.error(
            f'An error occurred while fixing certificates: {e}', exc_info=True)
    return df


def adjust_cpi(amount: float, date: int) -> float:
    if date < 2021:
        try:
            return cpi.inflate(amount, date)
        except Exception as e:
            logging.warning(
                f'CPI adjustment failed for {amount} on {date}: {e}')
            return -1
    else:
        return amount


def init_data_loading():

    movie = pd.read_csv(
        '/Users/iliasx/Documents/GitHub/box-office-prediction/data/retrieved_data/movie.csv')
    genre_mapping = pd.read_csv(
        '/Users/iliasx/Documents/GitHub/box-office-prediction/data/retrieved_data/genre_mapping.csv')
    keyword_mapping = pd.read_csv(
        '/Users/iliasx/Documents/GitHub/box-office-prediction/data/retrieved_data/keyword_mapping.csv')
    cast_mapping = pd.read_csv(
        '/Users/iliasx/Documents/GitHub/box-office-prediction/data/retrieved_data/cast_mapping.csv')
    collection_mapping = pd.read_csv(
        '/Users/iliasx/Documents/GitHub/box-office-prediction/data/retrieved_data/collection_mapping.csv')
    production_mapping = pd.read_csv(
        '/Users/iliasx/Documents/GitHub/box-office-prediction/data/retrieved_data/production_mapping.csv')

    keyword = pd.read_csv(
        '/Users/iliasx/Documents/GitHub/box-office-prediction/data/retrieved_data/keyword.csv')
    production = pd.read_csv(
        '/Users/iliasx/Documents/GitHub/box-office-prediction/data/retrieved_data/production.csv')
    collection = pd.read_csv(
        '/Users/iliasx/Documents/GitHub/box-office-prediction/data/retrieved_data/collection.csv')

    movies_with_sequel = pd.read_csv(
        '/Users/iliasx/Documents/GitHub/box-office-prediction/data/retrieved_data/movies_with_sequel.csv')
    
    movie_release_dates = pd.read_csv(
        '/Users/iliasx/Documents/GitHub/box-office-prediction/data/processed_data/movie_release_dates.csv')

    return movie, genre_mapping, keyword_mapping, cast_mapping, collection_mapping, production_mapping, \
        keyword, production, collection, movies_with_sequel, movie_release_dates


def movie_table_processing(df, movies_with_sequel_values):

    df['is_sequel_my'] = df['movie_id'].apply(
        lambda x: True if x in movies_with_sequel_values else False)
    df['release_date'] = pd.to_datetime(df['release_date'])
    # df['release_date'] = df['release_date'].tz_localize(None).astype('datetime64[ns]')

    df['quarter'] = df['release_date'].dt.quarter
    df['month'] = df['release_date'].dt.month.astype(np.uint8)
    df['year'] = df['release_date'].dt.year

    df['revenue_usd_adj'] = df.apply(
        lambda x: adjust_cpi(x['revenue_world'], x['year']), axis=1)
    df['budget_usd_adj'] = df.apply(lambda x: adjust_cpi(
        x['budget_usd_adj'], x['year']), axis=1)
    df['surplus'] = 0.5 * df['revenue_usd_adj'] - df['budget_usd_adj']
    df['ratio_adj'] = df['revenue_usd_adj'] / df['budget_usd_adj']
    df['roi'] = df['surplus'] / df['budget_usd_adj']

    df['ageCert'] = df['ageCert'].replace(CERTIFICATE_MAPPINGS).fillna("U")

    return df

def transform_company_hierarchy(df):
    """
    Transforms a DataFrame with company_id, company_name, and parent_id columns
    into a new format indicating level 1 and level 2 company relationships.

    Parameters:
    df (DataFrame): The original DataFrame with columns company_id, company_name, and parent_id.

    Returns:
    DataFrame: Transformed DataFrame with the structure:
               company_id_level_1, company_name_level_1, company_id_level_2, company_name_level_2, is_parent
    """
    transformed_data = []
    for index, row in df.iterrows():
        is_parent = row['company_id'] == row['parent_id']
        parent_rows = df[df['company_id'] == row['parent_id']]
        
        # Check if parent_rows is not empty
        if not parent_rows.empty:
            parent_name = parent_rows['company_name'].values[0]
        else:
            # Handle cases where parent_id doesn't match any company_id
            parent_name = None

        transformed_data.append({
            "parent_id": row['parent_id'],
            "parent_name": parent_name,
            "company_id": row['company_id'],
            "company_name": row['company_name'],
            "is_parent": is_parent
        })

    return pd.DataFrame(transformed_data)


def calculate_release_info_with_checks(df):

    # Ensure 'release_date' is in datetime format
    df['release_date'] = pd.to_datetime(df['release_date'])

    # Filter for theatrical releases and physical/digital releases
    countries = ['US', 'CN', 'FR', 'GB', 'JP']
    theatrical_releases = df[df['type'].isin([2, 3]) & df['country_code'].isin(countries)]
    physical_digital_releases = df[df['type'].isin([4, 5]) & df['country_code'].isin(countries)]

    # Get the earliest theatrical release date for each movie in each country
    theatrical_release_dates = theatrical_releases.groupby(['movie_id', 'country_code'])['release_date'].min().unstack()
    min_theatrical_release_dates = theatrical_releases.groupby('movie_id')['release_date'].min()

    # Get the earliest physical/digital release date for each movie
    min_physical_digital_dates = physical_digital_releases.groupby('movie_id')['release_date'].min()

    # Prepare the final DataFrame
    final_df = pd.DataFrame(index=df['movie_id'].unique())
    final_df = final_df.join(theatrical_release_dates[countries], how='left')
    final_df = final_df.join(min_physical_digital_dates.rename('digital_physical_date'), how='left')
    final_df = final_df.join(min_theatrical_release_dates.rename('min_theatrical_release_date'), how='left')

    # Add flags for release in each country and overall release
    for country in countries:
        final_df[f'is_released_{country}'] = final_df[country].notna()
    final_df['is_released'] = final_df[countries].notna().any(axis=1)

    # Calculate days from US release to digital/physical release
    final_df['days_from_us_release'] = ((final_df['digital_physical_date'] - final_df['US']).dt.days).fillna(9999)

    # Selecting the required columns
    final_df = final_df[['is_released_US', 'is_released_CN', 'is_released_FR', 'is_released_GB', 'is_released_JP', 'is_released', 'digital_physical_date', 'days_from_us_release']]
    final_df.reset_index(inplace=True)
    final_df.rename(columns={'index': 'movie_id'}, inplace=True)


    return final_df


def convert_to_machine_friendly(df, column_name):
    """
    Converts a string column in a pandas DataFrame to a machine-friendly format.
    This involves:
    - Converting to lowercase
    - Transliterating to ASCII (English characters)
    - Replacing special characters with underscores
    - Stripping leading and trailing spaces

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the column.
    column_name (str): The name of the column to convert.

    Returns:
    pandas.DataFrame: The DataFrame with the converted column.
    """

    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    # Transliterate to ASCII
    df[column_name] = df[column_name].apply(lambda x: unidecode(x))

    # Replace special characters with underscores
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\W+', '_', x))

    # Strip leading and trailing spaces
    df[column_name] = df[column_name].str.strip()

    # Convert to lowercase
    df[column_name] = df[column_name].str.lower()

    return df


if __name__ == "__main__":

    data = {'col1': ["Hello World!", "Èspañol test",
                     "Français & Deutsch!", "Русский язык", "日本語", "中文", "العربية", "🌍"]}
    df = pd.DataFrame(data)
    df = convert_to_machine_friendly(df, 'col1')
    print(df)
