import pandas as pd
import logging
from typing import Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_and_save_dataframe(df: pd.DataFrame, budget_categories: Dict[str, Tuple[int, Optional[int]]], filename_prefix: str) -> None:
    """
    Splits the dataframe into multiple categories based on the given budget thresholds
    and saves each split and the entire dataframe to CSV files.

    Parameters:
    df (pd.DataFrame): The dataframe to split.
    budget_categories (Dict[str, Tuple[int, Optional[int]]]): Dictionary with labels as keys and budget thresholds as values.
    filename_prefix (str): The prefix for the filenames of the CSV files.
    """
    try:
        for label, (low, high) in budget_categories.items():
            condition = (df.budget_usd_adj > low) & (df.budget_usd_adj <= high) if high is not None else (df.budget_usd_adj > low)
            split_df = df.loc[condition]
            split_filename = f'code/data_pipeline/data/{filename_prefix}_{label}.csv'
            split_df.to_csv(split_filename, index=False)
            logging.info(f'Saved {split_filename} with {len(split_df)} records.')

        # Save the entire dataframe
        df.to_csv(f'code/data_pipeline/data/{filename_prefix}_full.csv', index=False)
        logging.info(f'Saved the entire dataframe to {filename_prefix}_full.csv with {len(df)} records.')
    except Exception as e:
        logging.error(f'An error occurred: {e}', exc_info=True)

def one_hot_encode(data :pd.DataFrame, index_col: str, column_name: str, min_count: int) -> pd.DataFrame:
    """
    Helper function to perform one-hot encoding on a given DataFrame based on a minimum count.
    """
    count = data.groupby(by=column_name)[index_col].count().reset_index()
    filtered_keys = count[count[index_col] > min_count][column_name].values
    filtered_data = data[data[column_name].isin(filtered_keys)]
    filtered_data['v'] = 1
    one_hot = pd.pivot_table(filtered_data, index=index_col, columns=column_name, values='v').fillna(0)
    return one_hot
