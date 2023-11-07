import pandas as pd
import numpy as np
from math import sin, cos, pi
#import cpi
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
CLASSIFICATION_THRESHOLDS = {
    'Bankrupt': (None, 0),
    'Flop': (0, 1_000_000),
    'Small Movie': (1_000_000, 15_000_000),
    'Blockbuster': (15_000_000, 50_000_000),
    'Success': (50_000_000, None)
}

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
        logging.error(f'An error occurred while fixing certificates: {e}', exc_info=True)
    return df

def adjust_cpi(amount: float, date: int) -> float:
    if date < 2021:
        try:
            return cpi.inflate(amount, date)
        except Exception as e:
            logging.warning(f'CPI adjustment failed for {amount} on {date}: {e}')
            return -1
    else:
        return amount

def sin_cos(n: int, k: int) -> tuple:
    try:
        theta = (2 * pi * n) / k
        return sin(theta), cos(theta)
    except Exception as e:
        logging.error(f'An error occurred while calculating sin and cos: {e}', exc_info=True)
        return np.nan, np.nan



def classify_fixed_buckets(s: float) -> str:
    """
    Classifies a given value based on the classification thresholds.
    """
    for label, (low, high) in CLASSIFICATION_THRESHOLDS.items():
        if low is None and s < high:
            return label
        elif high is None and s >= low:
            return label
        elif low <= s < high:
            return label
    return 'Unknown'  # If none of the conditions are met


def classify_binary(s: float) -> str:
    return 'Bankrupt' if s < 0 else "Success"


def merge_data(df: pd.DataFrame, metrics: pd.DataFrame):

    """
    Merges the given metrics dataframe with the given dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe to merge with.
    metrics (pd.DataFrame): The metrics dataframe to merge.

    Returns:
    df (pd.DataFrame): The merged dataframe.
    metrics.columns (list): The columns of the metrics dataframe.
    """

    df = df.merge(metrics, on='movie_id', how='left')
    df[metrics.columns] = df[metrics.columns].fillna(0)

    return df, metrics.columns

