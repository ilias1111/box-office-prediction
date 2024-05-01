import pandas as pd
import numpy as np
from math import sin, cos, pi
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




