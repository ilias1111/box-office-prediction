import pandas as pd
import support_functions as sf


if __name__ == "__main__":
    # Load your dataframe here
    ml_data_reg = pd.read_csv('code/data_pipeline/ml_data_reg.csv')
    ml_data_reg = ml_data_reg[(ml_data_reg.ratio_adj > 0.1) & (ml_data_reg.ratio_adj < 100)]

    # Define your budget categories with thresholds and labels
    budget_categories = {
        'small': (0, 1_000_000),          # Small budget
        'medium': (1_000_000, 20_000_000), # Medium budget
        'large': (20_000_000, None)       # Large budget
    }

    # Call the function with the dataframe, the budget categories, and a filename prefix
    sf.split_and_save_dataframe(ml_data_reg, budget_categories, 'ml_data_reg')