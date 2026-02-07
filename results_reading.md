Interpreting Regression Metrics for Box Office Prediction
Here is a guide to reading the "Regression Results" tables.

1. R² (R-Squared)
What it is: The "Goodness of Fit."
Range: -∞ to 1.0 (1.0 is perfect).
How to read it:
0.7+: Strong predictive power. The model explains most of the variance in revenue.
0.3 - 0.5: Moderate. The model gets the general trend but misses details.
< 0: The model is worse than just guessing the average revenue for every movie.
2. MAE (Mean Absolute Error)
What it is: The average dollar amount the model is wrong by.
How to read it: "On average, our prediction is off by $45 Million."
Pros/Cons: Easy to understand, but can be skewed by a few massive failures (blockbusters).
3. Median AE (Median Absolute Error)
What it is: The "Typical" Error. The middle value of all errors.
How to read it: "For a typical movie, the prediction is off by only $12 Million."
Why comparing MAE vs. Median matters:
If MAE >> Median, it means the model generally works well but fails spectacularly on a few outliers (likely high-budget blockbusters).
If MAE ≈ Median, the error is consistent across all movies.
4. RMSE (Root Mean Squared Error)
What it is: Similar to MAE but "punishes" large errors more heavily.
How to read it: Always higher than MAE.
Use case: If you want to penalize a model that makes one huge $500M mistake more than a model that makes ten smaller $50M mistakes.
5. MAPE (Mean Absolute Percentage Error)
What it is: The average error as a percentage of the actual revenue.
How to read it: "On average, the prediction is 90% away from the actual value."
Warning: Can be misleading for small movies (e.g., predicting $200k for a $100k movie is a 100% error, but only a $100k difference).
6. Bias (Mean Residual)
What it is: The average direction of the error.
Formula: $Actual - Predicted$
How to read it:
Positive ($+): The model Under-predicts. (It predicted less than actual). "Conservative."
Negative ($-$): The model Over-predicts. (It predicted more than actual). "Hype machine."
Near Zero: The model is "Unbiased" (its over-predictions cancel out its under-predictions).
7. Max Error
What it is: The single worst prediction in the dataset.
How to read it: "In the worst case scenario, this model was off by $450 Million."
Use case: Risk assessment. "Can we afford to be this wrong?"