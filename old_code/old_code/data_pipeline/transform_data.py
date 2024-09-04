import pandas as pd
import numpy as np


def load_data():
    """
    Loads data from csv files and returns a dataframe
    """

    movies = pd.read_csv("code/data_pipeline/data/movies.csv")
    genres = pd.read_csv("code/data_pipeline/data/genres.csv")
    movies_with_sequel = pd.read_csv("code/data_pipeline/data/movies_with_sequel.csv")
    collection_metrics = pd.read_csv("code/data_pipeline/data/collection_metrics.csv")
    keywords = pd.read_csv("code/data_pipeline/data/keywords.csv")
    keywords_metrics = pd.read_csv("code/data_pipeline/data/keywords_metrics.csv")
    productions = pd.read_csv("code/data_pipeline/data/productions.csv")
    cast_keys = pd.read_csv("code/data_pipeline/data/cast_keys.csv")
    cast_data = pd.read_csv("code/data_pipeline/data/cast_data.csv")

    return (
        movies,
        genres,
        movies_with_sequel,
        collection_metrics,
        keywords,
        keywords_metrics,
        productions,
        cast_keys,
        cast_data,
    )


def transform_data(
    movies,
    genres,
    movies_with_sequel,
    collection_metrics,
    keywords,
    keywords_metrics,
    productions,
    cast_keys,
    cast_data,
):
    movies_with_sequel_values = movies_with_sequel.values

    # ## Transforming Genre data
    # genres_keys = pd.dataFrame(mydb.genres.find())
    # genres_keys = pd.dataFrame(genres_keys.groupby(by=['id', "name"]).groups.keys()).rename(
    #     columns={0: "genre_id", 1: "genre_name"})
    # genres['v'] = 1
    # genres = genres.merge(genres_keys, on='genre_id')
    # genres_one_hot = pd.pivot_table(genres, index="movie_id", columns='genre_name', values='v').fillna(0)

    ## Transforming Production Companies data
    productions_count = (
        productions.groupby(by="company_name")["movie_id"].count().reset_index()
    )
    prod_keys = productions_count[productions_count.movie_id > 10][
        "company_name"
    ].values
    prod_filtered = productions[productions["company_name"].isin(list(prod_keys))]
    prod_filtered["v"] = 1
    prod_one_hot = pd.pivot_table(
        prod_filtered, index="movie_id", columns="company_name", values="v"
    ).fillna(0)

    ## Transforming Keyword data
    keys_count = keywords.groupby(by="name")["keyword_id"].count().reset_index()
    keys = keys_count[keys_count.keyword_id > 10]["name"].values
    keywords_filtered = keywords[keywords["name"].isin(list(keys))]
    keywords_filtered["v"] = 1
    keywords_one_hot = pd.pivot_table(
        keywords_filtered, index="movie_id", columns="name", values="v"
    ).fillna(0)

    ## Transforming Cast data
    cast_filtered = cast_data[
        cast_data["person_id"].isin(cast_keys["person_id"].values)
    ]
    cast_filtered["v"] = 1
    cast_filtered["act_person_id"] = cast_filtered["person_id"].apply(
        lambda x: "cat_" + str(x)
    )
    cast_one_hot = pd.pivot_table(
        cast_filtered, index="movie_id", columns="act_person_id", values="v"
    ).fillna(0)

    # genres_feat = list(genres_one_hot.columns)
    keyword_feat = list(keywords_one_hot.columns)
    prod_feat = list(prod_one_hot.columns)
    cast_feat = list(cast_one_hot.columns)
    keywords_metrics_feat = list(keywords_metrics.columns[1:])
    collection_metrics_feat = list(collection_metrics.columns[1:])

    return (
        keyword_feat,
        prod_feat,
        cast_feat,
        keywords_metrics_feat,
        collection_metrics_feat,
        keywords_one_hot,
        prod_one_hot,
        cast_one_hot,
        keywords_metrics,
        collection_metrics,
        movies_with_sequel_values,
    )


def feature_creation(df, movies_with_sequel_values):
    df = movies.copy()

    df["is_sequel_my"] = (
        df["movie_id"]
        .apply(lambda x: 1 if x in movies_with_sequel_values else 0)
        .astype(np.uint8)
    )
    df["release_date"] = df["release_date"].astype("datetime64[ns]")
    # print(df['release_date'].astype('datetime64[ns]').dtype)
    df["day"] = df["release_date"].dt.day
    df["month"] = df["release_date"].dt.month.astype(np.uint8)
    df["year"] = df["release_date"].dt.year

    df["revenue_usd_adj"] = df.apply(
        lambda x: adjust_cpi(x["revenue_world"], x["year"]), axis=1
    )
    df["budget_usd_adj"] = df.apply(
        lambda x: adjust_cpi(x["budget_usd_adj"], x["year"]), axis=1
    )
    df["surplus"] = 0.5 * df["revenue_usd_adj"] - df["budget_usd_adj"]

    df = df[(df.revenue_usd_adj > 0) & (df.budget_usd_adj > 0)]
    df["revenue_usd_adj_log"] = np.log10(df["revenue_usd_adj"])
    df["budget_usd_adj_log"] = np.log10(df["budget_usd_adj"])

    df["tag"] = pd.qcut(
        df["surplus"], q=5, labels=[i for i in range(5)]
    )  # "Cat_"+str(i)
    df["my_tag"] = df["surplus"].apply(classify)

    df["ratio_adj"] = df["revenue_usd_adj"] / df["budget_usd_adj"]

    df["month_sin"] = df["month"].apply(lambda x: sin_cos(x, 12)[0])
    df["month_cos"] = df["month"].apply(lambda x: sin_cos(x, 12)[1])

    df = fix_certificates(df)

    return df


def merge_data(
    df,
    prod_one_hot,
    keywords_one_hot,
    cast_one_hot,
    collection_metrics,
    keywords_metrics,
):
    # df = df.merge(genres_one_hot, on='movie_id', how='left')
    df = df.merge(keywords_one_hot, on="movie_id", how="left")
    df = df.merge(prod_one_hot, on="movie_id", how="left")
    df = df.merge(cast_one_hot, on="movie_id", how="left")
    df = df.merge(keywords_metrics, on="movie_id", how="left")
    df = df.merge(collection_metrics, on="movie_id", how="left")

    df[keywords_metrics.columns] = df[keywords_metrics.columns].fillna(0)
    df[collection_metrics.columns] = df[collection_metrics.columns].fillna(0)
    # df[genres_one_hot.columns] = df[genres_one_hot.columns].fillna(0).astype(np.uint8)
    df[keywords_one_hot.columns] = (
        df[keywords_one_hot.columns].fillna(0).astype(np.uint8)
    )
    df[prod_one_hot.columns] = df[prod_one_hot.columns].fillna(0).astype(np.uint8)
    df[cast_one_hot.columns] = df[cast_one_hot.columns].fillna(0).astype(np.uint8)
    return df


movie_id = ["movie_id"]
basic_features = ["original_language", "runtime", "day", "month", "year"]
revenue_budget = ["revenue_usd_adj", "budget_usd_adj"]
revenue_budget_log = ["revenue_usd_adj_log", "budget_usd_adj_log"]
eng_features = ["month_cos", "month_sin", "is_sequel_my", "ageCert"]
#
# ## Adding Features to the dataset
# ml_data_reg = ml_data_reg[movie_id + basic_features + eng_features + keywords_metrics_feat + collection_metrics_feat +
#                  genres_feat + revenue_budget + revenue_budget_log]
#
# ## Filtering the dataset for outliers
# ml_data_reg = ml_data_reg[(ml_data_reg.ratio_adj > 0.1) & (ml_data_reg.ratio_adj < 100)]
#
# ## Dumming categorical features
# ml_data_reg = pd.get_dummies(ml_data_reg)
#
# ## Splitting the dataset for more targeted training
#
# ml_data_reg_s = ml_data_reg[(ml_data_reg.budget_usd_adj > 0) & (ml_data_reg.budget_usd_adj <= 1_000_000)]
# ml_data_reg_m = ml_data_reg[(ml_data_reg.budget_usd_adj > 1_000_000) & (ml_data_reg.budget_usd_adj <= 20_000_000)]
# ml_data_reg_l = ml_data_reg[(ml_data_reg.budget_usd_adj > 20_000_000)]

(
    movies,
    genres,
    movies_with_sequel,
    collection_metrics,
    keywords,
    keywords_metrics,
    productions,
    cast_keys,
    cast_data,
) = load_data()


(
    keyword_feat,
    prod_feat,
    cast_feat,
    keywords_metrics_feat,
    collection_metrics_feat,
    keywords_one_hot,
    prod_one_hot,
    cast_one_hot,
    keywords_metrics,
    collection_metrics,
    movies_with_sequel_values,
) = transform_data(
    movies,
    genres,
    movies_with_sequel,
    collection_metrics,
    keywords,
    keywords_metrics,
    productions,
    cast_keys,
    cast_data,
)
ml_data_reg = feature_creation(movies, movies_with_sequel_values)
ml_data_reg = merge_data(
    ml_data_reg,
    prod_one_hot,
    keywords_one_hot,
    cast_one_hot,
    collection_metrics,
    keywords_metrics,
)

ml_data_reg.to_csv("ml_data_reg.csv", index=False)


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

    df = df.merge(metrics, on="movie_id", how="left")
    df[metrics.columns] = df[metrics.columns].fillna(0)

    return df, metrics.columns


def one_hot_encode(
    data: pd.DataFrame, index_col: str, column_name: str, min_count: int
) -> pd.DataFrame:
    """
    Helper function to perform one-hot encoding on a given DataFrame based on a minimum count.
    """
    count = data.groupby(by=column_name)[index_col].count().reset_index()
    filtered_keys = count[count[index_col] > min_count][column_name].values
    filtered_data = data[data[column_name].isin(filtered_keys)]
    filtered_data["v"] = 1
    one_hot = pd.pivot_table(
        filtered_data, index=index_col, columns=column_name, values="v"
    ).fillna(0)
    return one_hot


def feature_creation(df, movies_with_sequel_values):
    df = df[(df.revenue_usd_adj > 0) & (df.budget_usd_adj > 0)]
    df["revenue_usd_adj_log"] = np.log10(df["revenue_usd_adj"])
    df["budget_usd_adj_log"] = np.log10(df["budget_usd_adj"])
    df["month_sin"] = df["month"].apply(lambda x: sin_cos(x, 12)[0])
    df["month_cos"] = df["month"].apply(lambda x: sin_cos(x, 12)[1])

    df["tag"] = pd.qcut(
        df["surplus"], q=5, labels=[i for i in range(5)]
    )  # "Cat_"+str(i)
    df["my_tag"] = df["surplus"].apply(classify)

    # # Load your dataframe here
    # ml_data_reg = pd.read_csv('code/data_pipeline/ml_data_reg.csv')
    # ml_data_reg = ml_data_reg[(ml_data_reg.ratio_adj > 0.1) & (ml_data_reg.ratio_adj < 100)]

    # # Define your budget categories with thresholds and labels
    # budget_categories = {
    #     'small': (0, 1_000_000),          # Small budget
    #     'medium': (1_000_000, 20_000_000), # Medium budget
    #     'large': (20_000_000, None)       # Large budget
    # }

    # # Call the function with the dataframe, the budget categories, and a filename prefix
    # sf.split_and_save_dataframe(ml_data_reg, budget_categories, 'ml_data_reg')
