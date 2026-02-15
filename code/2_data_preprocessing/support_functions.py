import pandas as pd
import logging
import re
from unidecode import unidecode
import cpi
import numpy as np
import uroman as ur

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

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
    "BPjM Restricted": "R",
}


def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    original_count = df.shape[0]
    current_count = original_count

    # Filter movies with runtime >= 70 minutes
    df = df[df["runtime"] >= 70]
    removed = current_count - df.shape[0]
    current_count = df.shape[0]
    print(
        f"Removed {removed} movies with runtime < 70 minutes. Remaining: {current_count}"
    )

    # Filter movies with revenue > 0
    df = df[df["revenue"] > 0]
    removed = current_count - df.shape[0]
    current_count = df.shape[0]
    print(f"Removed {removed} movies with revenue <= 0. Remaining: {current_count}")

    # Filter movies with budget > 0
    df = df[df["budget"] > 0]
    removed = current_count - df.shape[0]
    current_count = df.shape[0]
    print(f"Removed {removed} movies with budget <= 0. Remaining: {current_count}")

    # Filter movies released between 1970-01-01 and 2020-01-01
    df = df[(df["release_date"] >= "1970-01-01") & (df["release_date"] <= "2020-01-01")]
    removed = current_count - df.shape[0]
    current_count = df.shape[0]
    print(
        f"Removed {removed} movies released before 1970 or after 2020. Remaining: {current_count}"
    )

    total_removed = original_count - current_count
    print(f"Total movies removed: {total_removed}")
    print(f"Final number of movies: {current_count}")

    return df


def adjust_cpi(amount: float, date: int) -> float:
    if date < 2021:
        try:
            return cpi.inflate(amount, date)
        except Exception as e:
            logging.warning(f"CPI adjustment failed for {amount} on {date}: {e}")
            return -1
    else:
        return amount


def init_data_loading():
    DATA_DIR_TMDB = "./data/retrieved_data/tmdb"
    DATA_DIR_MANUAL = "./data/manual_data"
    DATA_DIR_WIKIPEDIA = "./data/retrieved_data/wikipedia"
    movie = pd.read_csv(f"{DATA_DIR_TMDB}/movies.csv")
    genre_mapping = pd.read_csv(f"{DATA_DIR_TMDB}/genres.csv")
    keyword_mapping = pd.read_csv(f"{DATA_DIR_TMDB}/keywords.csv")
    cast_mapping = pd.read_csv(f"{DATA_DIR_TMDB}/cast.csv")
    crew_mapping = pd.read_csv(f"{DATA_DIR_TMDB}/crew.csv")
    collection_mapping = pd.read_csv(f"{DATA_DIR_TMDB}/collections.csv")
    production_mapping = pd.read_csv(f"{DATA_DIR_TMDB}/production_companies.csv")

    keyword = pd.read_csv(f"{DATA_DIR_TMDB}/keyword_export.csv")
    production = pd.read_csv(f"{DATA_DIR_TMDB}/production_company_export.csv")
    collection = pd.read_csv(f"{DATA_DIR_TMDB}/collection_export.csv")
    movie_release_dates = pd.read_csv(f"{DATA_DIR_TMDB}/release_dates.csv")

    production_countries = pd.read_csv(f"{DATA_DIR_TMDB}/production_countries.csv")
    spoken_languages = pd.read_csv(f"{DATA_DIR_TMDB}/spoken_languages.csv")

    movie_financial_data_usd = pd.read_csv(
        f"{DATA_DIR_WIKIPEDIA}/movie_financial_data_usd.csv"
    )
    manual_adjustments = pd.read_csv(f"{DATA_DIR_MANUAL}/manual_adjustments.csv")
    mojo_scrapping = pd.read_csv(f"{DATA_DIR_MANUAL}/parsed_mojo.csv")

    return (
        movie,
        genre_mapping,
        keyword_mapping,
        cast_mapping,
        crew_mapping,
        collection_mapping,
        production_mapping,
        spoken_languages,
        production_countries,
        keyword,
        production,
        collection,
        movie_release_dates,
        movie_financial_data_usd,
        mojo_scrapping,
        manual_adjustments,
    )


def create_financial_data_usd(
    movie: pd.DataFrame,
    wikipedia_movie_financial_data_usd: pd.DataFrame,
    mojo_scrapping: pd.DataFrame,
    manual_adjustments: pd.DataFrame,
) -> pd.DataFrame:
    movie["revenue"] = movie["revenue"].replace(0, np.nan)
    movie["budget"] = movie["budget"].replace(0, np.nan)
    movie["revenue_original"] = movie["revenue"]
    movie["budget_original"] = movie["budget"]

    manual_adjustments = manual_adjustments[
        ["movie_id", "revenue_fix", "budget_fix"]
    ].replace(0, np.nan)
    movie = movie.merge(manual_adjustments, on="movie_id", how="left")

    wikipedia_movie_financial_data_usd = (
        wikipedia_movie_financial_data_usd[
            ["movie_id", "budget_avg_usd", "box_office_avg_usd", "url"]
        ]
        .rename(
            columns={
                "budget_avg_usd": "budget_wiki",
                "box_office_avg_usd": "revenue_wiki",
                "url": "url_wiki",
            }
        )
        .replace(0, np.nan)
    )
    movie = movie.merge(wikipedia_movie_financial_data_usd, on="movie_id", how="left")

    mojo_scrapping = (
        mojo_scrapping.reset_index()
        .rename(columns={"Unnamed: 0": "imdb_id", "worldwide": "revenue_mojo"})[
            ["imdb_id", "revenue_mojo", "error"]
        ]
        .replace(0, np.nan)
    )
    movie = movie.merge(mojo_scrapping, on="imdb_id", how="left")

    movie["revenue"] = (
        movie["revenue_fix"]
        .fillna(movie["revenue_mojo"])
        .fillna(movie["revenue"])
        .fillna(movie["revenue_wiki"])
        .fillna(0)
    )
    movie["budget"] = (
        movie["budget_fix"]
        .fillna(movie["budget"])
        .fillna(movie["budget_wiki"])
        .fillna(0)
    )

    return movie


def movie_table_processing(df):
    df["release_date"] = pd.to_datetime(df["release_date"])
    # df['release_date'] = df['release_date'].tz_localize(None).astype('datetime64[ns]')

    df["quarter"] = df["release_date"].dt.quarter
    df["month"] = df["release_date"].dt.month.astype(np.uint8)
    df["year"] = df["release_date"].dt.year

    df["revenue_usd_adj"] = df.apply(
        lambda x: adjust_cpi(x["revenue"], x["year"]), axis=1
    )
    df["budget_usd_adj"] = df.apply(
        lambda x: adjust_cpi(x["budget"], x["year"]), axis=1
    )
    df["surplus"] = 0.5 * df["revenue_usd_adj"] - df["budget_usd_adj"]
    df["ratio_adj"] = df["revenue_usd_adj"] / df["budget_usd_adj"]
    df["roi"] = df["surplus"] / df["budget_usd_adj"]

    df["ageCert"] = df["ageCert"].replace(CERTIFICATE_MAPPINGS).fillna("U")

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
        is_parent = row["company_id"] == row["parent_id"]
        parent_rows = df[df["company_id"] == row["parent_id"]]

        # Check if parent_rows is not empty
        if not parent_rows.empty:
            parent_name = parent_rows["company_name"].values[0]
        else:
            # Handle cases where parent_id doesn't match any company_id
            parent_name = None

        transformed_data.append(
            {
                "parent_id": row["parent_id"],
                "parent_name": parent_name,
                "company_id": row["company_id"],
                "company_name": row["company_name"],
                "is_parent": is_parent,
            }
        )

    return pd.DataFrame(transformed_data)


def calculate_release_info_with_checks(df):
    # Ensure 'release_date' is in datetime format
    df["release_date"] = pd.to_datetime(df["release_date"])

    # Filter for theatrical releases and physical/digital releases
    countries = ["US", "CN", "FR", "GB", "JP"]
    theatrical_releases = df[df["type"].isin([3]) & df["country_code"].isin(countries)]
    theatrical_releases_limited = df[df["type"].isin([4])]
    theatrical_releases_wide = df[df["type"].isin([3])]
    physical_digital_releases = df[
        df["type"].isin([4, 5]) & df["country_code"].isin(countries)
    ]

    # Get the earliest theatrical release date for each movie in each country
    theatrical_release_dates = (
        theatrical_releases.groupby(["movie_id", "country_code"])["release_date"]
        .min()
        .unstack()
    )
    min_theatrical_release_dates = theatrical_releases.groupby("movie_id")[
        "release_date"
    ].min()

    no_theatrical_releases_wide = theatrical_releases_wide.groupby("movie_id")[
        "release_date"
    ].count()
    no_theatrical_releases_limited = theatrical_releases_wide.groupby("movie_id")[
        "release_date"
    ].count()
    # Get the number of theatrical releases within 15 days of the first release min_theatrical_release_dates
    theatrical_releases_wide["first_release_date"] = theatrical_releases_wide.merge(
        min_theatrical_release_dates.rename("first_release_date"), on="movie_id"
    )["first_release_date"]
    theatrical_releases_wide["days_from_first_release"] = (
        theatrical_releases_wide["release_date"]
        - theatrical_releases_wide["first_release_date"]
    ).dt.days
    no_theatrical_releases_wide_15 = (
        theatrical_releases_wide[
            theatrical_releases_wide["days_from_first_release"] <= 15
        ]
        .groupby("movie_id")["release_date"]
        .count()
    )

    # Get the earliest physical/digital release date for each movie
    min_physical_digital_dates = physical_digital_releases.groupby("movie_id")[
        "release_date"
    ].min()

    # Get the certificates for the us release of a movie
    us_certificates = theatrical_releases[theatrical_releases["country_code"] == "US"]
    us_certificates = us_certificates.groupby("movie_id")["certification"].first()

    # Prepare the final DataFrame
    final_df = pd.DataFrame(index=df["movie_id"].unique())
    final_df = final_df.join(theatrical_release_dates[countries], how="left")
    final_df = final_df.join(
        min_physical_digital_dates.rename("digital_physical_date"), how="left"
    )
    final_df = final_df.join(
        min_theatrical_release_dates.rename("min_theatrical_release_date"), how="left"
    )
    final_df = final_df.join(
        no_theatrical_releases_wide.rename("no_theatrical_releases_wide"), how="left"
    )
    # final_df = final_df.join(no_theatrical_releases_limited.rename('no_theatrical_releases_limited'), how='left')
    final_df = final_df.join(
        no_theatrical_releases_wide_15.rename("no_theatrical_releases_wide_15"),
        how="left",
    )
    final_df = final_df.join(us_certificates.rename("ageCert"), how="left")

    # Add flags for release in each country and overall release
    for country in countries:
        final_df[f"is_released__{country}"] = final_df[country].notna()
    final_df["is_released__scope"] = final_df[countries].notna().any(axis=1)

    # Calculate days from US release to digital/physical release
    final_df["digital_physical_date"] = pd.to_datetime(
        final_df["digital_physical_date"].fillna(pd.Timestamp("2050-01-01", tz="UTC")),
        utc=True,
    )
    final_df["days_from_us_release"] = (
        (final_df["digital_physical_date"] - final_df["US"]).dt.days
    ).fillna(-1)

    # Selecting the required columns
    final_df = final_df[
        [
            "is_released__US",
            "is_released__CN",
            "is_released__FR",
            "is_released__GB",
            "is_released__JP",
            "is_released__scope",
            "digital_physical_date",
            "days_from_us_release",
            "ageCert",
        ]
    ]
    final_df.reset_index(inplace=True)
    final_df.rename(columns={"index": "movie_id"}, inplace=True)

    final_df["is_first_released_in_cinemas"] = np.where(
        final_df["days_from_us_release"] > 0, 1, 0
    ).astype(bool)
    final_df["is_first_released_in_cinemas_safe"] = np.where(
        final_df["days_from_us_release"] > 60, 1, 0
    ).astype(bool)
    final_df["is_released__scope"] = final_df["is_released__scope"].astype(bool)

    conditions = [
        (final_df["is_released__scope"]) & (~final_df["is_first_released_in_cinemas"]),
        (final_df["is_released__scope"])
        & (final_df["is_first_released_in_cinemas_safe"]),
        (final_df["is_released__scope"])
        & (final_df["is_first_released_in_cinemas"])
        & (~final_df["is_first_released_in_cinemas_safe"]),
        ~final_df["is_released__scope"],
    ]

    # Define the corresponding categories for each condition
    categories = [
        "Streaming release",
        "Far streaming release",
        "Close streaming release",
        "Not released in major markets",
    ]

    # Create the new categorical column
    final_df["release_category"] = np.select(conditions, categories, default="Other")

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
    uroman = ur.Uroman()

    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    # Romanize non-Latin characters
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].apply(lambda x: uroman.romanize_string(x))

    # Transliterate to ASCII
    df[column_name] = df[column_name].apply(lambda x: unidecode(x))

    # Replace special characters with underscores
    df[column_name] = df[column_name].apply(lambda x: re.sub(r"\W+", "_", x))

    # Strip leading and trailing spaces
    df[column_name] = df[column_name].str.strip()

    # Convert to lowercase
    df[column_name] = df[column_name].str.lower()

    return df


if __name__ == "__main__":
    data = {
        "col1": [
            "Hello World!",
            "Èspañol test",
            "Français & Deutsch!",
            "Русский язык",
            "日本語",
            "中文",
            "العربية",
            "🌍",
        ]
    }
    df = pd.DataFrame(data)
    df = convert_to_machine_friendly(df, "col1")
    print(df)
