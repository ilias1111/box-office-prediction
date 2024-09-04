import pandas as pd
import requests
from datetime import datetime
import logging
from typing import List, Tuple, Dict
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
MANUAL_DATA_DIR = "Data/manual_data/currency_data"


class CurrencyStats:
    def __init__(self):
        self.conversions = {}

    def record_conversion(self, currency, success):
        if currency not in self.conversions:
            self.conversions[currency] = {"success": 0, "fail": 0}

        if success:
            self.conversions[currency]["success"] += 1
        else:
            self.conversions[currency]["fail"] += 1

    def get_stats(self):
        stats = {}
        for currency, counts in self.conversions.items():
            total = counts["success"] + counts["fail"]
            success_rate = (counts["success"] / total) * 100 if total > 0 else 0
            stats[currency] = {
                "success": counts["success"],
                "fail": counts["fail"],
                "success_rate": success_rate,
            }

            total_success = sum(stat["success"] for stat in stats.values())
            total_fail = sum(stat["fail"] for stat in stats.values())
            total = total_success + total_fail
            total_success_rate = (total_success / total) * 100 if total > 0 else 0
            stats["total"] = {
                "success": total_success,
                "fail": total_fail,
                "success_rate": total_success_rate,
            }

        return stats


currency_stats = CurrencyStats()


def get_monthly_exchange_rates(
    from_currency: str, to_currency: str
) -> Dict[str, float]:
    if from_currency == to_currency:
        return {datetime.now().strftime("%Y-%m-%d"): 1.0}

    file_path = os.path.join(MANUAL_DATA_DIR, f"USD_{from_currency}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df["Price"] = df["Price"].replace(",", "", regex=True).astype(float)

        return df.set_index("Date")["Price"].to_dict()

    params = {
        "function": "FX_MONTHLY",
        "from_symbol": from_currency,
        "to_symbol": to_currency,
        "apikey": ALPHA_VANTAGE_API_KEY,
    }
    response = requests.get(ALPHA_VANTAGE_URL, params=params)
    data = response.json()

    if "Time Series FX (Monthly)" in data:
        rates = {
            k: float(v["4. close"]) for k, v in data["Time Series FX (Monthly)"].items()
        }
        pd.DataFrame(rates.items(), columns=["Date", "Price"]).to_csv(
            file_path, index=False
        )
        logging.info(f"Exchange rates for {from_currency} saved to {file_path}")
        return rates

    return {}


def get_all_exchange_rates(
    currency_tuple: Tuple[str, ...],
) -> Dict[str, Dict[str, float]]:
    return {
        currency: get_monthly_exchange_rates(currency, "USD")
        for currency in set(currency_tuple)
        if currency != "USD"
    }


def get_rate_for_date(
    rates: Dict[str, Dict[str, float]], currency: str, date: datetime
) -> float:
    if currency == "USD":
        return 1.0

    monthly_rates = rates.get(currency, {})
    while date.strftime("%Y-%m-%d") not in monthly_rates and date.year >= 1999:
        date -= pd.Timedelta(days=1)

    return monthly_rates.get(date.strftime("%Y-%m-%d"), None)


def convert_to_usd(row: pd.Series, rates: Dict[str, Dict[str, float]]) -> float:
    if (
        pd.isna(row["currency"])
        or pd.isna(row["release_date"])
        or pd.isna(row["amount"])
    ):
        currency_stats.record_conversion(row["currency"], False)
        return None, None

    rate = get_rate_for_date(rates, row["currency"], row["release_date"])

    if rate is not None:
        currency_stats.record_conversion(row["currency"], True)
        return row["amount"] * (1 / rate), (1 / rate)
    else:
        currency_stats.record_conversion(row["currency"], False)
        return None, None


def process_financial_data(
    df: pd.DataFrame, output_file: str, columns_pairs: List[Tuple[str, str]]
):
    # drop the rows that all the columns are null
    cols = [k for i in columns_pairs for k in i]
    df = df.dropna(subset=cols, how="all")
    df["release_date"] = (
        pd.to_datetime(df["release_date"]).dt.to_period("M").dt.to_timestamp()
    )
    currency_columns = tuple(set(pair[1] for pair in columns_pairs))

    # Calulate the unique exchange currency rates
    unique = set(df[[i for i in currency_columns]].values.flatten())
    rates = get_all_exchange_rates(unique)

    for amount_col, currency_col in columns_pairs:
        df[f"{amount_col}_usd"] = df.apply(
            lambda row: convert_to_usd(
                {
                    "amount": row[amount_col],
                    "currency": row[currency_col],
                    "release_date": row["release_date"],
                },
                rates,
            )[0],
            axis=1,
        )
        df[f"{amount_col}_usd_rate"] = df.apply(
            lambda row: convert_to_usd(
                {
                    "amount": row[amount_col],
                    "currency": row[currency_col],
                    "release_date": row["release_date"],
                },
                rates,
            )[1],
            axis=1,
        )

    df.to_csv(output_file, index=False)
    logger.info(f"Processed data saved to {output_file}")

    stats = currency_stats.get_stats()
    # Order by success rate
    stats_s = {
        k: v
        for k, v in sorted(
            stats.items(), key=lambda item: item[1]["success_rate"], reverse=True
        )
    }
    stats_s.pop("total")
    for currency, stat in stats_s.items():
        logger.info(
            f"{currency} - Success: {stat['success']}, Fail: {stat['fail']}, Success Rate: {stat['success_rate']:.2f}%"
        )

    # Total number of conversions
    logger.info(
        f"Total conversions - Success: {stats['total']['success']} Fail: {stats['total']['fail']}, Success Rate: {stats['total']['success_rate']:.2f}%"
    )


if __name__ == "__main__":
    input_file = pd.read_csv(
        "Data/retrieved_data/wikipedia/movie_financial_data.csv"
    ).rename({"tmdb_id": "movie_id"}, axis=1)
    movies = pd.read_csv("Data/retrieved_data/tmdb/movies.csv")

    columns_pairs = [
        ("budget_avg", "budget_currency"),
        ("box_office_avg", "box_office_currency"),
    ]
    input_file = input_file.merge(
        movies[["movie_id", "release_date"]], on="movie_id", how="left"
    )
    output_file = "data/retrieved_data/wikipedia/movie_financial_data_usd.csv"

    # We want either the budget or box office to be present
    input_file = input_file.dropna(subset=["budget_avg", "box_office_avg"], how="all")

    process_financial_data(input_file, output_file, columns_pairs)
