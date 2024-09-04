import pandas as pd

from set_up_connections import *


def get_data_from_bigquery():
    """
    Get data from BigQuery
    """

    credentials, project_id = get_gbq_credentials()
    with open("Code/data_pipeline/queries.json") as f:
        queries = json.load(f)

    for key, value in queries.items():
        # print(f'Getting {key} data from BigQuery')
        try:
            df = pd.read_gbq(value, project_id=project_id, credentials=credentials)
            df.to_csv(f"data/{key}.csv", index=False)
            print(f"{key} data saved to data/{key}.csv")
        except Exception:
            print(f"Error getting {key} data from BigQuery")
            pass


if __name__ == "__main__":
    get_data_from_bigquery()
