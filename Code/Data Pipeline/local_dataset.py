import pandas as pd

from set_up_connections import *


def get_data_from_bigquery(credentials_file):

    """
    Get data from BigQuery
    """

    credentials, project_id = get_gbq_credentials()
    with open('./queries.json') as f:
        queries = json.load(f)

    for key, value in queries.items():
        df = pd.read_gbq(value, project_id=project_id, credentials=credentials)
        df.to_csv(f'./Data/{key}.csv', index=False)


# get_data_from_bigquery('./credentials.json')
