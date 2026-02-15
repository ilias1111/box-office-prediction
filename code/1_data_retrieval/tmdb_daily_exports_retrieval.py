import pandas as pd
import json
import requests
import gzip
import shutil
import os


def get_todays_date():
    from datetime import date

    today = date.today() - pd.Timedelta(days=1)
    return today.strftime("%m_%d_%Y")


today = get_todays_date()

FILE_LINKS = {
    "collection_export": [
        f"http://files.tmdb.org/p/exports/collection_ids_{today}.json.gz",
        "collection",
    ],
    "keyword_export": [
        f"http://files.tmdb.org/p/exports/keyword_ids_{today}.json.gz",
        "keyword",
    ],
    "person_export": [
        f"http://files.tmdb.org/p/exports/person_ids_{today}.json.gz",
        "person",
    ],
    "production_company_export": [
        f"http://files.tmdb.org/p/exports/production_company_ids_{today}.json.gz",
        "company",
    ],
}


def daily_export_to_csv(key, new_path):
    # Download the file
    r = requests.get(FILE_LINKS[key][0], stream=True)
    print(f"Downloading {key} file")
    print(f"File name: {FILE_LINKS[key][0]}")
    print(f"File size: {r.headers['Content-Length']} bytes")
    print(f"Status code: {r.status_code}")

    # If not exist, create the file
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    # Save the file
    with open(f"{new_path}/{key}.json.gz", "wb") as f:
        shutil.copyfileobj(r.raw, f)

    # Unzip the file
    with gzip.open(f"{new_path}/{key}.json.gz", "rb") as f_in:
        with open(f"{new_path}/{key}.json", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Load the file into a DataFrame
    with open(f"{new_path}/{key}.json", "r") as file:
        json_list = [json.loads(line) for line in file]

    # Remove the downloaded files
    os.remove(f"{new_path}/{key}.json.gz")
    os.remove(f"{new_path}/{key}.json")

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(json_list).rename(
        columns={"id": f"{FILE_LINKS[key][1]}_id", "name": f"{FILE_LINKS[key][1]}_name"}
    )
    df.to_csv(f"{new_path}/{key}.csv", index=False)


if __name__ == "__main__":
    for key in FILE_LINKS.keys():
        daily_export_to_csv(key, "data/retrieved_data/tmdb")
        print(f"{key} has been processed and saved as a csv file")
