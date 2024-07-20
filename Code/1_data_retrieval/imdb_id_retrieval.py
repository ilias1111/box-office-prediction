import requests
import csv
import pandas as pd
from sympy import im
import os

# Function to calculate and format the statistics of the DataFrame
def calculate_id_statistics(df):
    # Total number of rows
    number_of_rows = len(df)

    # Numbers of both tmdb_id and imdb_id being there (not null)
    both_ids_count = df.dropna(subset=['imdb_id', 'tmdb_id']).shape[0]

    # Number of only imdb_id being there (tmdb_id is null)
    only_imdb_count = df[df['tmdb_id'].isnull() & df['imdb_id'].notnull()].shape[0]

    # Number of only tmdb_id being there (imdb_id is null)
    only_tmdb_count = df[df['imdb_id'].isnull() & df['tmdb_id'].notnull()].shape[0]

    # Calculating percentage rates
    percentage_both = (both_ids_count / number_of_rows) * 100
    percentage_only_imdb = (only_imdb_count / number_of_rows) * 100
    percentage_only_tmdb = (only_tmdb_count / number_of_rows) * 100

    # Formatting the result as a string
    result_text = (
        f'Total number of rows: {number_of_rows}\n'
        f'Number of entries with both TMDB and IMDB IDs: {both_ids_count} '
        f'({percentage_both:.2f}%)\n'
        f'Number of entries with only IMDB ID: {only_imdb_count} '
        f'({percentage_only_imdb:.2f}%)\n'
        f'Number of entries with only TMDB ID: {only_tmdb_count} '
        f'({percentage_only_tmdb:.2f}%)'
    )
    
    return result_text





def get_tmdb_id_from_imdb_id(imdb_id):
    # Define the API URL, replacing {imdb_id} with the actual IMDb ID
    url = f"https://api.themoviedb.org/3/find/{imdb_id}?external_source=imdb_id"

    TOKEN = os.environ["TMDB_API_TOKEN"]
    # Define the headers with the bearer token authorization
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TOKEN}"
    }

    # Make the GET request to the API
    response = requests.get(url, headers=headers)

    # Check if the response status code is 200 (OK)
    if response.status_code == 200:
        data = response.json()
        # Check if movie_results is not empty
        if data.get("movie_results"):
            # Return the IMDb ID and the movie's ID from TMDb
            return imdb_id, data["movie_results"][0]["id"]
        else:
            # Return None if there are no movie results
            return None
    else:
        # If the response is not OK, print the status code and return None
        print(f"Error: Received status code {response.status_code}")
        return None
    

# Main loop to iterate through IMDb IDs and write them to a CSV file
def store_matches(imdb_ids, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        counter = 0
        if file.tell() == 0:
            writer.writerow(['imdb_id', 'tmdb_id'])
            
            for imdb_id in imdb_ids:
                try:
                    result = get_tmdb_id_from_imdb_id(imdb_id)
                    if result:
                        writer.writerow([result[0], result[1]])
                        counter += 1
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue
        
            print(f"Successfully stored {counter} matches in {output_file}")

if __name__ == '__main__':

    df = pd.read_csv('data/manual_data/wikidata_export.csv')
    result = calculate_id_statistics(df)
    print(result)

    imdb_ids_to_lookup = df[df['tmdb_id'].isnull() & df['imdb_id'].notnull()]['imdb_id'].tolist()
    output_file_path = 'Code/retrieved_data/tmdb_imdb_match.csv'  # The path to the CSV file where data will be stored
    store_matches(imdb_ids_to_lookup, output_file_path)