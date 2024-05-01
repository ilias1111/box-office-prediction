import asyncio
import aiohttp
import csv
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIzYjBhZWRmYmQ4ZTg2NzFiN2Y0ZmFmYjNkNmVlY2ZjYSIsInN1YiI6IjVkZDgwYzAwZWY4YjMyMDAxNDhiODBlNSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.-vQJYhg3QoKR4TlL0Y85trvPmT0cnzv43zOVcbg8En0"#os.getenv('TMDB_API_TOKEN')

RATE_LIMIT = 50  # Set your preferred rate limit
BATCH_SIZE = 100  # Set your preferred batch size
RELEASE_DATES_CSV = 'data/processed_data/movie_release_dates.csv'
CAST_CSV = 'data/processed_data/movie_cast.csv'
CREW_CSV = 'data/processed_data/movie_crew.csv'

async def make_api_request_for_movie_details(session, movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    params = {"append_to_response": "release_dates,credits"}  # Include credits and release dates in the response
    async with session.get(url, headers=headers, params=params) as response:
        if response.status == 200:
            return await response.json()
        else:
            print(f"Error: HTTP {response.status} for movie ID {movie_id}")
            return None

def process_and_store_details(data, release_dates_writer, cast_writer, crew_writer):
    # Process and store release dates
    movie_id = data['id']
    for country_release in data.get('release_dates', {}).get('results', []):
        country_code = country_release['iso_3166_1']
        for release_date in country_release['release_dates']:
            release_dates_writer.writerow([
                movie_id, country_code, release_date['release_date'], 
                release_date['certification'], release_date['type']
            ])

    # Process and store cast
    for cast_member in data.get('credits', {}).get('cast', []):
        cast_writer.writerow([
            movie_id, cast_member['id'], cast_member['name'], cast_member['character']
        ])

    # Process and store crew
    for crew_member in data.get('credits', {}).get('crew', []):
        crew_writer.writerow([
            movie_id, crew_member['id'], crew_member['name'], crew_member['job'], crew_member['department']
        ])


async def make_api_request_for_movie_details(session, movie_id, semaphore):
    async with semaphore:  # This ensures that we don't exceed the rate limit
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        headers = {"Authorization": f"Bearer {TOKEN}"}
        params = {"append_to_response": "release_dates,credits"}
        while True:  # Keep trying until a successful response is received
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    await asyncio.sleep(1)  # Wait for 1 second before retrying
                else:
                    print(f"Error: HTTP {response.status} for movie ID {movie_id}")
                    return None

async def main():
    MOVIES_PROCESSED = 0
    tmdb_ids_to_retrieve = pd.read_csv('data/processed_data/movie.csv')['movie_id'].tolist()

    with open(RELEASE_DATES_CSV, mode='w', newline='', encoding='utf-8') as release_dates_file, \
         open(CAST_CSV, mode='w', newline='', encoding='utf-8') as cast_file, \
         open(CREW_CSV, mode='w', newline='', encoding='utf-8') as crew_file:

        release_dates_writer = csv.writer(release_dates_file)
        cast_writer = csv.writer(cast_file)
        crew_writer = csv.writer(crew_file)

        release_dates_writer.writerow(['Movie ID', 'Country Code', 'Release Date', 'Certification', 'Type'])
        cast_writer.writerow(['Movie ID', 'Actor ID', 'Name', 'Character'])
        crew_writer.writerow(['Movie ID', 'Crew ID', 'Name', 'Job', 'Department'])

        semaphore = asyncio.Semaphore(RATE_LIMIT)

        async with aiohttp.ClientSession() as session:
            tasks = [make_api_request_for_movie_details(session, movie_id, semaphore) for movie_id in tmdb_ids_to_retrieve]
            responses = await asyncio.gather(*tasks)

            if MOVIES_PROCESSED % (5* BATCH_SIZE) == 0:
                print(f"Processed {MOVIES_PROCESSED} movies.")
            for response in responses:
                if response:
                    process_and_store_details(response, release_dates_writer, cast_writer, crew_writer)
                    MOVIES_PROCESSED += 1

    print(f"Processed {MOVIES_PROCESSED} movies.")
    print(f'Finished processing {len(tmdb_ids_to_retrieve)} movies.')
    print(f'Finished processing successfully {MOVIES_PROCESSED / len(tmdb_ids_to_retrieve) * 100:.2f}% of movies.')


if __name__ == "__main__":
    asyncio.run(main())