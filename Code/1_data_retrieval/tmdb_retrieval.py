import asyncio
import aiohttp
import csv
import os
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
TOKEN = os.getenv("TMDB_API_TOKEN")
RATE_LIMIT = 50
BATCH_SIZE = 100
BASE_URL = "https://api.themoviedb.org/3"

# File paths
DATA_DIR = 'data/retrieved_data/tmdb'
CSV_FILES = {
    'release_dates': f'{DATA_DIR}/release_dates.csv',
    'cast': f'{DATA_DIR}/cast.csv',
    'crew': f'{DATA_DIR}/crew.csv',
    'spoken_languages': f'{DATA_DIR}/spoken_languages.csv',
    'genres': f'{DATA_DIR}/genres.csv',
    'production_companies': f'{DATA_DIR}/production_companies.csv',
    'production_countries': f'{DATA_DIR}/production_countries.csv',
    'keywords': f'{DATA_DIR}/keywords.csv',
    'collections': f'{DATA_DIR}/collections.csv',
    'movies': f'{DATA_DIR}/movies.csv',
}

async def make_api_request(session: aiohttp.ClientSession, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{BASE_URL}/{endpoint}"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    
    for attempt in range(3):  # Retry up to 3 times
        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    wait_time = int(response.headers.get('Retry-After', 1))
                    logging.warning(f"Rate limit reached. Waiting for {wait_time} seconds.")
                    await asyncio.sleep(wait_time)
                else:
                    logging.error(f"HTTP {response.status} for {url}")
                    return None
        except aiohttp.ClientError as e:
            logging.error(f"Request failed: {e}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    logging.error(f"Failed to get data for {url} after 3 attempts")
    return None

async def get_movie_details(session: aiohttp.ClientSession, tmdb_id: int, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    async with semaphore:
        return await make_api_request(session, f"movie/{tmdb_id}", {"append_to_response": "release_dates,external_ids,credits,keywords"})

def process_movie_details(data: Dict[str, Any], writers: Dict[str, csv.writer]) -> None:
    tmdb_id = data['id']

    # Process release dates
    for country_release in data.get('release_dates', {}).get('results', []):
        for release_date in country_release.get('release_dates', []):
            writers['release_dates'].writerow([
                tmdb_id, country_release.get('iso_3166_1'), release_date.get('release_date'),
                release_date.get('certification'), release_date.get('type')
            ])

    # Process cast and crew
    for cast_member in data.get('credits', {}).get('cast', []):
        writers['cast'].writerow([
            tmdb_id, cast_member.get('id'), cast_member.get('name'),
            cast_member.get('character'), cast_member.get('order')
        ])

    for crew_member in data.get('credits', {}).get('crew', []):
        writers['crew'].writerow([
            tmdb_id, crew_member.get('id'), crew_member.get('name'),
            crew_member.get('job'), crew_member.get('department')
        ])

    # Process other details
    for spoken_language in data.get('spoken_languages', []):
        writers['spoken_languages'].writerow([tmdb_id, spoken_language.get('iso_639_1'), spoken_language.get('english_name')])

    for genre in data.get('genres', []):
        writers['genres'].writerow([tmdb_id, genre.get('id'), genre.get('name')])

    for production_company in data.get('production_companies', []):
        writers['production_companies'].writerow([tmdb_id, production_company.get('id'), production_company.get('name')])

    for production_country in data.get('production_countries', []):
        writers['production_countries'].writerow([tmdb_id, production_country.get('iso_3166_1'), production_country.get('name')])

    for keyword in data.get('keywords', {}).get('keywords', []):
        writers['keywords'].writerow([tmdb_id, keyword.get('id'), keyword.get('name')])
    
    if data.get('belongs_to_collection'):
        writers['collections'].writerow([tmdb_id, data['belongs_to_collection'].get('id'), data['belongs_to_collection'].get('name')])

    # Process movie details
    writers['movies'].writerow([
        tmdb_id, data.get('title'), data.get('original_title'), data.get('original_language'),
        data.get('overview'), data.get('tagline'), data.get('release_date'), data.get('runtime'),
        data.get('budget'), data.get('revenue'), data.get('poster_path'), data.get('imdb_id'),
        data.get('external_ids', {}).get('facebook_id'), data.get('external_ids', {}).get('instagram_id'),
        data.get('external_ids', {}).get('twitter_id'), data.get('external_ids', {}).get('wikidata_id'),
    ])

async def main():
    tmdb_imdb_match = pd.read_csv('data/retrieved_data/tmdb_imdb_match.csv')['tmdb_id'].tolist()
    wikidata_export = pd.read_csv('data/manual_data/wikidata_export.csv')
    tmdb_ids = wikidata_export[wikidata_export['tmdb_id'].notnull()]['tmdb_id'].tolist()

    tmdb_ids_to_retrieve = list(set(tmdb_ids).union(set(tmdb_imdb_match)))

    logging.info(f'Starting to process {len(tmdb_ids_to_retrieve)} movies.')

    writers = {key: csv.writer(open(file_path, 'w', newline='', encoding='utf-8'))
               for key, file_path in CSV_FILES.items()}

    writers['release_dates'].writerow(['movie_id', 'country_code', 'release_date', 'certification', 'type'])
    writers['cast'].writerow(['movie_id', 'actor_id', 'name', 'character', 'order'])
    writers['crew'].writerow(['movie_id', 'crew_id', 'name', 'job', 'department'])
    writers['spoken_languages'].writerow(['movie_id', 'language', 'english_name'])
    writers['genres'].writerow(['movie_id', 'genre_id', 'genre_name'])
    writers['production_companies'].writerow(['movie_id', 'company_id', 'company_name'])
    writers['production_countries'].writerow(['movie_id', 'country_code', 'name'])
    writers['keywords'].writerow(['movie_id', 'keyword_id', 'keyword_name'])
    writers['collections'].writerow(['movie_id', 'collection_id', 'collection_name'])
    writers['movies'].writerow(['movie_id', 'title', 'original_title', 'original_language', 'overview', 'tagline', 'release_date', 'runtime', 'budget', 'revenue', 'poster_path', 'imdb_id', 'facebook_id', 'instagram_id', 'twitter_id', 'wikidata_id'])

    semaphore = asyncio.Semaphore(RATE_LIMIT)

    async with aiohttp.ClientSession() as session:
        tasks = [get_movie_details(session, tmdb_id, semaphore) for tmdb_id in tmdb_ids_to_retrieve]
        
        processed_count = 0
        for response in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing movies"):
            movie_data = await response
            if movie_data:
                process_movie_details(movie_data, writers)
                processed_count += 1

    logging.info(f"Processed {processed_count} movies successfully.")
    logging.info(f'Finished processing {processed_count / len(tmdb_ids_to_retrieve) * 100:.2f}% of movies.')

if __name__ == "__main__":
    asyncio.run(main())