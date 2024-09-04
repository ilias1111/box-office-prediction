import asyncio
import aiohttp
import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd

load_dotenv()
TOKEN = os.getenv("TMDB_API_TOKEN")
DB_URI = os.getenv("NEON_DB_URI")

BATCH_SIZE = 100  # Set your preferred batch size


def process_and_store_batch(batch_data, cursor, conn):
    for data in batch_data:
        process_and_store_data(data, cursor, conn)
    conn.commit()  # Commit after processing a batch


async def make_api_request_for_movie(session, movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            return await response.json()
        else:
            print(f"Error: HTTP {response.status} for movie ID {movie_id}")
            return None


def process_and_store_data(data, cursor, conn):
    release_date = data["release_date"] if data["release_date"] else None

    # Insert movie data into the movies table
    # Adapt the fields and values according to your table's schema
    query = """
    INSERT INTO movies (
        id, imdb_id, title, original_title, overview, tagline, original_language,
        status, runtime, revenue, budget, poster_path, release_date,
        vote_average, vote_count
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    ) ON CONFLICT (id) DO NOTHING;
    """
    cursor.execute(
        query,
        (
            data["id"],
            data["imdb_id"],
            data["title"],
            data["original_title"],
            data["overview"],
            data["tagline"],
            data["original_language"],
            data["status"],
            data["runtime"],
            data["revenue"],
            data["budget"],
            data["poster_path"],
            release_date,
            data["vote_average"],
            data["vote_count"],
        ),
    )

    # Insert genres
    for genre in data["genres"]:
        genre_query = """
        INSERT INTO genres (movie_id, genre_id, name)
        VALUES (%s, %s, %s)
        ON CONFLICT DO NOTHING;
        """
        cursor.execute(genre_query, (data["id"], genre["id"], genre["name"]))

    # Insert production companies
    for company in data["production_companies"]:
        company_query = """
        INSERT INTO production_companies (movie_id, company_id, name, logo_path, origin_country)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING;
        """
        cursor.execute(
            company_query,
            (
                data["id"],
                company["id"],
                company["name"],
                company["logo_path"],
                company["origin_country"],
            ),
        )

    # Insert production countries
    for country in data["production_countries"]:
        country_query = """
        INSERT INTO production_countries (movie_id, country_code, name)
        VALUES (%s, %s, %s)
        ON CONFLICT DO NOTHING;
        """
        cursor.execute(
            country_query, (data["id"], country["iso_3166_1"], country["name"])
        )

    # Insert spoken languages
    for language in data["spoken_languages"]:
        language_query = """
        INSERT INTO spoken_languages (movie_id, language_code, name)
        VALUES (%s, %s, %s)
        ON CONFLICT DO NOTHING;
        """
        cursor.execute(
            language_query, (data["id"], language["iso_639_1"], language["name"])
        )

    # Commit changes
    conn.commit()


def create_tables(cursor, conn):
    tables = [
        """CREATE TABLE IF NOT EXISTS movies (
    id INTEGER PRIMARY KEY,
    imdb_id VARCHAR(255),
    title VARCHAR(255),
    original_title VARCHAR(255),
    overview TEXT,
    tagline VARCHAR(255),
    original_language VARCHAR(2),

    status VARCHAR(100),
    runtime INTEGER,

    revenue BIGINT,
    budget BIGINT,

    poster_path VARCHAR(255),
    release_date DATE,

    vote_average DECIMAL(10, 3),
    vote_count INTEGER
        );""",
        """CREATE TABLE IF NOT EXISTS genres (
            movie_id INTEGER,
            genre_id INTEGER,
            name VARCHAR(255),
            PRIMARY KEY (movie_id, genre_id),
            FOREIGN KEY (movie_id) REFERENCES movies(id)
        );""",
        """CREATE TABLE IF NOT EXISTS production_companies (
            movie_id INTEGER,
            company_id INTEGER,
            name VARCHAR(255),
            logo_path VARCHAR(255),
            origin_country VARCHAR(2),
            PRIMARY KEY (movie_id, company_id),
            FOREIGN KEY (movie_id) REFERENCES movies(id)
        );""",
        """CREATE TABLE IF NOT EXISTS production_countries (
            movie_id INTEGER,
            country_code VARCHAR(2),
            name VARCHAR(255),
            PRIMARY KEY (movie_id, country_code),
            FOREIGN KEY (movie_id) REFERENCES movies(id)
        );""",
        """CREATE TABLE IF NOT EXISTS spoken_languages (
            movie_id INTEGER,
            language_code VARCHAR(2),
            name VARCHAR(255),
            PRIMARY KEY (movie_id, language_code),
            FOREIGN KEY (movie_id) REFERENCES movies(id)
        );""",
    ]
    for table_sql in tables:
        cursor.execute(table_sql)
    conn.commit()


def connect_to_database():
    conn = psycopg2.connect(DB_URI)
    return conn, conn.cursor()


def get_existing_movie_ids(cursor):
    query = "SELECT id FROM movies;"
    cursor.execute(query)
    return [row[0] for row in cursor.fetchall()]


def retrieve_movie_ids(file_path, existing_ids):
    df = pd.read_csv(file_path)
    all_ids = df[~df["tmdb_id"].isnull()]["tmdb_id"].tolist()
    return [tmdb_id for tmdb_id in all_ids if tmdb_id not in existing_ids][:1000]


async def main():
    conn = psycopg2.connect(DB_URI)
    cursor = conn.cursor()
    create_tables(cursor, conn)

    MOVIES_PROCESSED = 0
    existing_ids = get_existing_movie_ids(cursor)
    tmdb_ids_to_retrieve = retrieve_movie_ids("wikidata_results.csv", existing_ids)

    async with aiohttp.ClientSession() as session:
        tasks = [
            make_api_request_for_movie(session, movie_id)
            for movie_id in tmdb_ids_to_retrieve
        ]
        responses = await asyncio.gather(*tasks)

        batch_data = []
        for movie_id, response in zip(tmdb_ids_to_retrieve, responses):
            if response:
                batch_data.append(response)
                MOVIES_PROCESSED += 1
                if len(batch_data) >= BATCH_SIZE:
                    process_and_store_batch(batch_data, cursor, conn)
                    print(f"Processed batch of {BATCH_SIZE} movies.")
                    batch_data = []  # Reset the batch

        # Process any remaining items in the batch
        if batch_data:
            process_and_store_batch(batch_data, cursor, conn)

        cursor.close()
        conn.close()

    print(f"Processed {MOVIES_PROCESSED} movies.")
    print(f"Finished processing {len(tmdb_ids_to_retrieve)} movies.")
    print(
        f"Finished processing succesfully {MOVIES_PROCESSED / len(tmdb_ids_to_retrieve) * 100:.2f}% of movies."
    )


if __name__ == "__main__":
    asyncio.run(main())
