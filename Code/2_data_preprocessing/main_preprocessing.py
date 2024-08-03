import pandas as pd
from torch import ge
import support_functions as sf


if __name__ == "__main__":

    # Load your dataframe here
    movie, genre_mapping, keyword_mapping, cast_mapping, crew_mapping, collection_mapping, production_mapping, \
        spoken_languages, production_countries, \
        keyword, production, collection, movie_release_dates, movie_financial_data_usd, \
            mojo_scrapping, manual_adjustments = sf.init_data_loading()

    genre = pd.read_csv('data/manual_data/static_genre.csv')

    movie = sf.filter_rows(movie)

    keyword = sf.convert_to_machine_friendly(keyword, 'keyword_name')
    production = sf.convert_to_machine_friendly(production, 'company_name')
    collection = sf.convert_to_machine_friendly(collection, 'collection_name')

    # production = sf.transform_company_hierarchy(production)
    movie_release_features = sf.calculate_release_info_with_checks(movie_release_dates)

    print(movie_release_features.head(10))
    
    # Join keyword and keyword_mapping on keyword_id
    keyword_processed = keyword_mapping[['movie_id','keyword_id']].merge(keyword, on='keyword_id', how='left')
    collection_processed = collection_mapping[['movie_id','collection_id']].merge(collection, on='collection_id', how='left')
    genre_processed = genre_mapping.merge(genre, on='genre_id', how='left')
    production_processed = production_mapping[['movie_id','company_id']].merge(production, on='company_id', how='left')
    movie = movie.merge(movie_release_features, on='movie_id', how='left')

    movie = sf.create_financial_data_usd(movie, movie_financial_data_usd, mojo_scrapping, manual_adjustments)
    movie = sf.movie_table_processing(movie)





    movie.to_csv(
        './data/processed_data/movie.csv', index=False)
    keyword_processed.to_csv(
        './data/processed_data/keyword.csv', index=False)
    production_processed.to_csv(
        './data/processed_data/production.csv', index=False)
    collection_processed.to_csv(
        './data/processed_data/collection.csv', index=False)
    genre_processed.to_csv(
        './data/processed_data/genre.csv', index=False)
    
    cast_mapping.to_csv(
        './data/processed_data/movie_cast.csv', index=False)
    crew_mapping.to_csv(
        './data/processed_data/movie_crew.csv', index=False)

    spoken_languages.to_csv(
        './data/processed_data/spoken_languages.csv', index=False)
    production_countries.to_csv(
        './data/processed_data/production_countries.csv', index=False)