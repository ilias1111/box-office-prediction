import asyncio
import aiohttp
from bs4 import BeautifulSoup, Tag
import re
from statistics import mean
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging
import time

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
RATE_LIMIT = 100
MAX_RETRIES = 3
RETRY_DELAY = 8  # seconds

async def scrape_wikipedia_page(session: aiohttp.ClientSession, url: str, max_retries: int = MAX_RETRIES) -> Optional[BeautifulSoup]:
    for attempt in range(max_retries):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    return BeautifulSoup(html, 'html.parser')
                elif response.status == 429:  # Too Many Requests
                    retry_after = int(response.headers.get('Retry-After'*3, RETRY_DELAY))
                    logging.warning(f"Rate limited. Retrying after {retry_after} seconds.")
                    await asyncio.sleep(retry_after)
                else:
                    logging.warning(f"Failed to fetch {url}. Status: {response.status}. Attempt {attempt + 1}/{max_retries}")
            await asyncio.sleep(RETRY_DELAY)  # Wait before retrying
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logging.error(f"Error fetching {url}: {str(e)}. Attempt {attempt + 1}/{max_retries}")
            await asyncio.sleep(RETRY_DELAY)  # Wait before retrying
    logging.error(f"Failed to fetch {url} after {max_retries} attempts")
    return None

def extract_info_from_infobox(soup: BeautifulSoup, info_type: str) -> Optional[str]:
    try:
        if soup:
            infobox = soup.find('table', {'class': 'infobox vevent'})
            if infobox:
                row = infobox.find('th', string=info_type)
                if row:
                    value = row.find_next_sibling('td')
                    if value:
                        return value.text.strip()
    except Exception as e:
        logging.error(f"Error extracting {info_type} from infobox: {str(e)}")
    return None


def parse_financial_info(info_string: Optional[str]) -> Tuple[Optional[Tuple[float, float, float]], str, Optional[str]]:
    if not info_string or info_string.lower() == 'none':
        return None, "No information found", None

    try:
        # Remove citations and lowercase the string
        info_string = re.sub(r'\[.*?\]', '', info_string.lower())

        # Define scale patterns
        scale_patterns = {
            'trillion': 1e12, 'billion': 1e9, 'million': 1e6, 'thousand': 1e3,
            'trillions': 1e12, 'billions': 1e9, 'millions': 1e6, 'thousands': 1e3,
            'k': 1e3, 'b': 1e9, 'm': 1e6, 't': 1e12,
            'lakh': 1e5, 'crore': 1e7, 'crores': 1e7, 'arab': 1e9, 'kharab': 1e11, 'neel': 1e13,
            'padma': 1e15, 'shankh': 1e17, 'mahashankh': 1e19, 'vrinda': 1e21,
            'nghìn': 1e3, 'triệu': 1e6, 'tỷ': 1e9
        }

        # Check for admissions
        if 'admissions' in info_string:
            return None, "Admissions data, not financial", None

        # Detect all currencies and their positions
        currencies = detect_currencies(info_string)
        if not currencies:
            return None, f"No recognized currency in: {info_string}", None

        # Use the first currency found
        currency, currency_pos = currencies[0]

        # Find all number-scale pairs
        number_scale_pairs = re.findall(r'([\d,.]+)(?:\s*[–-]\s*([\d,.]+))?\s*([a-zA-Z]+)?', info_string)
        
        if not number_scale_pairs:
            return None, f"No numeric values found in: {info_string}", currency

        converted_values = []
        for pair in number_scale_pairs:
            try:
                start_value = float(pair[0].replace(',', ''))
                end_value = float(pair[1].replace(',', '')) if pair[1] else start_value
                scale = pair[2] if pair[2] else ''
                
                if scale.lower() in scale_patterns:
                    start_value *= scale_patterns[scale.lower()]
                    end_value *= scale_patterns[scale.lower()]
                
                converted_values.extend([start_value, end_value])
            except ValueError:
                continue

        if not converted_values:
            return None, f"Couldn't convert any values in: {info_string}", currency

        min_value = min(converted_values)
        max_value = max(converted_values)
        avg_value = sum(converted_values) / len(converted_values)

        details = f"{currency} Min: {min_value:.2f}, Max: {max_value:.2f}, Avg: {avg_value:.2f}"
        return (min_value, max_value, avg_value), details, currency
    except Exception as e:
        return None, f"Error parsing: {info_string}. Error: {str(e)}", None

def detect_currencies(info_string: str) -> list:
    currency_map = {
        'gbp': 'GBP', '£': 'GBP', 'pound': 'GBP',
        'eur': 'EUR', '€': 'EUR', 'euro': 'EUR',
        'jpy': 'JPY', '¥': 'JPY', 'yen': 'JPY',
        'cny': 'CNY', 'yuan': 'CNY', 'renminbi': 'CNY',
        'inr': 'INR', '₹': 'INR', 'rupee': 'INR',
        'cad': 'CAD', 'c$': 'CAD', 'canadian dollar': 'CAD', 'ca$': 'CAD',
        'aud': 'AUD', 'a$': 'AUD', 'australian dollar': 'AUD', 'aud$': 'AUD', '$a': 'AUD',
        'chf': 'CHF', 'fr.': 'CHF', 'swiss franc': 'CHF',
        'sek': 'SEK', 'kr': 'SEK', 'swedish krona': 'SEK',
        'nok': 'NOK', 'norwegian krone': 'NOK',
        'dkk': 'DKK', 'danish krone': 'DKK',
        'rub': 'RUB', 'ruble': 'RUB', '₽': 'RUB',
        'brl': 'BRL', 'real': 'BRL', 'r$': 'BRL',
        'mxn': 'MXN', 'peso': 'MXN', 'mexican peso': 'MXN',
        'dm': 'DEM', 'deutsche mark': 'DEM', 'german mark': 'DEM', 'mark': 'DEM', 'dm$': 'DEM', 'dem': 'DEM',
        'frf': 'FRF', 'french franc': 'FRF', 'franc': 'FRF',
        'itl': 'ITL', 'lira': 'ITL', 'italian lira': 'ITL', '₤': 'ITL',
        'esp': 'ESP', 'peseta': 'ESP', 'spanish peseta': 'ESP', '₧': 'ESP',
        'thb': 'THB', 'baht': 'THB', '฿': 'THB',
        'dong': 'VND', '₫': 'VND', 'vnd': 'VND', 'vietnamese': 'VND',
        'hkd': 'HKD', 'hong kong dollar': 'HKD', 'hk$': 'HKD',
        'sgd': 'SGD', 'singapore dollar': 'SGD', 'sg$': 'SGD',
        '₺': 'TRY', 'try': 'TRY', 'turkish lira': 'TRY',
        'myr': 'MYR', 'ringgit': 'MYR', 'malaysian ringgit': 'MYR',
        'czk': 'CZK', 'koruna': 'CZK', 'czech koruna': 'CZK', 'czk': 'CZK',
        'huf': 'HUF', 'forint': 'HUF', 'hungarian forint': 'HUF', 'ft': 'HUF', 
        "franc": "CHF", "swiss franc": "CHF", "swiss": "CHF", "fr": "CHF",
        'pln': 'PLN', 'zł': 'PLN', 'zloty': 'PLN', 'polish zloty': 'PLN',
        '₦': 'NGN', 'naira': 'NGN', 'nigerian naira': 'NGN',
        '₩': 'KRW', 'won': 'KRW', 'korean won': 'KRW', 'krw': 'KRW',
        '₱': 'PHP', 'peso': 'PHP', 'philippine peso': 'PHP', 'php': 'PHP',
        'usd': 'USD', '$': 'USD', 'dollar': 'USD', 'us$': 'USD',
    }
    
    lower_info = info_string.lower()
    found_currencies = []
    for key, value in currency_map.items():
        pos = lower_info.find(key)
        if pos != -1:
            found_currencies.append((value, pos))
    
    return sorted(found_currencies, key=lambda x: x[1])


def parse_external_links(soup: BeautifulSoup) -> Dict[str, str]:
    external_links = {}
    try:
        external_links_section = soup.find('span', {'id': 'External_links'})
        
        if external_links_section:
            current_element = external_links_section.parent
            while current_element and current_element.name != 'ul':
                current_element = current_element.find_next_sibling()
            
            if current_element and current_element.name == 'ul':
                for li in current_element.find_all('li', recursive=False):
                    links = li.find_all('a', href=True)
                    if len(links) >= 2:
                        url = links[0]['href']
                        if url.startswith('http'):
                            name = links[1].get_text(strip=True)
                            external_links[name] = url
                    elif len(links) == 1 and links[0]['href'].startswith('http'):
                        name = links[0].get_text(strip=True)
                        url = links[0]['href']
                        external_links[name] = url
    except Exception as e:
        logging.error(f"Error parsing external links: {str(e)}")
    return external_links

async def get_movie_info(session: aiohttp.ClientSession, url: str, info_types: List[str]) -> Dict[str, any]:
    soup = await scrape_wikipedia_page(session, url)
    movie_info = {'title': url}

    if soup is None:
        logging.error(f"Failed to fetch content for {url}")
        movie_info['error'] = "Failed to fetch content"
        return movie_info

    for info_type in info_types:
        try:
            info_string = extract_info_from_infobox(soup, info_type)
            amount_tuple, details, currency = parse_financial_info(info_string)
            
            movie_info[info_type] = {
                'min_value': amount_tuple[0] if amount_tuple else None,
                'max_value': amount_tuple[1] if amount_tuple else None,
                'average_value': amount_tuple[2] if amount_tuple else None,
                'currency': currency,
                'details': details,
                'original_string': info_string
            }
        except Exception as e:
            logging.error(f"Error processing {info_type} for {url}: {str(e)}")
            movie_info[info_type] = {
                'min_value': None, 'max_value': None, 'average_value': None,
                'currency': None, 'details': f"Error: {str(e)}",
                'original_string': None
            }

    try:
        movie_info['external_links'] = parse_external_links(soup)
    except Exception as e:
        logging.error(f"Error parsing external links for {url}: {str(e)}")
        movie_info['external_links'] = {}

    return movie_info

async def process_movie(session: aiohttp.ClientSession, data: Tuple[int, str], info_types: List[str] = ['Budget', 'Box office']) -> Dict[str, any]:
    tmdb_id, url = data
    processed_info = {'url': url, 'tmdb_id': int(tmdb_id)}

    try:
        movie_info = await get_movie_info(session, url, info_types)
        
        for info_type in info_types:
            info = movie_info.get(info_type, {})
            info_type = info_type.replace(' ', '_').lower()
            processed_info.update({
                f'{info_type.lower()}_original_string': info.get('original_string'),
                f'{info_type.lower()}_currency': info.get('currency'),
                f'{info_type.lower()}_min': info.get('min_value'),
                f'{info_type.lower()}_max': info.get('max_value'),
                f'{info_type.lower()}_avg': info.get('average_value'),
                f'{info_type.lower()}_error': info.get('details') if 'Error:' in info.get('details', '') else None
            })
        
        processed_info['external_links'] = movie_info.get('external_links', {})
        processed_info['error'] = movie_info.get('error')
        
    except Exception as e:
        logging.error(f"Error processing movie {tmdb_id} ({url}): {str(e)}")
        for info_type in info_types:
            info_type = info_type.replace(' ', '_').lower()
            processed_info.update({
                f'{info_type.lower()}_original_string': None,
                f'{info_type.lower()}_currency': None,
                f'{info_type.lower()}_min': None,
                f'{info_type.lower()}_max': None,
                f'{info_type.lower()}_avg': None,
                f'{info_type.lower()}_error': str(e)
            })
        processed_info['external_links'] = {}
        processed_info['error'] = str(e)

    return processed_info

async def main():
    try:
        wikidata_export = pd.read_csv('data/manual_data/wikidata_export.csv')
        list_of_movies = wikidata_export[(wikidata_export['wikipedia_url'].notnull()) & (wikidata_export['tmdb_id'].notnull())][['tmdb_id','wikipedia_url']].to_records(index=False)#[:1000]

        print(f"Processing {len(list_of_movies)} movies")
        # How many unique movies are there?
        ids = [movie[0] for movie in list_of_movies]
        print(f"Unique movies: {len(set(ids))}")

        logging.info(f"Processing {len(list_of_movies)} movies")

        parsed_data = []
        semaphore = asyncio.Semaphore(RATE_LIMIT)

        async with aiohttp.ClientSession() as session:
            tasks = [process_movie(session, data) for data in list_of_movies]
            
            for result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing movies"):
                async with semaphore:
                    parsed_data.append(await result)

        parsed_data_df = pd.DataFrame(parsed_data)

        parsed_data_df.to_csv('data/retrieved_data/wikipedia/movie_financial_data.csv', index=False)
        logging.info(f"There are {len(parsed_data_df)} rows")
        logging.info(f"There are {parsed_data_df['budget_min'].notnull().sum()} cases where the budget is available")
        logging.info(f"There are {parsed_data_df['box_office_min'].notnull().sum()} cases where the box office is available")
        logging.info(f"\n{parsed_data_df.head()}")

        # Log error statistics
        for column in parsed_data_df.columns:
            if 'error' in column:
                error_count = parsed_data_df[column].notnull().sum()
                logging.info(f"There are {error_count} errors in {column}")

    except Exception as e:
        logging.error(f"An error occurred in the main function: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())