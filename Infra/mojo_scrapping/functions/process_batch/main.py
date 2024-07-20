import base64
import json
import time
import functions_framework
from google.cloud import firestore
from google.cloud import storage
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_exponential
import os

last_request_time = 0
PROJECT_ID = os.environ.get('GCP_PROJECT')

def rate_limited_request(url, delay=1):
    global last_request_time
    current_time = time.time()
    if current_time - last_request_time < delay:
        time.sleep(delay - (current_time - last_request_time))
    response = requests.get(url)
    last_request_time = time.time()
    return response

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def access_specific_movie_id_from_mojo(id):
    url = f'https://www.boxofficemojo.com/title/{id}/'
    try:
        response = rate_limited_request(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('div', class_='a-section a-spacing-none mojo-performance-summary-table')
        if not table:
            raise ValueError("Could not find the expected table on the page")
        lines = table.find_all('div')
    except (RequestException, ValueError, AttributeError, IndexError) as e:
        print(f"Error processing movie ID {id}: {str(e)}")
        raise

    try:
        domestic = int(lines[0].find('span', class_='money').text.strip().replace(",", "").replace("$", ""))
    except:
        domestic = 0
    try:
        international = int(lines[1].find('span', class_='money').text.strip().replace(",", "").replace("$", ""))
    except:
        international = 0
    
    try:
        worldwide = int(lines[2].find('span', class_='money').text.strip().replace(",", "").replace("$", ""))
    except:
        worldwide = 0
        
    return {
        "domestic": domestic,
        "international": international,
        "worldwide": worldwide
    }

@functions_framework.cloud_event
def process_batch(cloud_event):
    pubsub_message = base64.b64decode(cloud_event.data["message"]["data"]).decode()
    message_data = json.loads(pubsub_message)
    
    movie_ids = message_data["movie_ids"]
    timestamp = message_data["timestamp"]
    batch_id = message_data["batch_id"]
    
    results = {}
    for movie_id in movie_ids:
        try:
            results[movie_id] = access_specific_movie_id_from_mojo(movie_id)
        except Exception as e:
            print(f"Failed to process movie ID {movie_id} after retries: {str(e)}")
            results[movie_id] = {"error": str(e)}
    
    # Store results in Firestore
    db = firestore.Client()
    collection_name = f'movie_data_{timestamp}'
    batch_ref = db.collection(collection_name).document(batch_id)
    batch_ref.set(results)

    # Update processing state
    state_ref = db.collection('processing_state').document('batch_status')
    state_ref.set({
        f'{timestamp}_{batch_id}': 'processed'
    }, merge=True)


    print(f"Processed batch {batch_id} of {len(movie_ids)} movie IDs and stored in collection {collection_name}")