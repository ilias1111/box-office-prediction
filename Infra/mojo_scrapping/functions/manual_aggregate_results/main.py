import functions_framework
from google.cloud import firestore, storage
import json
import os

PROJECT_ID = os.environ.get("GCP_PROJECT")


@functions_framework.http
def manual_aggregate_results(request):
    # Get the timestamp from the request
    timestamp = request.args.get("timestamp")
    if not timestamp:
        return "Please provide a timestamp", 400

    collection_name = f"movie_data_{timestamp}"

    db = firestore.Client()
    results = db.collection(collection_name).stream()

    aggregated_data = {}
    for doc in results:
        aggregated_data.update(doc.to_dict())

    # Store aggregated results in Cloud Storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(f"{PROJECT_ID}-movie-data-results")
    blob = bucket.blob(f"{timestamp}_aggregated_results.json")
    blob.upload_from_string(json.dumps(aggregated_data))

    print(
        f"Aggregated results for {len(aggregated_data)} movies from collection {collection_name}"
    )
    return f"Aggregation complete for collection {collection_name}", 200
