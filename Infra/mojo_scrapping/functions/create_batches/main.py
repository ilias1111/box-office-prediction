import functions_framework
from google.cloud import storage, firestore, pubsub_v1
import json
import os
from datetime import datetime

BATCH_SIZE = 30
PROJECT_ID = os.environ.get("GCP_PROJECT")
PUBSUB_TOPIC = f"projects/{PROJECT_ID}/topics/movie-data-processing"


@functions_framework.cloud_event
def create_batches(cloud_event):
    try:
        # Generate timestamp for this batch creation run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Read movie IDs from Cloud Storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(f"{PROJECT_ID}-movie-ids-input")
        blob = bucket.blob("movie_ids.json")

        print(f"Attempting to download blob: gs://{bucket.name}/{blob.name}")

        movie_ids_data = blob.download_as_string()
        print(f"Downloaded data: {movie_ids_data[:100]}...")  # Print first 100 chars

        movie_ids = json.loads(movie_ids_data)
        print(f"Parsed {len(movie_ids)} movie IDs")

        # Create batches and publish to Pub/Sub
        publisher = pubsub_v1.PublisherClient()
        db = firestore.Client()

        for i in range(0, len(movie_ids), BATCH_SIZE):
            batch = movie_ids[i : i + BATCH_SIZE]
            batch_id = f"batch_{i//BATCH_SIZE}"
            data = json.dumps(
                {"movie_ids": batch, "timestamp": timestamp, "batch_id": batch_id}
            ).encode("utf-8")
            publisher.publish(PUBSUB_TOPIC, data)

            # Update Firestore
            doc_ref = db.collection("processing_state").document("batch_status")
            doc_ref.set({f"{timestamp}_{batch_id}": "published"}, merge=True)

        # Set total batches in Firestore
        doc_ref.set(
            {f"{timestamp}_total_batches": len(movie_ids) // BATCH_SIZE + 1}, merge=True
        )

        print(
            f"Created and published {len(movie_ids) // BATCH_SIZE + 1} batches with timestamp {timestamp}"
        )

    except Exception as e:
        print(f"Error in create_batches: {str(e)}")
        raise  # Re-raise the exception to ensure the function is marked as failed
