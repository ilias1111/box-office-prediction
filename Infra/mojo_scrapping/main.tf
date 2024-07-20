terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable necessary APIs
resource "google_project_service" "services" {
  for_each = toset([
    "cloudfunctions.googleapis.com",
    "cloudscheduler.googleapis.com",
    "pubsub.googleapis.com",
    "firestore.googleapis.com",
    "storage.googleapis.com",
  ])
  service = each.key
  disable_on_destroy = false
}

# Cloud Storage buckets
resource "google_storage_bucket" "input_bucket" {
  name     = "${var.project_id}-movie-ids-input"
  location = var.region
}

resource "google_storage_bucket" "output_bucket" {
  name     = "${var.project_id}-movie-data-results"
  location = var.region
}

# Upload movie_ids.json to the input bucket
resource "google_storage_bucket_object" "movie_ids" {
  name   = "movie_ids.json"
  bucket = google_storage_bucket.input_bucket.name
  source = "${path.module}/movie_ids.json"  # Make sure this file exists in your Terraform directory
}

# Pub/Sub topics
resource "google_pubsub_topic" "processing_topic" {
  name = "movie-data-processing"
}

resource "google_pubsub_topic" "start_processing_topic" {
  name = "start-processing"
}

# Cloud Functions
resource "google_storage_bucket" "functions_bucket" {
  name     = "${var.project_id}-functions"
  location = var.region
}

# Create Batches Function
data "archive_file" "create_batches_source" {
  type        = "zip"
  source_dir  = "${path.module}/functions/create_batches"
  output_path = "/tmp/create_batches.zip"
}

resource "google_storage_bucket_object" "create_batches_archive" {
  name   = "create_batches-${data.archive_file.create_batches_source.output_md5}.zip"
  bucket = google_storage_bucket.functions_bucket.name
  source = data.archive_file.create_batches_source.output_path
}

resource "google_cloudfunctions_function" "create_batches" {
  name        = "create-batches"
  description = "Creates batches of movie IDs and publishes to Pub/Sub"
  runtime     = "python39"

  available_memory_mb   = 256
  source_archive_bucket = google_storage_bucket.functions_bucket.name
  source_archive_object = google_storage_bucket_object.create_batches_archive.name
  event_trigger {
    event_type = "google.pubsub.topic.publish"
    resource   = google_pubsub_topic.start_processing_topic.name
  }
  entry_point = "create_batches"
  
  environment_variables = {
    GCP_PROJECT = var.project_id
  }
}

# Process Batch Function
data "archive_file" "process_batch_source" {
  type        = "zip"
  source_dir  = "${path.module}/functions/process_batch"
  output_path = "/tmp/process_batch.zip"
}

resource "google_storage_bucket_object" "process_batch_archive" {
  name   = "process_batch-${data.archive_file.process_batch_source.output_md5}.zip"
  bucket = google_storage_bucket.functions_bucket.name
  source = data.archive_file.process_batch_source.output_path
}

resource "google_cloudfunctions_function" "process_batch" {
  name        = "process-batch"
  description = "Processes a batch of movie IDs"
  runtime     = "python39"

  available_memory_mb   = 256
  source_archive_bucket = google_storage_bucket.functions_bucket.name
  source_archive_object = google_storage_bucket_object.process_batch_archive.name
  event_trigger {
    event_type = "google.pubsub.topic.publish"
    resource   = google_pubsub_topic.processing_topic.name
  }
  entry_point = "process_batch"
  
  environment_variables = {
    GCP_PROJECT = var.project_id
  }
}
# Manual Aggregate Results Function
data "archive_file" "manual_aggregate_results_source" {
  type        = "zip"
  source_dir  = "${path.module}/functions/manual_aggregate_results"
  output_path = "/tmp/manual_aggregate_results.zip"
}

resource "google_storage_bucket_object" "manual_aggregate_results_archive" {
  name   = "manual_aggregate_results-${data.archive_file.manual_aggregate_results_source.output_md5}.zip"
  bucket = google_storage_bucket.functions_bucket.name
  source = data.archive_file.manual_aggregate_results_source.output_path
}

resource "google_cloudfunctions_function" "manual_aggregate_results" {
  name        = "manual-aggregate-results"
  description = "Manually triggered function to aggregate processed movie data results"
  runtime     = "python39"

  available_memory_mb   = 256
  source_archive_bucket = google_storage_bucket.functions_bucket.name
  source_archive_object = google_storage_bucket_object.manual_aggregate_results_archive.name
  trigger_http          = true
  entry_point           = "manual_aggregate_results"
  
  environment_variables = {
    GCP_PROJECT = var.project_id
  }
}

# IAM bindings for the Cloud Functions' service account
resource "google_storage_bucket_iam_member" "function_storage_access" {
  bucket = google_storage_bucket.input_bucket.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${var.project_id}@appspot.gserviceaccount.com"
}

resource "google_storage_bucket_iam_member" "function_storage_write_access" {
  bucket = google_storage_bucket.output_bucket.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${var.project_id}@appspot.gserviceaccount.com"
}

resource "google_project_iam_member" "function_firestore_access" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${var.project_id}@appspot.gserviceaccount.com"
}

# Cloud Scheduler Job
resource "google_cloud_scheduler_job" "start_processing_job" {
  name        = "start-movie-data-processing"
  description = "Triggers the movie data processing pipeline"
  schedule    = "0 0 * * *"

  pubsub_target {
    topic_name = google_pubsub_topic.start_processing_topic.id
    data       = base64encode("start")
  }
}

# Firestore
resource "google_firestore_database" "database" {
  project     = var.project_id
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"
}