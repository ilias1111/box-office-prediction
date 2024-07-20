# Output the URL of the create-batches function
output "create_batches_function_url" {
  value       = google_cloudfunctions_function.create_batches.https_trigger_url
  description = "The URL of the create-batches Cloud Function"
}

# Output the URL of the process-batch function
output "process_batch_function_url" {
  value       = google_cloudfunctions_function.process_batch.https_trigger_url
  description = "The URL of the process-batch Cloud Function"
}

# Output the URL of the manual-aggregate-results function
output "manual_aggregate_results_function_url" {
  value       = google_cloudfunctions_function.manual_aggregate_results.https_trigger_url
  description = "The URL of the manual-aggregate-results Cloud Function"
}

# Output the name of the Pub/Sub topic for processing
output "processing_topic_name" {
  value       = google_pubsub_topic.processing_topic.name
  description = "The name of the Pub/Sub topic for movie data processing"
}

# Output the name of the Pub/Sub topic for starting the process
output "start_processing_topic_name" {
  value       = google_pubsub_topic.start_processing_topic.name
  description = "The name of the Pub/Sub topic for starting the processing"
}

# Output the name of the input bucket
output "input_bucket_name" {
  value       = google_storage_bucket.input_bucket.name
  description = "The name of the Cloud Storage bucket for input data"
}

# Output the name of the output bucket
output "output_bucket_name" {
  value       = google_storage_bucket.output_bucket.name
  description = "The name of the Cloud Storage bucket for output data"
}

# Output the name of the Cloud Scheduler job
output "scheduler_job_name" {
  value       = google_cloud_scheduler_job.start_processing_job.name
  description = "The name of the Cloud Scheduler job that triggers the processing"
}

# Output the Firestore database name
output "firestore_database_name" {
  value       = google_firestore_database.database.name
  description = "The name of the Firestore database"
}