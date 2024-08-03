# Configure the Google Cloud provider
terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Define variables
variable "project_id" {
  description = "The ID of the GCP project"
  type        = string
}

variable "region" {
  description = "The region to deploy resources"
  default     = "us-central1"
}

variable "zone" {
  description = "The zone to deploy resources"
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment (dev or prod)"
  default     = "dev"
}

variable "network_name" {
  description = "Name of the VPC network"
  default     = "ml-network"
}

variable "repo_url" {
  description = "URL of the public Git repository"
  type        = string
}

variable "tmdb_api_token" {
  description = "API token for TMDB"
  type        = string
  sensitive   = true
}

# Define local values
locals {
  machine_types = {
    dev  = "n1-standard-2"  # 2 vCPUs, 7.5 GB memory
    prod = "n1-standard-16" # 16 vCPUs, 60 GB memory
  }
  disk_sizes = {
    dev  = 50
    prod = 200
  }
}

# Enable required APIs
resource "google_project_service" "services" {
  for_each = toset([
    "compute.googleapis.com"
  ])
  service = each.key
  disable_on_destroy = false
}

# Create a VPC network
resource "google_compute_network" "vpc_network" {
  name                    = var.network_name
  auto_create_subnetworks = true
  depends_on              = [google_project_service.services]
}

# Create a firewall rule to allow SSH
resource "google_compute_firewall" "allow-ssh" {
  name    = "${var.network_name}-allow-ssh"
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
}

# Create a firewall rule to allow Jupyter
resource "google_compute_firewall" "allow-jupyter" {
  name    = "${var.network_name}-allow-jupyter"
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["8888"]
  }

  source_ranges = ["0.0.0.0/0"]
}

# Create a spot VM instance with Deep Learning VM
resource "google_compute_instance" "spot_vm_instance" {
  name         = "ml-spot-vm-instance-${var.environment}"
  machine_type = local.machine_types[var.environment]
  zone         = var.zone

  tags = ["ml-instance", "jupyter", var.environment]

  scheduling {
    preemptible       = true
    automatic_restart = false
    provisioning_model = "SPOT"
  }

  boot_disk {
    initialize_params {
      image = "projects/ml-images/global/images/c0-deeplearning-common-cpu-v20240708-debian-11"
      size  = local.disk_sizes[var.environment]
      type  = "pd-ssd"
    }
  }

  network_interface {
    network = google_compute_network.vpc_network.name
    access_config {
      // Ephemeral IP
    }
  }

  metadata = {
    enable-oslogin = "TRUE"  # Enable OS Login for SSH access
    TMDB_API_TOKEN = var.tmdb_api_token
  }

  metadata_startup_script = <<-EOF
              #!/bin/bash
              
              # Clone your public repository
              git clone ${var.repo_url} 

              # Install any additional requirements
              if [ -f requirements.txt ]; then
                pip install -r requirements.txt
              fi

              EOF

  service_account {
    scopes = ["cloud-platform"]
  }

  # Allow stopping for update
  allow_stopping_for_update = true

  depends_on = [google_project_service.services]
}

# Create a Cloud Storage bucket for data and model storage
resource "google_storage_bucket" "ml_bucket" {
  name          = "${var.project_id}-ml-bucket-${var.environment}"
  location      = var.region
  force_destroy = true

  depends_on = [google_project_service.services]
}

# Output the instance IP and bucket name
output "instance_ip" {
  value = google_compute_instance.spot_vm_instance.network_interface[0].access_config[0].nat_ip
}

output "bucket_name" {
  value = google_storage_bucket.ml_bucket.name
}

output "instance_name" {
  value = google_compute_instance.spot_vm_instance.name
}

output "machine_type" {
  value = google_compute_instance.spot_vm_instance.machine_type
}