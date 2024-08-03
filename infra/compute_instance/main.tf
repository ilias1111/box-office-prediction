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
  description = "URL of the private Git repository"
  type        = string
}

variable "ssh_private_key" {
  description = "SSH private key for Git repository access"
  type        = string
  sensitive   = true
}

variable "ssh_public_key" {
  description = "SSH public key for VM access"
  type        = string
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
    "secretmanager.googleapis.com",
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

# Create a Secret Manager secret for the SSH key
resource "google_secret_manager_secret" "ssh_key" {
  secret_id = "ssh-key-secret"

  replication {
    auto {}
  }

  depends_on = [google_project_service.services]
}

# Store the SSH key in the secret
resource "google_secret_manager_secret_version" "ssh_key_version" {
  secret = google_secret_manager_secret.ssh_key.id
  secret_data = var.ssh_private_key
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
    ssh-keys = "jupyter:${var.ssh_public_key}"
  }


  metadata_startup_script = <<-EOF
              #!/bin/bash
              
              # Set up SSH for Git
              mkdir -p /home/jupyter/.ssh
              gcloud secrets versions access latest --secret=ssh-key-secret > /home/jupyter/.ssh/id_rsa
              chmod 600 /home/jupyter/.ssh/id_rsa
              ssh-keyscan github.com >> /home/jupyter/.ssh/known_hosts

              # Clone your private repository
              git clone ${var.repo_url} /home/jupyter/your-project
              cd /home/jupyter/your-project

              # Install any additional requirements
              if [ -f requirements.txt ]; then
                pip install -r requirements.txt
              fi

              # Set up Jupyter notebook to run on startup (if not already configured)
              if ! grep -q "jupyter notebook" /etc/systemd/system/jupyter.service; then
                echo "[Unit]
                Description=Jupyter Notebook

                [Service]
                Type=simple
                PIDFile=/run/jupyter.pid
                ExecStart=/opt/deeplearning/bin/jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --notebook-dir=/home/jupyter
                User=jupyter
                Group=jupyter
                WorkingDirectory=/home/jupyter
                Restart=always
                RestartSec=10

                [Install]
                WantedBy=multi-user.target" | sudo tee /etc/systemd/system/jupyter.service

                sudo systemctl enable jupyter.service
                sudo systemctl start jupyter.service
              fi

              # Create a swap file for additional memory management
              if [ ! -f /swapfile ]; then
                sudo fallocate -l 4G /swapfile
                sudo chmod 600 /swapfile
                sudo mkswap /swapfile
                sudo swapon /swapfile
                echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
              fi

              # Set some optimized kernel parameters for ML workloads
              echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
              echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf
              sudo sysctl -p

              # Add function to run ML jobs with nohup
              echo '
              run_ml_job() {
                  if [ $# -eq 0 ]; then
                      echo "Usage: run_ml_job <script.py> [args...]"
                      return 1
                  fi

                  script_name=$$(basename "$$1")
                  log_file="$${script_name%.*}_$$(date +%Y%m%d_%H%M%S).log"
                  nohup python "$$@" > "$$log_file" 2>&1 &
                  echo "Started ML job: $$1"
                  echo "Log file: $$log_file"
                  echo "PID: $$!"
              }
              ' | sudo tee -a /etc/profile.d/ml_utils.sh

              # Make the function available to all users
              sudo chmod +x /etc/profile.d/ml_utils.sh
              EOF

  
  service_account {
    scopes = ["cloud-platform"]
  }

  # Allow stopping for update
  allow_stopping_for_update = true

  depends_on = [google_project_service.services, google_secret_manager_secret.ssh_key]
}

# Grant the VM's service account access to Secret Manager
resource "google_project_iam_member" "secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_compute_instance.spot_vm_instance.service_account[0].email}"
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