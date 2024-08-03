# Configure the Google Cloud provider
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

# Create a VPC network
resource "google_compute_network" "vpc_network" {
  name                    = var.network_name
  auto_create_subnetworks = true
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
      image = "deeplearning-platform-release/m123"
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
    install-nvidia-driver = "True"
    ssh-keys = "ubuntu:${var.ssh_private_key}"
  }

  metadata_startup_script = <<-EOF
              #!/bin/bash
              sudo apt-get update
              sudo apt-get install -y git

              # Set up SSH for Git
              mkdir -p ~/.ssh
              gcloud secrets versions access latest --secret=ssh-key-secret > ~/.ssh/id_rsa
              chmod 600 ~/.ssh/id_rsa
              ssh-keyscan github.com >> ~/.ssh/known_hosts  # Adjust if not using GitHub

              # Clone your private repository
              git clone ${var.repo_url} /home/ubuntu/your-project
              cd /home/ubuntu/your-project

              # Function to check if a package is installed
              is_package_installed() {
                python -c "import $1" 2>/dev/null
                return $?
              }

              # Read requirements.txt and install only missing packages
              while read requirement; do
                package_name=$(echo $requirement | sed 's/[<>=].*//')
                if ! is_package_installed $package_name; then
                  pip install $requirement
                else
                  echo "Package $package_name is already installed, skipping."
                fi
              done < requirements.txt

              # Set up Jupyter notebook to run on startup
              echo "cd /home/ubuntu/your-project && jupyter notebook --ip=0.0.0.0 --no-browser --allow-root" | sudo tee -a /etc/rc.local

              # Create a swap file for additional memory management
              sudo fallocate -l 4G /swapfile
              sudo chmod 600 /swapfile
              sudo mkswap /swapfile
              sudo swapon /swapfile
              echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

              # Set some optimized kernel parameters for ML workloads
              echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
              echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf
              sudo sysctl -p
              EOF

  service_account {
    scopes = ["cloud-platform"]
  }

  # Allow stopping for update
  allow_stopping_for_update = true
}

# Create a Cloud Storage bucket for data and model storage
resource "google_storage_bucket" "ml_bucket" {
  name          = "${var.project_id}-ml-bucket-${var.environment}"
  location      = var.region
  force_destroy = true
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