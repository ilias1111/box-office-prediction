#!/bin/bash
set -e

echo "Updating apt-get..."
apt-get update

echo "Installing OpenCL libraries and clinfo..."
apt-get install -y ocl-icd-libopencl1 clinfo

echo "Configuring NVIDIA OpenCL for LightGBM..."
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

echo "Verifying installation with clinfo..."
clinfo | grep "Platform Name"

echo "Done! OpenCL should be working now."
