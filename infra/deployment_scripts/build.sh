#!/bin/bash
set -e

echo "Building SentinelVision Docker image..."

cd "$(dirname "$0")/../.."

docker build -t sentinelvision/inference:latest -f infra/docker/Dockerfile .

echo "Build complete. Image: sentinelvision/inference:latest"
