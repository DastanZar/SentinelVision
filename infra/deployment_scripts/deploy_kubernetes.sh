#!/bin/bash
set -e

echo "Deploying to Kubernetes..."

kubectl apply -f infra/kubernetes/namespace.yaml
kubectl apply -f infra/kubernetes/pvc.yaml
kubectl apply -f infra/kubernetes/inference-deployment.yaml
kubectl apply -f infra/kubernetes/monitoring-deployment.yaml

echo "Waiting for deployments..."
kubectl wait --for=condition=available --timeout=120s deployment/sentinelvision-inference -n sentinelvision
kubectl wait --for=condition=available --timeout=120s deployment/sentinelvision-monitoring -n sentinelvision

echo "Deployment complete!"
kubectl get pods -n sentinelvision
