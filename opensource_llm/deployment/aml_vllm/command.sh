#!/bin/bash  
  
# Ensure the script exits on errors  
set -e  
  
# Set the subscription  
SUBSCRIPTION_ID="840b5c5c-3f4a-459a-94fc-6bad2a969f9d"  
az account set --subscription "$SUBSCRIPTION_ID"  
  
# Configure default workspace and resource group  
WORKSPACE="ws01ent"  
RESOURCE_GROUP="ml"  
az configure --defaults workspace="${WORKSPACE}" group="${RESOURCE_GROUP}"  
  
# Check if the online endpoint exists  
ENDPOINT_NAME=$(yq e '.name' endpoint.yml) # Extract the endpoint name from endpoint.yml  
if az ml online-endpoint show --name "$ENDPOINT_NAME" 2>/dev/null; then  
  echo "Endpoint '${ENDPOINT_NAME}' already exists. Updating endpoint..."  
else  
  echo "Endpoint '${ENDPOINT_NAME}' does not exist. Creating endpoint..."  
  az ml online-endpoint create -f endpoint.yml  
fi  
  
# Check if the deployment exists  
DEPLOYMENT_NAME=$(yq e '.name' vllm_deployment.yml) # Extract the deployment name from vllm_deployment.yml  
if az ml online-deployment show --endpoint-name "$ENDPOINT_NAME" --name "$DEPLOYMENT_NAME" 2>/dev/null; then  
  echo "Deployment '${DEPLOYMENT_NAME}' already exists. Updating deployment..."  
else  
  echo "Deployment '${DEPLOYMENT_NAME}' does not exist. Creating deployment..."  
  az ml online-deployment create -f vllm_deployment.yml  
fi  