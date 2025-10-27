#!/usr/bin/env python  
"""  
ft_job_and_deploy.py  
  
This script performs the following steps for fine-tuning an Azure OpenAI model:  
- Uploads the training and validation dataset files (JSONL format) using the Azure OpenAI SDK.  
- Creates a fine-tuning job with the uploaded file IDs.  
- Polls the job status (and optionally lists events and checkpoints).  
- Downloads the result file (results.csv) for analysis.  
- Deploys the fine-tuned model via the Azure management (control plane) API.  
  
Make sure the following environment variables are set (for example, via a .env file):  
- AZURE_OPENAI_ENDPOINT (e.g. "https://<your-endpoint>.openai.azure.com/")  
- AZURE_OPENAI_API_KEY (your API key)  
- AZURE_OPENAI_API_VERSION (set to "2024-10-21" or later)  
- AZURE_SUBSCRIPTION_ID (your Azure subscription ID)  
- AZURE_RESOURCE_GROUP (your Azure resource group name)  
- AZURE_OPENAI_RESOURCE_NAME (your Azure OpenAI resource name)  
- AZURE_MANAGEMENT_TOKEN (an access token from “az account get-access-token”)  
  
Usage:  
    python ft_job_and_deploy.py  
"""  
import os  
import time  
import json  
import requests  
from openai import AzureOpenAI  
from dotenv import load_dotenv  
  
# -----------------------------------------------------------------------------  
# Helper functions for file upload and fine-tuning job management  
# -----------------------------------------------------------------------------  
  
def upload_file(client, file_path, purpose="fine-tune"):  
    with open(file_path, "rb") as f:  
        response = client.files.create(file=f, purpose=purpose)  
    return response.id  
  
def create_finetune_job(client, training_file_id, validation_file_id, model, seed=105, hyperparameters=None):  
    if hyperparameters is None:  
        hyperparameters = {"n_epochs": 2}  
    response = client.fine_tuning.jobs.create(  
        training_file=training_file_id,  
        validation_file=validation_file_id,  
        model=model,  
        seed=seed,  
        hyperparameters=hyperparameters  
    )  
    return response  
  
def poll_job_status(client, job_id, poll_interval=30, timeout=3600):  
    start_time = time.time()  
    while True:  
        response = client.fine_tuning.jobs.retrieve(job_id)  
        status = response.status  
        print(f"Job status: {status}")  
        if status in ["succeeded", "failed"]:  
            return response  
        if time.time() - start_time > timeout:  
            raise TimeoutError("Fine-tuning job polling timed out.")  
        time.sleep(poll_interval)  
  
def list_job_events(client, job_id, limit=10):  
    response = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=limit)  
    print("Job events:")  
    print(response.model_dump_json(indent=2))  
  
def list_checkpoints(client, job_id):  
    response = client.fine_tuning.jobs.checkpoints.list(job_id)  
    print("Checkpoints:")  
    print(response.model_dump_json(indent=2))  
    return response  
  
def download_result_file(client, job_response):  
    # If the job succeeded and attached a results file (typically a CSV file)  
    if job_response.status == "succeeded" and job_response.result_files:  
        result_file_id = job_response.result_files[0]  
        file_info = client.files.retrieve(result_file_id)  
        print(f"Downloading result file: {result_file_id} -> {file_info.filename}")  
        content = client.files.content(result_file_id).read()  
        with open(file_info.filename, "wb") as f:  
            f.write(content)  
        print("Result file downloaded.")  
    else:  
        print("No result file available.")  
  
# -----------------------------------------------------------------------------  
# Deployment function using the control plane API  
# -----------------------------------------------------------------------------  
  
def deploy_fine_tuned_model(fine_tuned_model):  
    # Get deployment variables from the environment  
    token = os.getenv("AZURE_MANAGEMENT_TOKEN")  # Generate this token via: az account get-access-token  
    subscription = os.getenv("AZURE_SUBSCRIPTION_ID")  
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")  
    resource_name = os.getenv("AZURE_OPENAI_RESOURCE_NAME")  
    model_deployment_name = "gpt-4o-mini-ft"  # Custom deployment name you choose  
  
    deploy_params = {'api-version': "2024-10-01"}  # Use control plane API version  
    deploy_headers = {  
        'Authorization': f'Bearer {token}',  
        'Content-Type': 'application/json'  
    }  
    deploy_data = {  
        "sku": {"name": "standard", "capacity": 1},  
        "properties": {  
            "model": {  
                "format": "OpenAI",  
                "name": fine_tuned_model,  # e.g., "gpt-4o-mini-0409.ft-<id>"  
                "version": "1"  
            }  
        }  
    }  
    deploy_data_json = json.dumps(deploy_data)  
    request_url = f'https://management.azure.com/subscriptions/{subscription}/resourceGroups/{resource_group}/providers/Microsoft.CognitiveServices/accounts/{resource_name}/deployments/{model_deployment_name}'  
  
    print("Creating a new deployment...")  
    r = requests.put(request_url, params=deploy_params, headers=deploy_headers, data=deploy_data_json)  
    print("Deployment response status code:", r.status_code)  
    try:  
        print("Deployment result:", r.json())  
    except Exception as e:  
        print("Error parsing deployment response:", e)  
  
# -----------------------------------------------------------------------------  
# Main processing function  
# -----------------------------------------------------------------------------  
  
def main():  
    load_dotenv()  # Load environment variables from .env  
  
    # Initialize the AzureOpenAI client using the dataplane API.  
    client = AzureOpenAI(  
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")  # Must be "2024-10-21" or later  
    )  
  
    training_file_name = 'training_set.jsonl'  
    validation_file_name = 'validation_set.jsonl'  
  
    # --- Upload training and validation files  
    print("Uploading training file...")  
    training_file_id = upload_file(client, training_file_name, purpose="fine-tune")  
    print("Training file ID:", training_file_id)  
  
    print("Uploading validation file...")  
    validation_file_id = upload_file(client, validation_file_name, purpose="fine-tune")  
    print("Validation file ID:", validation_file_id)  
  
    # --- Create the fine-tuning job.  
    base_model = "gpt-4o-mini-0409"  # Use your desired base model name.  
    print("Creating fine-tuning job...")  
    ft_job_response = create_finetune_job(client, training_file_id, validation_file_id, base_model, seed=105, hyperparameters={"n_epochs": 2})  
    job_id = ft_job_response.id  
    print("Fine-tuning Job ID:", job_id)  
  
    # --- Poll for job status until it completes.  
    print("Polling fine-tuning job status...")  
    final_job_response = poll_job_status(client, job_id)  
    print("Final job status:", final_job_response.status)  
    print(final_job_response.model_dump_json(indent=2))  
  
    # --- Optional: List fine-tuning events and checkpoints.  
    list_job_events(client, job_id)  
    list_checkpoints(client, job_id)  
  
    # --- Download the result file (e.g., results.csv) if available.  
    download_result_file(client, final_job_response)  
  
    # --- Deploy the fine-tuned model if the job succeeded.  
    if final_job_response.status == "succeeded":  
        fine_tuned_model = final_job_response.fine_tuned_model  # e.g., "gpt-4o-mini-0409.ft-<id>"  
        print("Fine-tuned model:", fine_tuned_model)  
        deploy_fine_tuned_model(fine_tuned_model)  
    else:  
        print("Fine-tuning job did not succeed. No deployment will be performed.")  
  
if __name__ == "__main__":  
    main()  