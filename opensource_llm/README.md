# Fine-Tuning Open-Source LLMs with Azure Machine Learning  
  
This section of the repository focuses on fine-tuning **open-source large language models (LLMs)** at scale using **Azure Machine Learning (Azure ML)**. The examples and best practices here are designed to handle the challenges of fine-tuning multi-billion-parameter models, including infrastructure setup, distributed training, hyperparameter optimization, and deployment.   
  
## Overview of Contents  
  
### 1. [Single Step](single_step)  
The **Single Step** folder demonstrates the quick setup of scalable **Distributed Data Parallel (DDP)** and **Model Parallel** fine-tuning workflows. This is ideal for users who want to quickly get started with fine-tuning large models across multiple GPUs while leveraging Azure ML's infrastructure.  
  
### 2. [Pipelines](pipelines)  
The **Pipelines** folder provides examples of setting up **end-to-end workflows** for model fine-tuning. This includes:  
  
- Dataset preparation and preprocessing.  
- Model training with distributed strategies.  
- Hyperparameter optimization (HPO).  
- Model testing and evaluation.  
  
These pipelines are built for scalability and efficiency, enabling users to automate the training process for large-scale models.  
  
### 3. [Deployment](deployment)  
The **Deployment** folder focuses on deploying trained models using optimal techniques in Azure ML. This includes:  
  
- Leveraging Azure ML endpoints for serving LLMs.  
- Optimizing inference performance for cost and latency.  
- Example setups for deployment on various GPU-backed compute instances.  
  
---  
  
## Prerequisites and Setup  
  
### 1. Install Azure ML CLI  
To use the examples provided in this section, you need to have the **Azure ML CLI** installed. Follow these steps:  
  
1. **Azure CLI Installed**: Ensure that the Azure CLI is installed and configured on your machine.    
   [Install Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)  
  
2. **Azure ML CLI Extension Installed**: Install the Azure Machine Learning CLI extension.    
   ```bash  
   az extension add -n ml -y  
3. **Workspace Configuration**: Ensure you are logged into your Azure subscription and have set the active Azure ML workspace.
   ```bash
   az account set --subscription <your-subscription-id>  
   az configure --defaults group=<your-resource-group> workspace=<your-workspace>  
### 2. Azure ML Compute Quota  
Fine-tuning large models requires GPU-backed compute instances. Ensure that your Azure subscription has the necessary quota for the following compute instance types:  
  
#### Supported Compute Series  
- **NCv4 (A100 GPUs)**: Designed for high-performance training with A100 GPUs.  
- **NCv5 (H100 GPUs)**: Next-generation GPU compute optimized for heavy workloads.  
- **NDv5 (H100 GPUs with NVLink)**: Ideal for distributed training and workloads requiring high inter-GPU communication.  
  
#### Quota Requirements  
Check your quota for these VM series using the Azure CLI:  
```bash  
az vm list-usage --location <region> --output table  
```
If you do not have enough quota, request an increase through the Azure portal:  
  
1. Navigate to **Help + Support** > **New Support Request**.  
2. Choose the appropriate subscription and request a quota increase for the desired GPU series.  