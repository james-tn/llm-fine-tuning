<#  
.SYNOPSIS  
A PowerShell script to create or update Azure ML resources.  
#>  
  
# Define variables  
$SubscriptionId = "840b5c5c-3f4a-459a-94fc-6bad2a969f9d"  
$Workspace = "ws01ent"  
$ResourceGroup = "ml"  
$EndpointConfig = "endpoint.yml"  
$DeploymentConfig = "vllm_deployment.yml"  
  
# Set the subscription  
az account set --subscription $SubscriptionId  
  
# Configure workspace and resource group defaults  
az configure --defaults workspace=$Workspace group=$ResourceGroup  
  
# Function to parse YAML files and extract values using PowerShell  
function Get-YAMLValue {  
    param (  
        [string]$FilePath,  
        [string]$Key  
    )  
    $yaml = Get-Content -Path $FilePath -Raw | ConvertFrom-Yaml  
    return $yaml[$Key]  
}  
  
# Extract endpoint name from YAML file  
$EndpointName = Get-YAMLValue -FilePath $EndpointConfig -Key "name"  
  
# Check if the endpoint exists  
$endpointExists = az ml online-endpoint show --name $EndpointName 2>$null  
  
if ($endpointExists) {  
    Write-Output "Endpoint '$EndpointName' already exists. Updating endpoint..."  
    # Perform any update logic here (if needed)  
} else {  
    Write-Output "Endpoint '$EndpointName' does not exist. Creating endpoint..."  
    az ml online-endpoint create -f $EndpointConfig  
}  
  
# Extract deployment name from YAML file  
$DeploymentName = Get-YAMLValue -FilePath $DeploymentConfig -Key "name"  
  
# Check if the deployment exists  
$deploymentExists = az ml online-deployment show --endpoint-name $EndpointName --name $DeploymentName 2>$null  
  
if ($deploymentExists) {  
    Write-Output "Deployment '$DeploymentName' already exists. Updating deployment..."  
    # Perform any update logic here (if needed)  
} else {  
    Write-Output "Deployment '$DeploymentName' does not exist. Creating deployment..."  
    az ml online-deployment create -f $DeploymentConfig  
}  