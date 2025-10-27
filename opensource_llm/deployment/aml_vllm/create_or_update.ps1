<#  
.SYNOPSIS  
A PowerShell script to create or update Azure ML resources.  
#>  
  
# Define variables  
$SubscriptionId = "840b5c5c-3f4a-459a-94fc-6bad2a969f9d"  
$Workspace = "ws02ent"  
$ResourceGroup = "ml"  
$EndpointConfig = "endpoint.yml"  
$DeploymentConfig = "vllm_deployment.yml"  
  
# Set the subscription  
Write-Output "Setting Azure subscription..."  
az account set --subscription $SubscriptionId  
  
# Configure default workspace and resource group  
Write-Output "Configuring defaults..."  
az configure --defaults workspace=$Workspace group=$ResourceGroup  
  
# Function to extract values from a YAML file using Select-String  
function Get-YAMLValue {  
    param (  
        [string]$FilePath,  
        [string]$Key  
    )  
    # Use Select-String to find the line containing the key and extract the value  
    $line = Select-String -Path $FilePath -Pattern "^\s*$Key\s*:\s*(.+)$"  
    if ($line) {  
        return $line.Matches.Groups[1].Value.Trim()  
    } else {  
        Write-Error "Key '$Key' not found in file '$FilePath'."  
        return $null  
    }  
}  
  
# Extract endpoint name from YAML file  
Write-Output "Extracting endpoint details from '$EndpointConfig'..."  
$EndpointName = Get-YAMLValue -FilePath $EndpointConfig -Key "name"  
if (-not $EndpointName) {  
    Write-Error "Unable to extract endpoint name. Exiting script."  
    exit 1  
}  
  
# Check if the endpoint exists  
Write-Output "Checking if endpoint '$EndpointName' exists..."  
$endpointExists = az ml online-endpoint show --name $EndpointName 2>$null  
  
if ($endpointExists) {  
    Write-Output "Endpoint '$EndpointName' already exists. Skipping creation..."  
    # Perform any update logic here (if needed)  
} else {  
    Write-Output "Endpoint '$EndpointName' does not exist. Creating endpoint..."  
    az ml online-endpoint create -f $EndpointConfig  
}  
  
# Extract deployment name from YAML file  
Write-Output "Extracting deployment details from '$DeploymentConfig'..."  
$DeploymentName = Get-YAMLValue -FilePath $DeploymentConfig -Key "name"  
if (-not $DeploymentName) {  
    Write-Error "Unable to extract deployment name. Exiting script."  
    exit 1  
}  
  
# Check if the deployment exists  
Write-Output "Checking if deployment '$DeploymentName' exists..."  
$deploymentExists = az ml online-deployment show --endpoint-name $EndpointName --name $DeploymentName 2>$null  
  
if ($deploymentExists) {  
    Write-Output "Deployment '$DeploymentName' already exists. doing update..."
    az ml online-deployment update -f $DeploymentConfig  
    # Perform any update logic here (if needed)  
} else {  
    Write-Output "Deployment '$DeploymentName' does not exist. Creating deployment..."  
    az ml online-deployment create -f $DeploymentConfig  
}  
  
Write-Output "Script execution completed."  