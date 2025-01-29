import os  
import socket  
import time  
import json  
import subprocess  
import ray  
from azure.storage.blob import BlobServiceClient  
from azure.identity import DefaultAzureCredential  
  
# Constants for Azure Blob Storage  
STORAGE_ACCOUNT_NAME = os.getenv("STORAGE_ACCOUNT_NAME")  # e.g., "mystorageaccount"  
CONTAINER_NAME = "ray-cluster"  
BLOB_NAME = "headnode.json"  
  
# Ray-related constants  
RAY_PORT = 6379  
  
# Global Ray client  
ray_client = None  
  
def get_ip():  
    """Get the private IP address of the current node."""  
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  
    try:  
        s.connect(('10.255.255.255', 1))  
        ip = s.getsockname()[0]  
    except Exception:  
        ip = '127.0.0.1'  
    finally:  
        s.close()  
    return ip  
  
  
def setup_azure_blob():  
    """Set up Azure Blob Storage client and container using AAD and managed identity."""  
    # Use DefaultAzureCredential for AAD authentication  
    credential = DefaultAzureCredential()  
    blob_service_client = BlobServiceClient(  
        account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/",  
        credential=credential  
    )  
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)  
    try:  
        container_client.create_container()  
        print(f"Container '{CONTAINER_NAME}' created or already exists.")  
    except Exception as e:  
        print(f"Container creation failed (possibly already exists): {e}")  
    return container_client  
  
  
def get_blob_content(container_client):  
    """Read the content of the blob if it exists."""  
    try:  
        blob_client = container_client.get_blob_client(BLOB_NAME)  
        if blob_client.exists():  
            blob_data = blob_client.download_blob().readall().decode('utf-8')  
            return json.loads(blob_data)  
    except Exception as e:  
        print(f"Failed to read blob content: {e}")  
    return None  
  
  
def write_blob_content(container_client, content):  
    """Write content to the blob."""  
    try:  
        blob_client = container_client.get_blob_client(BLOB_NAME)  
        blob_client.upload_blob(json.dumps(content), overwrite=True)  
        print("Blob content updated successfully.")  
    except Exception as e:  
        print(f"Failed to write blob content: {e}")  
  
  
def start_ray_head():  
    """Start the Ray head node."""  
    print("Starting Ray head node...")  
    cmd = f"ray start --head --port={RAY_PORT}"  
    subprocess.run(cmd, shell=True, check=False)  
    return get_ip()  
  
  
def start_ray_worker(head_ip):  
    """Start the Ray worker node."""  
    print(f"Starting Ray worker node connecting to head at {head_ip}...")  
    cmd = f"ray start --address={head_ip}:{RAY_PORT}"  
    subprocess.run(cmd, shell=True, check=False)  
  
  
def init():  
    """Initialize the Ray cluster during deployment."""  
    global ray_client  # Declare the global Ray client  
  
    # Get the current node's IP address  
    current_ip = get_ip()  
    print(f"Current node IP: {current_ip}")  
      
    # Set up Azure Blob Storage  
    container_client = setup_azure_blob()  
  
    # Retry mechanism to ensure we can determine the head node  
    max_retries = 20  
    retry_delay = 5  # seconds  
    head_ip = None  
  
    for _ in range(max_retries):  
        # Check the shared note for existing head node info  
        blob_content = get_blob_content(container_client)  
          
        if blob_content and "head_ip" in blob_content:  
            # Head node already exists, connect as a worker  
            head_ip = blob_content["head_ip"]  
            print(f"Head node found: {head_ip}")  
            start_ray_worker(head_ip)  
            break  
        else:  
            # Try to become the head node  
            print("No head node found. Attempting to become the head node...")  
            head_ip = start_ray_head()  
              
            # Write the head node information to the blob  
            write_blob_content(container_client, {"head_ip": head_ip})  
              
            # Verify if we successfully became the head node  
            blob_content = get_blob_content(container_client)  
            if blob_content and blob_content.get("head_ip") == head_ip:  
                print("Successfully became the head node.")  
                break  
          
        # Wait before retrying  
        print("Retrying in 5 seconds...")  
        time.sleep(retry_delay)  
      
    if head_ip is None:  
        print("Failed to determine head node after multiple retries.")  
        raise RuntimeError("Ray cluster setup failed.")  
  
    # Initialize the Ray client globally  
    print(f"Initializing Ray client connected to head node at {head_ip}...")  
    ray.init(address=f"ray://{head_ip}:10001", ignore_reinit_error=True)  
    ray_client = ray  
    print("Ray client initialized successfully.")  
  
  
def run(data):  
    """Run a distributed task on the Ray cluster."""  
    global ray_client  # Use the global Ray client  
    try:  
        if ray_client is None:  
            raise RuntimeError("Ray client is not initialized.")  
          
        print("Connected to Ray cluster. Running distributed task...")  
  
        # Define a simple Ray task  
        @ray.remote  
        def simple_task(x):  
            return x * x  
  
        # Run the task in parallel  
        inputs = [1, 2, 3, 4, 5]  
        results = ray_client.get([simple_task.remote(x) for x in inputs])  
  
        print(f"Ray task results: {results}")  
        return {"results": results}  
    except Exception as e:  
        print(f"Ray cluster execution failed: {e}")  
        return {"error": str(e)}  