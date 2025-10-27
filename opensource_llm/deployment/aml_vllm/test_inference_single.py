from openai import OpenAI
import time  
import concurrent.futures  
import dotenv
import os
import base64
# Load environment variables from .env file if it exists
dotenv.load_dotenv()
# Define constants for the load test  
BATCHES = 10  # Number of batches  
REQUESTS_PER_BATCH = 1  # Concurrent requests per batch  
COMPUTE_COST_PER_HOUR = 4.78  # Cost per hour in dollars  

# Set OpenAI's API key and API base to use vLLM's API server.
api_key = os.getenv("ENDPOINT_KEY")  # Ensure you have set this in your environment
api_base = os.getenv("ENDPOINT_URL")  # Ensure you have set this in your environment
extra_headers= None
extra_headers={

    "Authorization": "Bearer " + api_key,
    "azureml-model-deployment": "vllm-custom"
}

client = OpenAI(
    api_key="None",
    base_url=api_base,
        # headers=headers

)  
# Function to send a single request  
prompt = "Provide the positions of all payment options and output their coordinates in JSON format. Each entry should include a 'label' (the name of the payment option) and a 'bbox_2d' (the corresponding coordinates). "

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



prompt = "Provide the positions of all payment options and output their coordinates in JSON format. Each entry should include a 'label' (the name of the payment option) and a 'bbox_2d' (the corresponding coordinates). "

# image_path1 = "data/152.png"  # Convert local path to file:// URI
# base64_image1 = encode_image(image_path1)


# image_url1=f"data:image/jpeg;base64,{base64_image1}"

image_path2 = "data/152.png"  # Convert local path to file:// URI
base64_image2 = encode_image(image_path2)


image_url2=f"data:image/jpeg;base64,{base64_image2}"

# Create the structured input format

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            },
                        {
                "type": "image_url",
                "image_url": {
                    "url": image_url2  # Use the file:// URI
                }
            },

        ],
    },
    
]

def send_request():  
    start_time = time.time()  
    outputs =  client.chat.completions.create( model="Qwen/Qwen2.5-VL-7B-Instruct", messages=messages,extra_headers=extra_headers
)
    print(outputs)
    end_time = time.time()  
    return end_time - start_time  # Return the time taken for the request  
  
# Run the load test  
total_requests = BATCHES * REQUESTS_PER_BATCH  
total_time_taken = 0  # Total time taken for all requests  
request_times = []  # Store individual request times  
  
start_load_test = time.time()  # Start time of the load test  
  
for batch in range(BATCHES):  
    print(f"Running batch {batch + 1}/{BATCHES}...")  
    with concurrent.futures.ThreadPoolExecutor(max_workers=REQUESTS_PER_BATCH) as executor:  
        # Start concurrent requests for the current batch  
        futures = [executor.submit(send_request) for _ in range(REQUESTS_PER_BATCH)]  
        # Wait for all requests in the batch to complete  
        for future in concurrent.futures.as_completed(futures):  
            request_time = future.result()  
            request_times.append(request_time)  
            total_time_taken += request_time  
  
end_load_test = time.time()  # End time of the load test  
  
# Calculate metrics  
average_time_per_request = total_time_taken / total_requests  
load_test_duration = end_load_test - start_load_test  # Total duration of the load test  
compute_cost_per_second = COMPUTE_COST_PER_HOUR / 3600  # Cost per second  
compute_cost_per_request = average_time_per_request * compute_cost_per_second * 100  # Cost per request in cents  
  
# Print results  
print(f"Total load test duration: {load_test_duration:.2f} seconds")  
print(f"Average time per request: {average_time_per_request:.2f} seconds")  
print(f"Cost per request: {compute_cost_per_request:.2f} cents")  