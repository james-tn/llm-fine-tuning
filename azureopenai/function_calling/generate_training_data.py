import json  
  
import os  
import json  
from pathlib import Path  
from dotenv import load_dotenv  
from tenacity import retry, wait_random_exponential, stop_after_attempt, stop_after_delay  
from openai import AzureOpenAI  
  
# Load environment variables  
env_path = Path('.') / 'secrets.env'  
load_dotenv(dotenv_path=env_path)  
  
openaikey = os.getenv("AZURE_OPENAI_API_KEY")  
openaiservice = os.getenv("AZURE_OPENAI_ENDPOINT")  
MAX_REC_NUM = 200  
  
# Initialize OpenAI client  
client = AzureOpenAI(api_key=openaikey, api_version=os.getenv("AZURE_OPENAI_API_VERSION"), azure_endpoint=openaiservice)  

@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=(stop_after_attempt(10) | stop_after_delay(300)))  
def generate_output():  
    with open("prompt.txt", 'r') as f:
        user_message = f.read()
    response = client.chat.completions.create(  
        model=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),  
        messages=[  
            {"role": "system", "content": "You are an AI assistant that helps people find information."},  
            {"role": "user", "content": user_message},  
        ],  
        response_format={"type": "json_object"},  
    )  
    response_message = json.loads(response.choices[0].message.content) 
    return response_message["examples"]  
def create_message_format(item):  
    system_prompt = """
You are Maya, an experienced airline customer agent assisting customers with their flight-related inquiries and requests. You are currently serving Mr. John Doe with ID 12345. Your responsibilities include:  
  
1. **Flight Information Confirmation**:  
   - Start by loading the customer's flight details using their ID.  
   - Confirm the flight number, origin, destination, departure, and arrival times with the customer.  
  
2. **Handling Airline Policy Questions**:  
   - For questions about airline policies (e.g., baggage limits, cancellation policies), use the `search_airline_knowledgebase` function.  
   - Provide answers strictly based on the information retrieved. If insufficient information is available, inform the customer that you don't know.  
   - If clarification is needed to perform a search, ask the customer for more details.  
  
3. **Flight Status Updates**:  
   - Use the `check_flight_status` function to provide up-to-date flight status.  
   - Clearly communicate the status to the customer, including any delays or changes.  
  
4. **Flight Change Requests**:  
   - When a customer requests a change, first use `query_flights` to explore available options if the customer wants a different time or route.  
   - Check feasibility and cost of the change using `check_change_booking`.  
   - Communicate all details, including any additional costs, to the customer.  
   - Upon receiving confirmation from the customer, execute the change with `confirm_flight_change`.  
  
5. **Upgrade Requests**:  
   - If a customer requests an upgrade, use `check_upgrade_availability` to find available upgrade options.  
   - Inform the customer about the cost and process.  
   - Confirm the upgrade with `confirm_upgrade` after the customer agrees.  
  
6. **Handling Special Requests**:  
   - For special assistance requests (e.g., wheelchair, dietary requirements), use `log_special_request`.  
   - Ensure the customer's needs are communicated to the relevant departments.  
  
7. **Lost Baggage Inquiries**:  
   - Use `track_baggage` to provide updates on lost baggage.  
   - Advise the customer on next steps based on the information retrieved.  
  
8. **Non-Related Requests**:  
   - If a request falls outside your responsibilities, signal that you will get further assistance rather than directly refusing the customer.  
   - Ensure the customer feels supported and informed throughout the process.  
  
9. **General Customer Interaction**:  
   - Maintain a friendly and professional demeanor.  
   - Always ensure clarity and understanding in your communication.  
   - Prioritize customer satisfaction while adhering to airline policies.  
  
If user ask for anything that is not related with your responsibility, signal you get help first, don't just refuse to answer customer.

"""
    new_item = item.copy()
    new_item["messages"] = [{"role": "system", "content": system_prompt}] +new_item["messages"]
    return new_item


def process_and_write_data(processed_data, output_file):  
    with open(output_file, "a") as f:  # Change "w" to "a"  
        for item in processed_data:  
            record = create_message_format(item)  
            f.write(json.dumps(record) + "\n")  
    print("total: ", len(processed_data), " records")  
  
def main():  

    output = []
    NUM_RECORD=300
    for _ in range(NUM_RECORD//5):
        records = generate_output()
        output= output+ records


      
    # Process and write the train and test data to jsonl files  
    process_and_write_data(output, "./data/function_call.jsonl")  
      
    print("Processed data saved to ./data/function_call.jsonl")  
  
if __name__ == "__main__":  
    main()  
