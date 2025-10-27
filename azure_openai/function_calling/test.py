import json  
import openai  
import os  
from pathlib import Path  
import inspect  
from sqlalchemy import create_engine  
from sqlalchemy.orm import sessionmaker  
from datetime import datetime, timedelta  
from dateutil import parser  
  
# Load environment variables  
env_path = Path('.') / 'secrets.env'  
load_dotenv(dotenv_path=env_path)  
  
# Configuration  
emb_engine = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT")  
chat_engine = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")  
client = openai.AzureOpenAI(api_key=os.environ.get("AZURE_OPENAI_API_KEY"),   
                            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),   
                            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"))  
  
sqllite_db_path = os.environ.get("SQLITE_DB_PATH", "data/flight_db.db")  
engine = create_engine(f'sqlite:///{sqllite_db_path}')  
Session = sessionmaker(bind=engine)  
session = Session()  
  
# Define the functions (implementations omitted for brevity)  
def search_airline_knowledgebase(search_query): pass  
def check_flight_status(flight_num, from_): pass  
def query_flights(from_, to, departure_time): pass  
def confirm_flight_change(current_ticket_number, new_flight_num, new_departure_time, new_arrival_time): pass  
def check_change_booking(current_ticket_number, current_flight_num, new_flight_num, from_): pass  
def load_user_flight_info(user_id): pass  
  
# Define the function specifications  
FLIGHT_FUNCTIONS_SPEC = [  
    {"type": "function", "function": {"name": "search_airline_knowledgebase", "parameters": {"type": "object", "properties": {"search_query": {"type": "string"}}, "required": ["search_query"]}}},  
    {"type": "function", "function": {"name": "check_flight_status", "parameters": {"type": "object", "properties": {"flight_num": {"type": "string"}, "from_": {"type": "string"}}, "required": ["flight_num", "from_"]}}},  
    {"type": "function", "function": {"name": "query_flights", "parameters": {"type": "object", "properties": {"from_": {"type": "string"}, "to": {"type": "string"}, "departure_time": {"type": "string"}}, "required": ["from_", "to", "departure_time"]}}},  
    {"type": "function", "function": {"name": "confirm_flight_change", "parameters": {"type": "object", "properties": {"current_ticket_number": {"type": "string"}, "new_flight_num": {"type": "string"}, "new_departure_time": {"type": "string"}, "new_arrival_time": {"type": "string"}}, "required": ["current_ticket_number", "new_flight_num", "new_departure_time", "new_arrival_time"]}}},  
    {"type": "function", "function": {"name": "check_change_booking", "parameters": {"type": "object", "properties": {"current_ticket_number": {"type": "string"}, "current_flight_num": {"type": "string"}, "new_flight_num": {"type": "string"}, "from_": {"type": "string"}}, "required": ["current_ticket_number", "current_flight_num", "new_flight_num", "from_"]}}},  
    {"type": "function", "function": {"name": "load_user_flight_info", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}}  
]  
  
# Map function names to actual functions  
FLIGHT_AVAILABLE_FUNCTIONS = {  
    "search_airline_knowledgebase": search_airline_knowledgebase,  
    "check_flight_status": check_flight_status,  
    "query_flights": query_flights,  
    "confirm_flight_change": confirm_flight_change,  
    "check_change_booking": check_change_booking,  
    "load_user_flight_info": load_user_flight_info  
}  
  
# Utility function to check function arguments  
def check_args(function, args):  
    sig = inspect.signature(function)  
    params = sig.parameters  
    for name in args:  
        if name not in params:  
            return False  
    for name, param in params.items():  
        if param.default is param.empty and name not in args:  
            return False  
    return True  
  
# Load the JSONL data file  
def load_data(file_path):  
    with open(file_path, 'r') as file:  
        return [json.loads(line) for line in file]  
  
# Test the model's tool call accuracy  
def test_model_accuracy(data_file):  
    data = load_data(data_file)  
    total_tests = len(data)  
    correct_tool_calls = 0  
  
    for example in data:  
        conversation = example["messages"]  
        conversation_history = []  
  
        for message in conversation:  
            if message["role"] == "user":  
                user_input = message["content"]  
                assistant = Smart_Agent(FLIGHT_PERSONA, FLIGHT_FUNCTIONS_SPEC, FLIGHT_AVAILABLE_FUNCTIONS)  
                request_help, conversation_history, assistant_response = assistant.run(user_input, conversation_history)  
  
                # Check if the tool call matches the expected tool call in the example  
                for response in conversation_history:  
                    if response["role"] == "assistant" and "tool_calls" in response:  
                        expected_tool_calls = message.get("tool_calls", [])  
                        actual_tool_calls = response["tool_calls"]  
  
                        if expected_tool_calls == actual_tool_calls:  
                            correct_tool_calls += 1  
  
    accuracy_rate = correct_tool_calls / total_tests  
    print(f"Accuracy Rate: {accuracy_rate * 100:.2f}%")  
  
# Run the test  
test_model_accuracy('path/to/your/jsonl/data/file')  
