import os  
import json  
import sqlite3  
from dotenv import load_dotenv  
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential  
  
# Load environment variables from .env file  
load_dotenv()  
  
# Azure OpenAI credentials  
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  
DEPLOYMENT1 = os.getenv("DEPLOYMENT1")  # Deployment without reasoning  
DEPLOYMENT2 = os.getenv("DEPLOYMENT2")  # Deployment with reasoning  
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")  
  
  
# Function to execute and compare SQL queries  
def compare_sql_results(predicted_sql_query, ground_truth_query, db_path='northwind.db'):  
    def _execute_and_format(sql_query):  
        with sqlite3.connect(db_path) as conn:  
            cursor = conn.cursor()  
            try:  
                cursor.execute(sql_query)  
                result = cursor.fetchall()  
                if len(result) != 1:  
                    return None  
                return sorted(map(str, result[0]))  # Sort values as strings  
            except sqlite3.Error:  
                return None  
  
    # Get normalized results  
    pred_result = _execute_and_format(predicted_sql_query)  
    truth_result = _execute_and_format(ground_truth_query)  
  
    # Compare sorted value lists  
    return (pred_result is not None) and (truth_result is not None) and (pred_result == truth_result)  
  
  
# Function to query the OpenAI deployments  
def query_openai_deployment(deployment_name, question, knowledge_graph_context):  
    client = AzureOpenAI(  
        azure_endpoint=AZURE_OPENAI_ENDPOINT,  
        api_key=AZURE_OPENAI_API_KEY,  
        api_version=AZURE_OPENAI_API_VERSION  
    )  
  
    prompt = f"""  
    Use the following database schema and business context to answer the question below:  
  
    {knowledge_graph_context}  
  
    Question: {question}  
  
    Output the SQL query written in SQLITE syntax.  
    """  
  
    response = client.chat.completions.create(  
        model=deployment_name,  
        messages=[  
            {"role": "system", "content": "You are a smart AI assistant with excellent SQL and data analysis skills."},  
            {"role": "user", "content": prompt.strip()}  
        ]  
    )  
  
    # Return the generated SQL query  
    return response.choices[0].message.content  
  
  
# Function to compute accuracy for a deployment  
def compute_accuracy(deployment_name, test_data, knowledge_graph_context, with_reasoning=False):  
    correct_predictions = 0  
    total_predictions = len(test_data)  
  
    for record in test_data:  
        question = record["user"]  
        ground_truth_query = record["sql_result"]  
  
        # For deployment with reasoning, parse the SQL query after reasoning  
        if with_reasoning:  
            assistant_response = query_openai_deployment(deployment_name, question, knowledge_graph_context)  
            if "###final_sql_query:" in assistant_response:  
                predicted_sql_query = assistant_response.split("###final_sql_query:")[-1].strip()  
            else:  
                predicted_sql_query = ""  
        else:  
            # For deployment without reasoning, directly parse the response  
            predicted_sql_query = query_openai_deployment(deployment_name, question, knowledge_graph_context).strip()  
  
        # Compare predicted SQL query with the ground truth  
        if compare_sql_results(predicted_sql_query, ground_truth_query):  
            correct_predictions += 1  
  
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0  
    return accuracy  
  
  
def main():  
    # Load test data  
    with open("sql_result__test_v4.jsonl", "r") as f:  
        test_data = [json.loads(line) for line in f]  
  
    # Load knowledge graph context  
    with open("analytic_graph_v2.json", "r") as f:  
        knowledge_graph = json.load(f)  
  
    knowledge_graph_context = json.dumps(knowledge_graph, indent=4)  
  
    # Compute accuracy for each deployment  
    print("Computing accuracy for Deployment 1 (without reasoning)...")  
    accuracy_deployment1 = compute_accuracy(DEPLOYMENT1, test_data, knowledge_graph_context, with_reasoning=False)  
    print(f"Accuracy for Deployment 1: {accuracy_deployment1:.2%}")  
  
    print("Computing accuracy for Deployment 2 (with reasoning)...")  
    accuracy_deployment2 = compute_accuracy(DEPLOYMENT2, test_data, knowledge_graph_context, with_reasoning=True)  
    print(f"Accuracy for Deployment 2: {accuracy_deployment2:.2%}")  
  
  
if __name__ == "__main__":  
    main()  