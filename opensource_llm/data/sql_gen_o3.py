#!/usr/bin/env python  
  
import os  
import json  
import time  
import sqlite3  
from openai import AzureOpenAI  
from dotenv import load_dotenv  
  
# Load environment variables  
load_dotenv()  
  
# Function to build the user prompt with the knowledge graph context  
def craft_prompt(question, knowledge_graph_context):  
    return f"""Use the following database schema and business context to answer the question below:  
{knowledge_graph_context}  
Question: {question}  

Output only the SQL query inside a ```sql``` code block written in SQLITE syntax. 
End your response with ###END.
 
"""  
  
# Function to extract the SQL query (ignoring any reasoning)  
def process_response(response_content):  
    sql_query = ""  
    if "sql" in response_content:  
        parts = response_content.split("sql")  
        if len(parts) >= 2:  
            # Extract the last code block after "sql"  
            sql_query_block = parts[-1].split("```")  
            if len(sql_query_block) >= 1:  
                sql_query = sql_query_block[0].strip()  
    return sql_query  
  
# Function to send a request via the Azure OpenAI client (o3‑mini)  
def azure_send_request(client, system_message, user_message, model, timeout=200, reasoning_effort="high"):  
    messages = []  
    if system_message:  
        messages.append({"role": "system", "content": system_message})  
    messages.append({"role": "user", "content": user_message})  
    response = client.chat.completions.create(  
        model=model,  
        messages=messages,  
        timeout=timeout,
        # reasoning_effort= reasoning_effort,
        stop= "###END" 
    )  
    return response
  
# Retry helper that wraps the OAI call with a retry mechanism  
def retry_request(client, system_message, user_message, model, retries=3, timeout=90):  
    attempts = 0  
    while attempts < retries:  
        try:  

            response = azure_send_request(client, system_message, user_message, model, timeout)  
            return response  
        except Exception as exc:  
            print(f"Retry {attempts + 1}/{retries} failed: {exc}")  
            attempts += 1  
            time.sleep(10)  # Optional: delay between retries  
    return None  
  
# Execute the generated SQL query to verify it returns at least one row  
def execute_and_verify_sql(sql_query, db_path='northwind.db'):  
    with sqlite3.connect(db_path) as conn:  
        cursor = conn.cursor()  
        try:  
            cursor.execute(sql_query)  
            result = cursor.fetchall()  
            return len(result) > 0  
        except sqlite3.Error as e:  
            print(f"SQL execution failed: {e}\nQuery: {sql_query}")  
            return False  
  
def main():  
    # Load questions and the analytic knowledge graph  
    with open("questions_v4.json", "r") as f:  
        user_questions = json.load(f)  
    with open("analytic_graph_v2.json", "r") as f:  
        knowledge_graph = json.load(f)  
  
    # Convert the knowledge graph into a formatted string  
    knowledge_graph_context = json.dumps(knowledge_graph, indent=4)  
      
    # Prepare the Azure OpenAI client  
    openai_api_key = os.getenv("AZURE_OPENAI_O3_API_KEY")  
    openai_endpoint = os.getenv("AZURE_OPENAI_O3_ENDPOINT")  
    if not openai_api_key or not openai_endpoint:  
        raise Exception("Please set the AZURE_OPENAI_O3_API_KEY and AZURE_OPENAI_O3_ENDPOINT environment variables.")  
    client = AzureOpenAI(  
        api_key=openai_api_key,  
        api_version=os.getenv("AZURE_OPENAI_O3_API_VERSION"),  
        azure_endpoint=openai_endpoint  
    )  
      
    # Specify the deployment (model) name for o3-mini  
    model_deployment = os.environ.get("AZURE_OPENAI_O3_DEPLOYMENT")  
    if not model_deployment:  
        raise Exception("A model deployment name must be provided in the AZURE_OPENAI_O3_DEPLOYMENT environment variable.")  
      
    # Set the system message with the reasoning_effort parameter  
    system_message = (  
        "You are an expert SQL query generator specialized in SQLITE"  
    )  
      
    # Retrieve all questions from the JSON file  
    questions = user_questions["questions"]  
    batch_size = 1  
    question_batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]  
    results = []  
      
    for batch in question_batches:  
        for question in batch:  
            # Create the prompt that includes the knowledge graph context and the question  
            prompt = craft_prompt(question, knowledge_graph_context)  
              
            # Send the request with retry logic  
            response = retry_request(client, system_message, prompt, model_deployment, retries=3, timeout=90)  
            if response:  
                # Extract the SQL query from the response  
                response_content = response.choices[0].message.content  
                sql_query = process_response(response_content)  
                if sql_query:  
                    # Verify the SQL by executing it against the SQLite database  
                    if execute_and_verify_sql(sql_query):  
                        print(f"Question: {question}")  
                        print(f"SQL Query: {sql_query}\n")  
                        results.append({  
                            "user": question,  
                            "sql_result": sql_query  
                        })  
                    else:  
                        print(f"SQL query did not return rows or failed for question '{question}': {sql_query}")  
                else:  
                    print(f"Could not parse a valid SQL query from the response for the question: {question}")  
            else:  
                print(f"Failed to process question '{question}' after retries.")  
      
    # Save results to a JSONL file  
    output_file = "sql_result__test_v5.jsonl"  
    with open(output_file, "w") as f_out:  
        for result in results:  
            f_out.write(json.dumps(result) + "\n")  
      
    print(f"Results saved to {output_file}")  
  
if __name__ == "__main__":  
    main()  