import os  
import json  
import time  
import sqlite3  
from azure.ai.inference import ChatCompletionsClient  
from azure.core.credentials import AzureKeyCredential  
from dotenv import load_dotenv  
  
# Load environment variables  
load_dotenv()  
  
# Function to send a request and return the response  
def send_request(client, payload):  
    response = client.complete(payload)  
    return response  
  
def craft_prompt(question, knowledge_graph_context):  
    return f"""You are an expert SQL query generator specialized in SQLITE. Your task is to use the detailed database schema and business context below to write an accurate and efficient SQL query that answers the user's question.  
      
    Before generating the query, follow these steps:  
    - Thoroughly review the provided schema and business concepts.  
    - Identify any key business metrics or entities mentioned in the question—such as "Gross Profit", "Late Shipment Rate", "Business Account", etc.—and note their descriptions, formulas, and the associated tables.  
    - Refer to the table definitions and ensure you use the correct columns and table names.  
    - If the business concept is time-dependent (as indicated in the schema), include date-based filtering using the date format "{knowledge_graph_context.split('"date_format":')[1].split(',')[0].strip()}" (i.e. YYYY-MM-DD).  
    - Use the "table_relationships" section to include proper join conditions when your query involves more than one table.  
    - Consider additional requirements such as aggregation, grouping (e.g., for metrics like "Regional Freight Distribution" which requires GROUP BY region), and any subqueries specified in the business formulas.  
      
    Finally, construct a SQL query in valid SQLite syntax that fully addresses the question.  
      
    Database Schema and Business Context:  
    {knowledge_graph_context}  
      
    Question: {question}  
      
    Output the SQL query in a sql block written in SQLITE syntax. End your response with ###END."""    
def process_response(response_content):  
    # Extract reasoning and SQL query from the response  
    reasoning, sql_query = "", ""  
    if "```sql" in response_content:  
        parts = response_content.split("```sql")  
        if len(parts) == 2:  
            reasoning = parts[0].strip()  # Everything before the ```sql block  
            sql_query_block = parts[1].split("```")  # Split by the closing ``` block  
            if len(sql_query_block) >= 1:  
                sql_query = sql_query_block[0].strip()  
    return reasoning, sql_query  
  
def retry_request(client, payload, retries=3):  
    """Retry the request to the LLM up to `retries` times."""  
    attempts = 0  
    while attempts < retries:  
        try:  
            response = send_request(client, payload)  
            return response  
        except Exception as exc:  
            print(f"Retry {attempts + 1}/{retries} failed: {exc}")  
            attempts += 1  
            time.sleep(10)  # Optional: Add a small delay between retries  
    return None  
  
def execute_and_verify_sql(sql_query, db_path='northwind.db'):  
    """  
    Execute the generated SQL query and verify if it returns at least one row.  
    """  
    with sqlite3.connect(db_path) as conn:  
        cursor = conn.cursor()  
        try:  
            cursor.execute(sql_query)  
            result = cursor.fetchall()  
            # Return True if at least one row is fetched  
            return len(result) > 0  
        except sqlite3.Error as e:  
            print(f"SQL execution failed: {e}")  
            return False  
  
def main():  
    # Load user questions and knowledge graph  
    with open("questions_v4_deduped.json", "r") as f:  # Updated file name to remove "by_difficulty"  
        user_questions = json.load(f)  
  
    with open("analytic_graph_v2.json", "r") as f:  
        knowledge_graph = json.load(f)  
  
    # Extract knowledge graph context  
    knowledge_graph_context = json.dumps(knowledge_graph, indent=4)  
  
    # Prepare API client  
    api_key = os.getenv("AZURE_INFERENCE_CREDENTIAL")  
    if not api_key:  
        raise Exception("A key should be provided to invoke the endpoint")  
    client = ChatCompletionsClient(  
        endpoint=os.getenv("AZURE_INFERENCE_ENDPOINT"),  
        credential=AzureKeyCredential(api_key),  
    )  
  
    # Extract all questions  
    questions = user_questions["questions"]  
  
    # Group questions into batches of up to batch_size  
    batch_size = 1  
    question_batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]  
  
    results = []  
  
    # Process each batch sequentially  
    for batch in question_batches:  
        for question in batch:  
            # Create the prompt for the question  
            prompt = craft_prompt(question, knowledge_graph_context)  
  
            # Prepare payload for DeepSeek LLM  
            payload = {  
                "messages": [{"role": "user", "content": prompt}],  
                "max_tokens": 4000,  
                "stop": "###END"  
            }  
  
            # Submit the request to the LLM  
            response = retry_request(client, payload, retries=3)  
  
            if response:  # Check if the response is not None (successful)  
                response_content = response.choices[0].message.content  
                reasoning, sql_query = process_response(response_content)  
                if len(sql_query) > 0 and len(reasoning) > 0:  
                    # Verify the SQL query  
                    if execute_and_verify_sql(sql_query):  
                        # Print out question, reasoning, and SQL query  
                        print(f"Question: {question}")  
                        print(f"Reasoning: {reasoning}")  
                        print(f"SQL Query: {sql_query}")  
                        print(f"Results: {len(results)}")  
                        results.append({  
                            "user": question,  
                            "assistant_reasoning": reasoning,  
                            "sql_result": sql_query  
                        })  
                    else:  
                        print(f"SQL query did not return any rows or failed: {sql_query}")  
            else:  
                print(f"Failed to process question '{question}' after retries.")  
  
    # Save results to JSONL file  
    with open("sql_result__test_v4.jsonl", "w") as f:  
        for result in results:  
            f.write(json.dumps(result) + "\n")  
  
    print("Results saved to sql_result__test_v4.jsonl")  
  
if __name__ == "__main__":  
    main()  