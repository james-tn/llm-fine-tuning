import os  
import json  
import time  
from azure.ai.inference import ChatCompletionsClient  
from azure.core.credentials import AzureKeyCredential  
from concurrent.futures import ThreadPoolExecutor, as_completed  
from dotenv import load_dotenv  
  
# Load environment variables  
load_dotenv()  
  
# Function to send a request and return the response  
def send_request(client, payload):  
    response = client.complete(payload)  
    return response  
  
def craft_prompt(question, knowledge_graph_context):  
    return f"""  
    You are an expert SQL query generator specialized in SQLITE.Use the following database schema and business context to answer the question below:  
  
    {knowledge_graph_context}  
  
    Question: {question}  
  
    Output the SQL query in a ```sql``` block written in SQLITE syntax. End your response with ###END. 
    """  
  
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
  
def main():  
    # Load user questions and knowledge graph  
    with open("all_scenarios_questions_v1_unique.json", "r") as f:  
        user_questions = json.load(f)  
  
    with open("analytic_graph.json", "r") as f:  
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
  
    # Extract all questions with metadata  
    all_questions_metadata = []  
    for category, data in user_questions.items():  
        for question_obj in data["questions"]:  
            all_questions_metadata.append({  
                "scenario": category,  
                "difficulty": data.get("difficulty", "unknown"),  # Default to "unknown" if difficulty is missing  
                "question": question_obj  
            })  
  
    # Group questions into batches of up to 20  
    batch_size = 20  
    question_batches = [all_questions_metadata[i:i + batch_size] for i in range(0, len(all_questions_metadata), batch_size)]  
  
    results = []  
  
    # Process each batch sequentially  
    for batch in question_batches:  
        futures = []  
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:  
            for question_metadata in batch:  
                question = question_metadata["question"]  
                scenario = question_metadata["scenario"]  
                difficulty = question_metadata["difficulty"]  
  
                # Create the prompt for the question  
                prompt = craft_prompt(question, knowledge_graph_context)  
  
                # Prepare payload for DeepSeek LLM  
                payload = {  
                    "messages": [{"role": "user", "content": prompt}],  
                    "max_tokens": 2000,  
                    "stop": "###END"  
                }  
  
                # Submit the request to the LLM  
                future = executor.submit(retry_request, client, payload, retries=3)  
                futures.append((question_metadata, future))  
  
            # Collect responses  
            for question_metadata, future in futures:  
                try:  
                    response = future.result()  
                    if response:  # Check if the response is not None (successful)  
                        response_content = response.choices[0].message.content  
                        reasoning, sql_query = process_response(response_content)  
                        if len(sql_query) and len(reasoning) > 0:  
                            #print out question, reasoning and sql_query
                            print(f"Question: {question_metadata['question']}")
                            print(f"Reasoning: {reasoning}")
                            print(f"SQL Query: {sql_query}")
                            #print the count of results
                            print(f"Results: {len(results)}")
                            results.append({  
                                "scenario": question_metadata["scenario"],  
                                "difficulty": question_metadata["difficulty"],  
                                "user": question_metadata["question"],  
                                "assistant_reasoning": reasoning,  
                                "sql_result": sql_query  
                            })  
                    else:  
                        print(f"Failed to process question '{question_metadata['question']}' after retries.")  
                except Exception as exc:  
                    print(f"Final exception for question '{question_metadata['question']}': {exc}")  
  
    # Save results to JSONL file  
    with open("sql_results.jsonl", "w") as f:  
        for result in results:  
            f.write(json.dumps(result) + "\n")  
  
    print("Results saved to sql_results.jsonl")  
  
if __name__ == "__main__":  
    main()  