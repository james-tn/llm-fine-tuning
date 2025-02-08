import json  
import random  
  
def craft_prompt(question, knowledge_graph_context):  
    """  
    Create a prompt for the AI assistant to generate SQL queries.  
    """  
    return f"""  
    Use the following database schema and business context to answer the question below:  
      
    {knowledge_graph_context}  
      
    Question: {question}  
      
    Output the SQL query in a ```sql``` block written in SQLITE syntax. End your response with ###END.  
    """  
  
def main():  
    # Load input files  
    with open("sql_result_v2.jsonl", "r") as f:  
        input_data = [json.loads(line) for line in f]  
  
    with open("analytic_graph.json", "r") as f:  
        knowledge_graph = json.load(f)  
  
    # Extract knowledge graph context  
    knowledge_graph_context = json.dumps(knowledge_graph, indent=4)  
  
    # Prepare results  
    results = []  
  
    for record in input_data:  
        question = record["user"]  
        reasoning = record["assistant_reasoning"]  
        sql_query = record["sql_result"]  
  
        # Create the prompt  
        prompt = craft_prompt(question, knowledge_graph_context)  
  
        # Construct the formatted output  
        results.append({  
            "messages": [  
                {  
                    "role": "system",  
                    "content": "You are a smart AI assistant with excellent SQL and data analysis skills"  
                },  
                {  
                    "role": "user",  
                    "content": prompt.strip()  
                },  
                {  
                    "role": "assistant",  
                    "content": f"{reasoning.strip()}\n\n{sql_query.strip()}"  
                }  
            ]  
        })  
  
    # Shuffle results to ensure random distribution of training and validation data  
    random.shuffle(results)  
  
    # Split data into training (90%) and validation (10%)  
    split_index = int(len(results) * 0.9)  
    train_data = results[:split_index]  
    val_data = results[split_index:]  
  
    # Save training data to `open_ai_data_train.jsonl`  
    with open("open_ai_data_train.jsonl", "w") as train_file:  
        for result in train_data:  
            train_file.write(json.dumps(result) + "\n")  
  
    # Save validation data to `open_ai_data_val.jsonl`  
    with open("open_ai_data_val.jsonl", "w") as val_file:  
        for result in val_data:  
            val_file.write(json.dumps(result) + "\n")  
  
    print("Results saved to open_ai_data_train.jsonl and open_ai_data_val.jsonl")  
  
if __name__ == "__main__":  
    main()  