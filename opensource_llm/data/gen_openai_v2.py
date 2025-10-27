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
    Output the SQL query written in SQLITE syntax.  
    """  
  
def main(include_reasoning=False):  
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
        reasoning = record.get("assistant_reasoning", "")  
        sql_query = record["sql_result"]  
  
        # Create the prompt  
        prompt = craft_prompt(question, knowledge_graph_context)  
  
        # Construct the formatted output  
        assistant_content = ""  
        if include_reasoning:  
            output_file_name = "open_ai_data_with_reasoning.jsonl"
            assistant_content = f"{reasoning.strip()}\n\n###final_sql_query:\n{sql_query.strip()}"  
        else:  
            output_file_name = "open_ai_data.jsonl"
            assistant_content = sql_query.strip()  
  
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
                    "content": assistant_content  
                }  
            ]  
        })  
  
  
    # Save all data to `open_ai_data.jsonl`
    #   
    with open(output_file_name, "w") as output_file:  
        for result in results:  
            output_file.write(json.dumps(result) + "\n")  
  
    print("Results saved to open_ai_data.jsonl")  
  
if __name__ == "__main__":  
    main(include_reasoning=False)  # Set to True or False based on whether you want reasoning included  