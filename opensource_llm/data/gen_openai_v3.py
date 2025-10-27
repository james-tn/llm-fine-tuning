import json  
import random  
  
def craft_prompt(question, knowledge_graph_context):  
    return f"""Your task is to use the detailed database schema and business context below to write an accurate and efficient SQL query that answers the user's question.  
      
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
  
    Output the SQL query in a sql block written in SQLITE syntax."""  
  
def main(include_reasoning=False):  
    # Load input files  
    with open("sql_result_train_v5.jsonl", "r") as f:  
        input_data = [json.loads(line) for line in f]  
  
    with open("analytic_graph_v2.json", "r") as f:  
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
            output_file_name = "open_ai_data_with_reasoning_v2.jsonl"
            assistant_content = f"{reasoning.strip()}\n\n###final_sql_query:\n{sql_query.strip()}"  
        else:  
            output_file_name = "open_ai_data_v2.jsonl"
            assistant_content = sql_query.strip()  
  
        results.append({  
            "messages": [  
                {  
                    "role": "system",  
                    "content": "You are an expert SQL query generator specialized in SQLITE."  
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
  
    print("Results saved to open_ai_data_v2.jsonl")  
  
if __name__ == "__main__":  
    main(include_reasoning=True)  # Set to True or False based on whether you want reasoning included  