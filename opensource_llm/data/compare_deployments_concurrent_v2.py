import os  
import json  
import sqlite3  
from dotenv import load_dotenv  
from openai import AzureOpenAI  
from azure.core.credentials import AzureKeyCredential  
from concurrent.futures import ThreadPoolExecutor, as_completed  
import re  
  
# Load environment variables from .env file  
load_dotenv()  
  
# Azure OpenAI credentials  
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  
DEPLOYMENT1 = os.getenv("DEPLOYMENT1")  # Deployment without reasoning  
DEPLOYMENT2 = os.getenv("DEPLOYMENT2")  # Deployment with reasoning  
DEPLOYMENT3 = os.getenv("DEPLOYMENT3")  # Third deployment  
DEPLOYMENT4 = os.getenv("DEPLOYMENT4")  # Fourth deployment  
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")  
SQL_JUDGE_DEPLOYMENT = os.getenv("SQL_JUDGE_DEPLOYMENT")  # SQL judge deployment
  
# Function to execute and compare SQL queries  
def has_order_by(sql):  
    # Remove comments  
    sql = re.sub(r'--.*', '', sql, flags=re.MULTILINE)  # Remove line comments  
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)  # Remove block comments  
  
    # Normalize whitespace and case  
    sql = re.sub(r'\s+', ' ', sql).upper().strip()  
  
    # Check for ORDER BY clause not within parentheses  
    order_by_pos = sql.rfind(' ORDER BY ')  
    if order_by_pos == -1:  
        return False  
  
    # Check if there's a closing parenthesis after ORDER BY  
    after_order_by = sql[order_by_pos:]  
    if ')' in after_order_by:  
        return False  
  
    return True  
  
def compare_sql_results(predicted_sql, ground_truth_sql, db_path='northwind.db'):  
    def _normalize_structure(sql_query):  
        try:  
            with sqlite3.connect(db_path) as conn:  
                cursor = conn.cursor()  
                cursor.execute(sql_query)  
                  
                # Get the number of columns and rows  
                num_columns = len(cursor.description) if cursor.description else 0  
                rows = cursor.fetchall()  
                num_rows = len(rows)  
                  
                # Normalize rows for comparison  
                normalized_rows = []  
                for row in rows:  
                    # Convert each item to str (or None) and sort values within the row for non-ordered comparison  
                    sorted_values = sorted(  
                        [str(item) if item is not None else None for item in row],  
                        key=lambda x: (x is None, x)  
                    )  
                    normalized_rows.append(tuple(sorted_values))  
                  
                return {  
                    'columns': num_columns,  
                    'rows': num_rows,  
                    'data': normalized_rows  
                }  
        except sqlite3.Error:  
            return None  
  
    # Execute both SQL queries  
    pred = _normalize_structure(predicted_sql)  
    truth = _normalize_structure(ground_truth_sql)  
  
    # If either query fails, or if the basic structure mismatches, return False immediately  
    if not pred or not truth:  
        return False  
    if pred['columns'] != truth['columns'] or pred['rows'] != truth['rows']:  
        return False  
  
    # Determine if ordering matters based on the ground truth query  
    truth_has_order_by = has_order_by(ground_truth_sql)  
    if not truth_has_order_by:  
        pred_data = sorted(pred['data'])  
        truth_data = sorted(truth['data'])  
    else:  
        pred_data = pred['data']  
        truth_data = truth['data']  
  
    # If the normalized results are exactly equal, we can return True without invoking the LLM  
    if pred_data == truth_data:  
        return True  
  
    # Otherwise, invoke the SQL judge LLM to decide  
    prompt = f"""  
        You are a SQL result evaluator tasked with determining whether a predicted SQL query produces an equivalent result set to the ground truth SQL query when executed on the same database.  
        Below is the provided information:  
  
        Ground Truth SQL Query:  
        {ground_truth_sql}  
  
        Ground Truth Result:  
        Number of Columns: {truth['columns']}  
        Number of Rows: {truth['rows']}  
        Data: {truth['data']}  
  
        Predicted SQL Query:  
        {predicted_sql}  
  
        Predicted Result:  
        Number of Columns: {pred['columns']}  
        Number of Rows: {pred['rows']}  
        Data: {pred['data']}  
  
        Note: If no explicit ORDER BY clause is present, row order may vary, so consider only the content and structure.  
  
        Based solely on this information, does the predicted SQL query return the same result as the ground truth query?   
        Please answer with a single word: True or False.  
    """.strip()  
  
    # Initialize the Azure OpenAI client and use the SQL judge deployment  
    client = AzureOpenAI(  
        azure_endpoint=AZURE_OPENAI_ENDPOINT,  
        api_key=AZURE_OPENAI_API_KEY,  
        api_version=AZURE_OPENAI_API_VERSION  
    )  
    response = client.chat.completions.create(  
        model=SQL_JUDGE_DEPLOYMENT,  
        messages=[  
            {"role": "system", "content": "You are an expert SQL result judge."},  
            {"role": "user", "content": prompt}  
        ]  
    )  
    judge_output = response.choices[0].message.content.strip().lower()  
  
    # Expect the LLM response to be "true" or "false"  
    return True if judge_output in ["true"] or "true" in judge_output else False  
  
# Function to query the OpenAI deployments  
def query_openai_deployment(deployment_name, question, knowledge_graph_context):  
    """Uses the given deployment to query the API with the provided question and context. Returns the API’s full response."""  
    client = AzureOpenAI(  
        azure_endpoint=AZURE_OPENAI_ENDPOINT,  
        api_key=AZURE_OPENAI_API_KEY,  
        api_version=AZURE_OPENAI_API_VERSION  
    )  
    prompt= f"""Your task is to use the detailed database schema and business context below to write an accurate and efficient SQL query that answers the user's question.  
      
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
    response = client.chat.completions.create(  
        model=deployment_name,  
        messages=[  
            {"role": "system", "content": "You are an expert SQL query generator specialized in SQLITE."},  
            {"role": "user", "content": prompt.strip()}  
        ]  
    )  

    # Extract the SQL query from the response  
    output = response.choices[0].message.content.strip()  
    sql_query = extract_sql_query(output)  
  
    return sql_query  
  
# Helper function to parse the SQL query from the LLM output  
def extract_sql_query(output):  
    """  
    Extracts the SQL query from the model's output. If the output contains  
    ` ```sql ` blocks, return the content of the last block. Otherwise, return the entire response.  
    """  
    sql_blocks = re.findall(r'```sql\s+(.*?)\s+```', output, re.DOTALL)  
    if sql_blocks:  
        return sql_blocks[-1].strip()  # Use the last SQL block  
    return output  # If no SQL block exists, return the entire response  
  
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
def compute_accuracies_concurrently(test_data, knowledge_graph_context):  
    results = {}  
    with ThreadPoolExecutor() as executor:  
        future_to_deployment = {  
            executor.submit(  
                compute_accuracy,  
                deployment_name,  
                test_data,  
                knowledge_graph_context,  
                with_reasoning=(deployment_name == DEPLOYMENT2)  
            ): deployment_name  
            for deployment_name in [DEPLOYMENT1, DEPLOYMENT2, DEPLOYMENT3, DEPLOYMENT4]  
        }  
  
        for future in as_completed(future_to_deployment):  
            deployment_name = future_to_deployment[future]  
            try:  
                results[deployment_name] = future.result()  
            except Exception as e:  
                print(f"Error occurred while computing accuracy for {deployment_name}: {e}")  
    return results  

# Main function  
def main():  
    # Load test data  
    with open("sql_result_test_v5.jsonl", "r") as f:  
        test_data = [json.loads(line) for line in f]  
  
    # Load knowledge graph context  
    with open("analytic_graph_v2.json", "r") as f:  
        knowledge_graph = json.load(f)  
  
    knowledge_graph_context = json.dumps(knowledge_graph, indent=4)  
  
    # Compute accuracies concurrently  
    print("Computing accuracies for deployments concurrently...")  
    accuracies = compute_accuracies_concurrently(test_data, knowledge_graph_context)  
  
    # Print results  
    for deployment_name, accuracy in accuracies.items():  
        print(f"Accuracy for {deployment_name}: {accuracy:.2%}")  
  
  
if __name__ == "__main__":  
    main()  
