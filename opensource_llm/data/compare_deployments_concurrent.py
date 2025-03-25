#!/usr/bin/env python3  
  
import os  
import json  
import sqlite3  
import re  
import pandas as pd  # for generating the Excel file  
from dotenv import load_dotenv  
from openai import AzureOpenAI  
from azure.core.credentials import AzureKeyCredential  
from concurrent.futures import ThreadPoolExecutor  
#import retry
from tenacity import retry, stop_after_attempt, wait_fixed
  
# Load environment variables from .env file  
load_dotenv()  
  
# Azure OpenAI credentials and deployments  
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  
DEPLOYMENT1 = os.getenv("DEPLOYMENT1")  # Deployment without reasoning  
DEPLOYMENT2 = os.getenv("DEPLOYMENT2")  # Deployment with reasoning  
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")  
SQL_JUDGE_DEPLOYMENT = os.getenv("SQL_JUDGE_DEPLOYMENT")  # SQL judge deployment  
SQL_JUDGE_ENDPOINT = os.getenv("SQL_JUDGE_ENDPOINT")  # SQL judge endpoint
SQL_JUDGE_API_KEY = os.getenv("SQL_JUDGE_API_KEY")  # SQL judge key
SQL_JUDGE_API_VERSION = os.getenv("SQL_JUDGE_API_VERSION")  # SQL judge API version
  
def extract_sql_query(output):
    """Extracts the SQL query from the model's output."""
    sql_blocks = re.findall(r'```sql\s+(.*?)\s+```', output, re.DOTALL)
    if sql_blocks:
        return sql_blocks[-1].strip()  # Use the last SQL block
    return output  # No SQL block found
  
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
    after_order_by = sql[order_by_pos:]  
    if ')' in after_order_by:  
        return False  
    return True  
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))  
def compare_sql_results(predicted_sql, ground_truth_sql, question, db_path='northwind.db'):  
    """Executes both SQL queries, compares results, and returns (passed_bool, explanation_str)."""  
  
    def _normalize_structure(sql_query):  
        try:  
            with sqlite3.connect(db_path) as conn:  
                cursor = conn.cursor()  
                cursor.execute(sql_query)  
                num_columns = len(cursor.description) if cursor.description else 0  
                rows = cursor.fetchall()  
                normalized_rows = [  
                    tuple(sorted(str(item) if item is not None else '' for item in row))  
                    for row in rows  
                ]  
                return {'columns': num_columns, 'rows': len(rows), 'data': normalized_rows}  
        except sqlite3.Error:  
            return None  
  
    pred = _normalize_structure(predicted_sql)  
    truth = _normalize_structure(ground_truth_sql)  
  
    if not pred or not truth:  
        return (False, "One or both queries failed to execute.")  
  
    if pred['data'] == truth['data']:  
        return (True, "Exact match of query results.")  
  
    truth_has_order_by = has_order_by(ground_truth_sql)  
    if not truth_has_order_by:  
        pred_sorted = sorted(pred['data'])  
        truth_sorted = sorted(truth['data'])  
        if pred_sorted == truth_sorted:  
            return (True, "Results match after ignoring row order (no ORDER BY in ground truth).")  
  
    prompt = f"""  
    You are a SQL result evaluator tasked with determining whether a predicted SQL query returns a result set that essentially answers the user's question in the intended way, compared to the ground truth query.  
    Even if the predicted query produces a result with different column names, ordering, aggregation levels, or additional irrelevant columns, if it correctly answers the question, it should be considered correct.  
  
    Question: {question}  
    Ground Truth SQL Query: {ground_truth_sql}  
    Ground Truth Result: Number of Columns: {truth['columns']} Number of Rows: {truth['rows']} Data: {truth['data']}  
    Predicted SQL Query: {predicted_sql}  
    Predicted Result: Number of Columns: {pred['columns']} Number of Rows: {pred['rows']} Data: {pred['data']}  
      
    Please reply with a single line starting with 'True' or 'False', followed by a colon and a brief explanation.  
    Example: 'True: The predicted query correctly answers despite different column names.'  
    """.strip()  
  
    client = AzureOpenAI(  
        azure_endpoint=SQL_JUDGE_ENDPOINT,  
        api_key=SQL_JUDGE_API_KEY,  
        api_version=SQL_JUDGE_API_VERSION  
    )  
    response = client.chat.completions.create(  
        model=SQL_JUDGE_DEPLOYMENT,  
        messages=[  
            {"role": "system", "content": "You evaluate SQL results to determine correctness."},  
            {"role": "user", "content": prompt}  
        ]  
    )  
  
    judge_response = response.choices[0].message.content.strip()  
    if ':' in judge_response:  
        verdict_part, explanation_part = judge_response.split(':', 1)  
    else:  
        verdict_part = judge_response  
        explanation_part = "No explanation provided."  
  
    verdict = verdict_part.strip().lower() == 'true'  
    explanation = explanation_part.strip()  
    return (verdict, explanation)  
  

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))  
def query_openai_deployment(deployment_name, question, knowledge_graph_context):  
    """Uses the given deployment to query the API with the provided question and context."""  
    client = AzureOpenAI(  
        azure_endpoint=AZURE_OPENAI_ENDPOINT,  
        api_key=AZURE_OPENAI_API_KEY,  
        api_version=AZURE_OPENAI_API_VERSION  
    )  
  
    # Extract the date format info from the knowledge graph context (if available)  
    try:  
        date_format = knowledge_graph_context.split('"date_format":')[1].split(',')[0].strip()  
    except IndexError:  
        date_format = "YYYY-MM-DD"  # fallback default  
  
    prompt = f"""  
    Your task is to use the detailed database schema and business context below to write an accurate and efficient SQL query that answers the user's question.  
  
    Before generating the query, follow these steps:  
    Thoroughly review the provided schema and business concepts.  
    Identify any key business metrics or entities mentioned in the question—such as "Gross Profit", "Late Shipment Rate", "Business Account", etc.—and note their descriptions, formulas, and the associated tables.  
    Refer to the table definitions and ensure you use the correct columns and table names.  
    If the business concept is time-dependent (as indicated in the schema), include date-based filtering using the date format "{date_format}".  
    Use the "table_relationships" section to include proper join conditions when your query involves more than one table.  
    Consider additional requirements such as aggregation, grouping (e.g., for metrics like "Regional Freight Distribution" which requires GROUP BY region), and any subqueries specified in the business formulas.  
    Finally, construct a SQL query in valid SQLite syntax that fully addresses the question.  
  
    Database Schema and Business Context: {knowledge_graph_context}  
    Question: {question}  
    Output the SQL query in a sql block written in SQLITE syntax.  
    """  
    response = client.chat.completions.create(  
        model=deployment_name,  
        messages=[  
            {"role": "system", "content": "You are an expert SQL query generator specialized in SQLITE."},  
            {"role": "user", "content": prompt.strip()}  
        ]  
    )  
    return response.choices[0].message.content  
  
def process_record(record, knowledge_graph_context):  
    """Processes one test record and captures explanations."""  
    question = record.get("user", "")  
    ground_truth = record.get("sql_result", "")  
    row_log = {"Question": question, "Ground Truth Query": ground_truth}  
  
    for dep in [DEPLOYMENT1, DEPLOYMENT2]:  
        dep_label = "Deployment1" if dep == DEPLOYMENT1 else "Deployment2" 
        predicted_query = None 
        try:  
            complete_response = query_openai_deployment(dep, question, knowledge_graph_context)  
            if "###final_sql_query:" in complete_response:  
                predicted_query = complete_response.split("###final_sql_query:")[-1].strip()  
            else:  
                predicted_query = extract_sql_query(complete_response)  
  
        except Exception as e:  
            pass_fail = "Fail"  
            comparison_detail = f"Error during evaluation: {str(e)}"  
        if predicted_query:  
            passed, explanation = compare_sql_results(predicted_query, ground_truth, question)  
            pass_fail = "Pass" if passed else "Fail"  
            comparison_detail = explanation  
        else:  
            pass_fail = "Fail"  
            comparison_detail = "No predicted query provided."  

  
        row_log[f"{dep_label} Complete Response"] = complete_response  
        row_log[f"{dep_label} Predicted Query"] = predicted_query  
        row_log[f"{dep_label} Pass/Fail"] = pass_fail  
        row_log[f"{dep_label} Comparison Details"] = comparison_detail  
  
    return row_log  
  
def run_tests_and_log(test_data, knowledge_graph_context, output_file="detailed_query_results_v2.1.xlsx"):  
    """Runs through all test records, queries both deployments concurrently."""  
    log_entries = []  
  
    # Use a ThreadPoolExecutor to process records concurrently  
    with ThreadPoolExecutor() as executor:  
        futures = [executor.submit(process_record, record, knowledge_graph_context) for record in test_data]  
        for future in futures:  
            log_entries.append(future.result())  
  
    # Write detailed logs to an Excel file  
    df = pd.DataFrame(log_entries)  
    df.to_excel(output_file, index=False)  
    print(f"Detailed results logged to {output_file}")  
  
    # Compute aggregated accuracy from the cached results  
    aggregated_results = {  
        DEPLOYMENT1: {"correct": 0, "total": len(test_data)},  
        DEPLOYMENT2: {"correct": 0, "total": len(test_data)}  
    }  
    for entry in log_entries:  
        if entry.get("Deployment1 Pass/Fail") == "Pass":  
            aggregated_results[DEPLOYMENT1]["correct"] += 1  
        if entry.get("Deployment2 Pass/Fail") == "Pass":  
            aggregated_results[DEPLOYMENT2]["correct"] += 1  
  
    return aggregated_results  
  
def main():  
    # Load test data (assumed to be a jsonl file)  
    with open("sql_result_test_v5.jsonl", "r") as f:  
        test_data = [json.loads(line) for line in f]  
  
    # Load knowledge graph context and format it as JSON for the prompt context  
    with open("analytic_graph_v2.json", "r") as f:  
        knowledge_graph = json.load(f)  
        knowledge_graph_context = json.dumps(knowledge_graph, indent=4)  
  
    # Run tests: process each record (query both deployments), log details, and compute accuracy  
    print("Running tests, evaluating responses, and writing detailed results to Excel...")  
    aggregated_results = run_tests_and_log(test_data, knowledge_graph_context)  
  
    # Print aggregated accuracies  
    for deployment_name, counts in aggregated_results.items():  
        reasoning_flag = "(with reasoning)" if deployment_name == DEPLOYMENT2 else "(without reasoning)"  
        accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0  
        print(f"Accuracy for {deployment_name} {reasoning_flag}: {accuracy:.2%}")  
  
if __name__ == "__main__":  
    main()  