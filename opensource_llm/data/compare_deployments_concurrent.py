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
  
# Load environment variables from .env file  
load_dotenv()  
  
# Azure OpenAI credentials and deployments  
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  
DEPLOYMENT1 = os.getenv("DEPLOYMENT1")  # Deployment without reasoning  
DEPLOYMENT2 = os.getenv("DEPLOYMENT2")  # Deployment with reasoning  
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")  
SQL_JUDGE_DEPLOYMENT = os.getenv("SQL_JUDGE_DEPLOYMENT")  # SQL judge deployment  
  
def extract_sql_query(output):  
    """Extracts the SQL query from the model's output."""  
    sql_blocks = re.findall(r'sql\s+(.?)\s+```', output, re.DOTALL)  
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
  
def compare_sql_results(predicted_sql, ground_truth_sql, question, db_path='northwind.db'):  
    """Executes the predicted and ground truth SQL queries, normalizes their outcomes, and compares the results."""  
    def _normalize_structure(sql_query):  
        try:  
            with sqlite3.connect(db_path) as conn:  
                cursor = conn.cursor()  
                cursor.execute(sql_query)  
  
                # Get the number of columns and rows  
                num_columns = len(cursor.description) if cursor.description else 0  
                rows = cursor.fetchall()  
                num_rows = len(rows)  
  
                # Normalize rows for comparison: convert items to strings (or None) and sort within each row  
                normalized_rows = []  
                for row in rows:  
                    sorted_values = sorted(  
                        [str(item) if item is not None else None for item in row],  
                        key=lambda x: (x is None, x)  
                    )  
                    normalized_rows.append(tuple(sorted_values))  
                return {'columns': num_columns, 'rows': num_rows, 'data': normalized_rows}  
        except sqlite3.Error:  
            return None  
  
    # Execute both SQL queries  
    pred = _normalize_structure(predicted_sql)  
    truth = _normalize_structure(ground_truth_sql)  
  
    # If either query fails to execute, we cannot reliably compare; return False.  
    if not pred or not truth:  
        return False  
  
    # If the normalized results are exactly equal, return True without invoking the LLM.  
    if pred['data'] == truth['data']:  
        return True  
  
    # Determine if ordering matters based on the ground truth query  
    truth_has_order_by = has_order_by(ground_truth_sql)  
    if not truth_has_order_by:  
        pred_data = sorted(pred['data'])  
        truth_data = sorted(truth['data'])  
    else:  
        pred_data = pred['data']  
        truth_data = truth['data']  
  
    if pred_data == truth_data:  
        return True  
  
    # Otherwise, invoke the SQL judge LLM to decide.  
    prompt = f"""  
    You are a SQL result evaluator tasked with determining whether a predicted SQL query returns a result set that essentially  
    answers the user's question in the intended way, compared to the ground truth query.  
    Even if the predicted query produces a result with different column names or ordering compared to the ground truth,  
    if it correctly answers the question, then it should be considered correct.  
  
    Question: {question}  
    Ground Truth SQL Query: {ground_truth_sql}  
    Ground Truth Result:  
    Number of Columns: {truth['columns']}  
    Number of Rows: {truth['rows']}  
    Data: {truth['data']}  
  
    Predicted SQL Query: {predicted_sql}  
    Predicted Result:  
    Number of Columns: {pred['columns']}  
    Number of Rows: {pred['rows']}  
    Data: {pred['data']}  
  
    Based solely on this information, does the predicted SQL query answer the question correctly?  
    Please reply with a single word: True or False.  
    """.strip()  
  
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
    return True if "true" in judge_output else False  
  
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
    """Processes one test record by querying both deployments."""  
    question = record.get("user", "")  
    ground_truth = record.get("sql_result", "")  
  
    # Start log with the question and ground truth query  
    row_log = {"Question": question, "Ground Truth Query": ground_truth}  
  
    # Process both deployments in one pass  
    for dep in [DEPLOYMENT1, DEPLOYMENT2]:  
        dep_label = "Deployment1" if dep == DEPLOYMENT1 else "Deployment2"  
        is_reasoning = (dep == DEPLOYMENT2)  
  
        try:  
            complete_response = query_openai_deployment(dep, question, knowledge_graph_context)  
            error_msg = ""  
        except Exception as e:  
            complete_response = ""  
            error_msg = f"Error during API call: {str(e)}"  
  
        if is_reasoning:  
            # If using reasoning deployment, look for an explicit marker; if none, try parsing a SQL block.  
            if "###final_sql_query:" in complete_response:  
                predicted_query = complete_response.split("###final_sql_query:")[-1].strip()  
            else:  
                predicted_query = extract_sql_query(complete_response)  
                if not predicted_query:  
                    error_msg = "Cannot parse SQL query"  
        else:  
            predicted_query = complete_response.strip()  
  
        # Compare predicted SQL with ground truth and capture detailed errors  
        try:  
            if predicted_query:  
                passed = compare_sql_results(predicted_query, ground_truth, question)  
                pass_fail = "Pass" if passed else "Fail"  
                comparison_detail = "Predicted SQL matches ground truth sufficiently to answer the question." if passed else "Predicted SQL does not match the ground truth in terms of answering the question."  
            else:  
                pass_fail = "Fail"  
                comparison_detail = "No predicted query provided."  
        except Exception as ex:  
            error_msg += f" Exception during comparison: {str(ex)}"  
            pass_fail = "Fail"  
            comparison_detail = error_msg  
  
        # Save details for this deployment in the log  
        row_log[f"{dep_label} Complete Response"] = complete_response  
        row_log[f"{dep_label} Predicted Query"] = predicted_query  
        row_log[f"{dep_label} Pass/Fail"] = pass_fail  
        row_log[f"{dep_label} Error Message"] = error_msg  
        row_log[f"{dep_label} Comparison Details"] = comparison_detail  
  
    return row_log  
  
def run_tests_and_log(test_data, knowledge_graph_context, output_file="detailed_query_results_v2.xlsx"):  
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