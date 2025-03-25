#!/usr/bin/env python  
"""  
data_gen_program.py  
  
This program consolidates the following steps in one pipeline:  
1. Generate unique business questions based on an ontology.  
2. Generate a SQL query for each question using a model chosen from:  
   • o3 (using AzureOpenAI) or  
   • deepseek (using ChatCompletionsClient).  
   Both options now use the same detailed prompt (from deepseek) so they produce a similar “user” message.  
3. Wrap each question/SQL pair into an OpenAI-style conversation record using the system prompt:  
   "You are an expert SQL query generator specialized in SQLITE."  
4. Split the records into 80% training, 10% testing, and 10% inference JSONL files.  
  
Usage example:  
python data_gen_program.py --num_questions 50 --sql_model deepseek --include_reasoning  
(omit --include_reasoning for deepseek output without reasoning; use --sql_model o3 to use the o3 option)  
  
Make sure to set up your .env file with the required Azure keys and endpoints.  
"""  
  
import os  
import sys  
import json  
import time  
import random  
import argparse  
import sqlite3  
from difflib import SequenceMatcher  
from dotenv import load_dotenv  
from tenacity import retry, wait_random_exponential, stop_after_attempt, stop_after_delay  
from openai import AzureOpenAI  
from azure.ai.inference import ChatCompletionsClient  
from azure.core.credentials import AzureKeyCredential  
  
# ----------------------------------------------------------------------------- Utility functions for file operations  
  
def load_json(filepath):  
    try:  
        with open(filepath, "r") as f:  
            return json.load(f)  
    except Exception as exc:  
        print(f"Error loading JSON from {filepath}: {exc}")  
        sys.exit(1)  
  
def save_jsonl(records, filepath):  
    try:  
        with open(filepath, "w") as f:  
            for rec in records:  
                f.write(json.dumps(rec) + "\n")  
        print(f"Saved {len(records)} records to {filepath}")  
    except Exception as exc:  
        print(f"Error saving {filepath}: {exc}")  
  
# ----------------------------------------------------------------------------- Deduplication functions  
  
def is_similar(q1, q2, threshold=0.9):  
    return SequenceMatcher(None, q1, q2).ratio() >= threshold  
  
def deduplicate_questions(questions, threshold=0.9):  
    unique = []  
    for q in questions:  
        if all(not is_similar(q, ex, threshold) for ex in unique):  
            unique.append(q)  
    return unique  
  
# ----------------------------------------------------------------------------- STEP 1 – Business Question Generation  
  
@retry(wait=wait_random_exponential(multiplier=1, max=60),  
       stop=(stop_after_attempt(20) | stop_after_delay(300)))  
def generate_questions_from_ontology(ontology_str, question_client):  
    prompt = f"""Using the provided analytics graph, create at least 20 distinct business questions that can each be answered with a single SQL query.  
  
Analytics Graph:  
{ontology_str}  
  
Instructions:  
Develop a wide array of business questions ranging from basic to advanced complexity.  
Ensure that each question can be resolved using quantitative data.  
Focus on questions that are practical and applicable to real-world business scenarios.  
Frame the questions so that they have a clear, singular answer.  
Base all questions on data from the years 2020 to 2024.  
Do not include numbering, formulas, or SQL queries in the questions.  
  
Output Format:  
Output the questions in the following JSON format:  
{{"questions": ["question1", "question2", ...]}}  
"""  
    messages = [  
        {"role": "system", "content": "You are a smart AI assistant skilled in generating diverse business questions."},  
        {"role": "user", "content": prompt}  
    ]  
    response = question_client.chat.completions.create(  
        model=os.environ.get("AZURE_OPENAI_GPT4_DEPLOYMENT"),  
        messages=messages,  
        response_format={"type": "json_object"},  
        timeout=90,  
    )  
    response_content = response.choices[0].message.content  
    try:  
        resp_obj = json.loads(response_content)  
    except Exception as exc:  
        raise Exception(f"Response parsing error: {exc}\nContent received: {response_content}")  
    if "questions" not in resp_obj:  
        raise Exception("Response JSON does not contain 'questions'.")  
    return resp_obj  
  
def generate_question_list(ontology_data, num_required, question_client):  
    ontology_str = json.dumps(ontology_data, indent=4)  
    final_qs = []  
    while len(final_qs) < num_required:  
        resp = generate_questions_from_ontology(ontology_str, question_client)  
        new_qs = resp.get("questions", [])  
        final_qs.extend(new_qs)  
        final_qs = deduplicate_questions(final_qs, threshold=0.9)  
        print(f"Collected {len(final_qs)} unique questions so far...")  
        if len(final_qs) < num_required:  
            time.sleep(5)  
    return final_qs[:num_required]  
  
# ----------------------------------------------------------------------------- Helper: Build the SQL-generation prompt (same for both models)  
  
def craft_prompt(question, knowledge_graph_context):  
    try:  
        date_format = knowledge_graph_context.split('"date_format":')[1].split(',')[0].strip()  
    except Exception:  
        date_format = "YYYY-MM-DD"  
    return f"""Your task is to use the detailed database schema and business context below to write an accurate and efficient SQL query that answers the user's question.  
  
Before generating the query, follow these steps:  
1. Thoroughly review the provided schema and business concepts.  
2. Identify any key business metrics or entities mentioned in the question—such as "Gross Profit", "Late Shipment Rate", "Business Account", etc.—and note their descriptions, formulas, and the associated tables.  
3. Refer to the table definitions and ensure you use the correct columns and table names.  
4. If the business concept is time-dependent (as indicated in the schema), include date-based filtering using the date format {date_format} (i.e. {date_format}).  
5. Use the "table_relationships" section to include proper join conditions when your query involves more than one table.  
6. Consider additional requirements such as aggregation, grouping, and any subqueries specified in the business formulas.  
7. Finally, construct a SQL query in valid SQLite syntax that fully addresses the question.  
  
Database Schema and Business Context:  
{knowledge_graph_context}  
  
Question:  
{question}  
  
Output the SQL query in a sql block written in SQLITE syntax. End your response with ###END."""  
  
# ----------------------------------------------------------------------------- Helper: Process the LLM response to extract reasoning and SQL query  
  
def process_sql_response(response_content):  
    reasoning = ""  
    sql_query = ""  
    if "```sql" in response_content:  
        parts = response_content.split("```sql")  
        if len(parts) >= 2:  
            reasoning = parts[0].strip()  
            sql_parts = parts[1].split("```")  
            if sql_parts and len(sql_parts) >= 1:  
                sql_query = sql_parts[0].strip()  
            else:  
                sql_query = response_content.strip()  
    return reasoning, sql_query  
  
# ----------------------------------------------------------------------------- Helper: Verify the generated SQL query returns data against an SQLite DB  
  
def execute_and_verify_sql(sql_query, db_path='northwind.db'):  
    try:  
        with sqlite3.connect(db_path) as conn:  
            cursor = conn.cursor()  
            cursor.execute(sql_query)  
            results = cursor.fetchall()  
            return len(results) > 0  
    except sqlite3.Error as exc:  
        print(f"SQL execution failed: {exc}\nQuery: {sql_query}")  
        return False  
  
# ----------------------------------------------------------------------------- STEP 2 – SQL Generation  
  
def generate_sql_o3(question, knowledge_graph_context, o3_client, model_deployment, include_reasoning=False):  
    prompt = craft_prompt(question, knowledge_graph_context)  
    system_message = "You are an expert SQL query generator specialized in SQLITE."  
    messages = [{"role": "system", "content": system_message},  
                {"role": "user", "content": prompt.strip()}]  
    try:  
        response = o3_client.chat.completions.create(  
            model=model_deployment,  
            messages=messages,  
            timeout=200,  
            stop="###END"  
        )  
        response_content = response.choices[0].message.content  
        reasoning, sql_query = process_sql_response(response_content)  
        if sql_query and execute_and_verify_sql(sql_query):  
            if include_reasoning:  
                assistant_content = f"{reasoning}\n\n###final_sql_query:\n{sql_query}"  
            else:  
                assistant_content = sql_query  
            record = {  
                "messages": [  
                    {"role": "system", "content": system_message},  
                    {"role": "user", "content": prompt.strip()},  
                    {"role": "assistant", "content": assistant_content}  
                ]  
            }  
            return record  
        else:  
            print(f"o3 model did not produce a valid SQL for question: {question}")  
            return None  
    except Exception as exc:  
        print(f"Error in o3 processing question '{question}': {exc}")  
        return None  
  
def generate_sql_deepseek(question, knowledge_graph_context, deepseek_client, include_reasoning):  
    prompt = craft_prompt(question, knowledge_graph_context)  
    system_message = "You are an expert SQL query generator specialized in SQLITE."  
    payload = {  
        "messages": [{"role": "user", "content": prompt.strip()}],  
        "max_tokens": 4000,  
        "stop": "###END"  
    }  
    attempts = 0  
    response = None  
    while attempts < 3:  
        try:  
            response = deepseek_client.complete(payload)  
            break  
        except Exception as exc:  
            attempts += 1  
            print(f"Deepseek request attempt {attempts} failed for question '{question}': {exc}")  
            time.sleep(10)  
    if response is None:  
        print(f"Deepseek failed to process question '{question}' after 3 attempts.")  
        return None  
  
    response_content = response.choices[0].message.content  
    reasoning, sql_query = process_sql_response(response_content)  
    if sql_query and execute_and_verify_sql(sql_query):  
        if include_reasoning:  
            assistant_content = f"{reasoning}\n\n###final_sql_query:\n{sql_query}"  
        else:  
            assistant_content = sql_query  
        record = {  
            "messages": [  
                {"role": "system", "content": system_message},  
                {"role": "user", "content": prompt.strip()},  
                {"role": "assistant", "content": assistant_content}  
            ]  
        }  
        return record  
    else:  
        print(f"Deepseek did not produce a valid SQL for question: {question}")  
        return None  
  
# ----------------------------------------------------------------------------- STEP 3 – Split the records into train (80%), test (10%), and inference (10%)  
  
def split_data(records):  
    random.shuffle(records)  
    total = len(records)  
    train_count = int(total * 0.8)  
    remaining = total - train_count  
    test_count = remaining // 2  
    inference_count = remaining - test_count  
    train_set = records[:train_count]  
    test_set = records[train_count:train_count + test_count]  
    inference_set = records[train_count + test_count:]  
    return train_set, test_set, inference_set  
  
# ----------------------------------------------------------------------------- Main processing function  
  
def main():  
    load_dotenv()  # load environment variables from .env  
  
    parser = argparse.ArgumentParser(  
        description="Consolidated data generation program for SQL fine-tuning."  
    )  
    parser.add_argument("--num_questions", type=int, default=20,  
                        help="Number of unique business questions to generate.")  
    parser.add_argument("--sql_model", choices=["o3", "deepseek"], default="o3",  
                        help="Which model to use for SQL generation: o3 or deepseek.")  
    parser.add_argument("--include_reasoning", action="store_true",  
                        help="(Deepseek and optionally o3) Include reasoning details along with SQL output.")  
    parser.add_argument("--question_ontology", type=str, default="analytic_graph.json",  
                        help="File path for the ontology used in question generation.")  
    parser.add_argument("--sql_ontology", type=str, default="analytic_graph_v2.json",  
                        help="File path for the ontology used in SQL generation.")  
    args = parser.parse_args()  
  
    # Load ontology definitions  
    print("Loading ontology files...")  
    question_ontology_data = load_json(args.question_ontology)  
    sql_ontology_data = load_json(args.sql_ontology)  
    knowledge_graph_context = json.dumps(sql_ontology_data, indent=4)  
  
    # Generate business questions  
    print("Generating business questions ...")  
    question_client = AzureOpenAI(  
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),  
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),  
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),  
    )  
    questions = generate_question_list(question_ontology_data, args.num_questions, question_client)  
    print(f"Generated {len(questions)} unique questions.")  
  
    # Generate SQL queries  
    records = []  
    if args.sql_model == "o3":  
        # Prepare the o3 client  
        o3_client = AzureOpenAI(  
            api_key=os.environ.get("AZURE_OPENAI_O3_API_KEY"),  
            azure_endpoint=os.environ.get("AZURE_OPENAI_O3_ENDPOINT"),  
            api_version=os.environ.get("AZURE_OPENAI_O3_API_VERSION"),  
        )  
        model_deployment = os.environ.get("AZURE_OPENAI_O3_DEPLOYMENT")  
        if not model_deployment:  
            print("Missing AZURE_OPENAI_O3_DEPLOYMENT environment variable.")  
            sys.exit(1)  
        print("Generating SQL queries using the o3 model (with shared prompt)...")  
        for q in questions:  
            rec = generate_sql_o3(q, knowledge_graph_context, o3_client, model_deployment, args.include_reasoning)  
            if rec is not None:  
                records.append(rec)  
            time.sleep(1)  # slight delay to avoid rate limits  
    elif args.sql_model == "deepseek":  
        # Prepare deepseek client  
        deepseek_key = os.environ.get("AZURE_INFERENCE_CREDENTIAL")  
        deepseek_endpoint = os.environ.get("AZURE_INFERENCE_ENDPOINT")  
        if not deepseek_key or not deepseek_endpoint:  
            print("Missing AZURE_INFERENCE_CREDENTIAL or AZURE_INFERENCE_ENDPOINT environment variables.")  
            sys.exit(1)  
        deepseek_client = ChatCompletionsClient(  
            endpoint=deepseek_endpoint,  
            credential=AzureKeyCredential(deepseek_key),  
        )  
        print("Generating SQL queries using the deepseek model...")  
        for q in questions:  
            rec = generate_sql_deepseek(q, knowledge_graph_context, deepseek_client, args.include_reasoning)  
            if rec is not None:  
                records.append(rec)  
            time.sleep(1)  
    else:  
        print("Invalid sql_model option.")  
        sys.exit(1)  
  
    print(f"Successfully generated {len(records)} valid SQL record(s).")  
    if not records:  
        print("No valid records were generated. Exiting.")  
        sys.exit(1)  
  
    # Split the records into training, testing, and inference splits  
    train_set, test_set, inference_set = split_data(records)  
    print(f"Split into {len(train_set)} train, {len(test_set)} test, and {len(inference_set)} inference records.")  
  
    # Save the output records in OpenAI chat format (JSONL)  
    save_jsonl(train_set, "open_ai_data_train.jsonl")  
    save_jsonl(test_set, "open_ai_data_test.jsonl")  
    save_jsonl(inference_set, "open_ai_data_inference.jsonl")  
  
if __name__ == "__main__":  
    main()  