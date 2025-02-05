import os  
import json  
from pathlib import Path  
from dotenv import load_dotenv  
from tenacity import retry, wait_random_exponential, stop_after_attempt, stop_after_delay  
from openai import AzureOpenAI  
from concurrent.futures import ThreadPoolExecutor  
  
# Load environment variables  
load_dotenv()  
MAX_REC_NUM=500
openaikey = os.getenv("AZURE_OPENAI_API_KEY")  
openaiservice = os.getenv("AZURE_OPENAI_ENDPOINT")  
  
# Initialize OpenAI client  
client = AzureOpenAI(api_key=openaikey, api_version=os.getenv("AZURE_OPENAI_API_VERSION"), azure_endpoint=openaiservice)  
  
@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=(stop_after_attempt(20) | stop_after_delay(300)))  
def generate_single_record_questions(schema):  
    user_message = f"""  
    Use the provided database schema and business metrics definitions to generate at least 20 business questions that can be answered with a single record (e.g., top 1 best-selling product, highest revenue-generating store, customer with the most purchases).   
    ## Database Schema and Business Metrics Definitions  
    {schema}  
    ## Instructions:  
    - Ensure that each question generates an answer that is a single record.  
    - Focus on questions where aggregate functions like MAX, MIN, COUNT, or sorting (e.g., ORDER BY) are used to derive the top result.  
    - Keep questions simple and relevant to real business use cases.  
    - Examples of questions:  
      - What is the top 1 best-selling product by units sold?  
      - Which store generated the highest revenue in the last month?  
      - Who is the customer with the highest number of purchases?  
    - Do not number the questions. Do not add formula or SQL queries to the questions. 
    ## Output Format:  
    Output the questions in the following JSON format:  
    {{  
        "questions": ["question1", "question2", ...],  
        "difficulty": ["easy", "medium", "advanced", ...]  
    }}  
    """  
    response = client.chat.completions.create(  
        model=os.environ.get("AZURE_OPENAI_GPT4_DEPLOYMENT"),  
        messages=[  
            {"role": "system", "content": "You are a smart AI assistant, you excel in generating business questions that return single-record results."},  
            {"role": "user", "content": user_message},  
        ],  
        response_format={"type": "json_object"},  
        timeout=90,  
    )  
    response_message = json.loads(response.choices[0].message.content)  
    assert "questions" in response_message and "difficulty" in response_message  
    return response_message  
  
def process_scenario_step1_single_record(scenario_name, scenario_data):  
    schema = json.dumps(scenario_data, indent=4)  
    print("Generating single-record questions for scenario:", scenario_name)  
    final_data = {"questions": [], "difficulty": []}  
  
    while len(final_data["questions"]) < MAX_REC_NUM:  
        generated_data = generate_single_record_questions(schema)  
        final_data["questions"].extend(generated_data["questions"])  
        final_data["difficulty"].extend(generated_data["difficulty"])  
  
        # Deduplication  
        unique_questions = {q: d for q, d in zip(final_data["questions"], final_data["difficulty"])}  
        final_data = {  
            "questions": list(unique_questions.keys()),  
            "difficulty": list(unique_questions.values())  
        }  
  
        if len(final_data["questions"]) > MAX_REC_NUM:  
            final_data["questions"] = final_data["questions"][:MAX_REC_NUM]  
            final_data["difficulty"] = final_data["difficulty"][:MAX_REC_NUM]  
  
    return scenario_name, final_data  
@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=(stop_after_attempt(20) | stop_after_delay(300)))  
def generate_sql_queries(schema, question):  
    user_message = f"""  
    Given the following database schema, generate a SQL query that answers the provided business question.  
    ## Database Schema  
    {schema}  
    ## Business Question  
    {question}  
    ## Output Format:  
    {{  
        "sql_query": "Your SQL query here"  
    }}  
    """  
    response = client.chat.completions.create(  
        model=os.environ.get("AZURE_OPENAI_GPT4_DEPLOYMENT"),  
        messages=[  
            {"role": "system", "content": "You are a SQL expert. Write correct SQL queries based on the provided questions."},  
            {"role": "user", "content": user_message},  
        ],  
        timeout=90,  
    )  
    response_message = json.loads(response.choices[0].message.content)  
    assert "sql_query" in response_message  
    return response_message["sql_query"]  
  
def process_scenario_step2(scenario_name, questions_data, scenario_schema):  
    schema = json.dumps(scenario_schema, indent=4)  
    print("Generating SQL queries for scenario:", scenario_name)  
    sql_queries = []  
  
    for question in questions_data["questions"]:  
        sql_query = generate_sql_queries(schema, question)  
        sql_queries.append(sql_query)  
  
    return {  
        "questions": questions_data["questions"],  
        "sql_queries": sql_queries,  
        "difficulty": questions_data["difficulty"]  
    }  
if __name__ == "__main__":  
    # Load ontology  
    with open("./analytic_graph.json", "r") as file:  
        ontology = json.load(file)  
  
    split_scenarios = {}  
    for scenario in ontology["business_scenarios"]:  
        scenario_name = scenario["scenario"]  
        split_scenarios[scenario_name] = {  
            "date_format": ontology["date_format"],  
            "scenario": scenario,  
            "tables": {},  
            "metrics": []  
        }  
        for mapping in ontology["scenario_metric_mapping"]:  
            if mapping["scenario"] == scenario_name:  
                for metric_name in mapping["metrics"]:  
                    for metric in ontology["business_metrics"]:  
                        if metric["name"] == metric_name:  
                            split_scenarios[scenario_name]["metrics"].append(metric)  
                            for table in metric["tables"]:  
                                if table not in split_scenarios[scenario_name]["tables"]:  
                                    split_scenarios[scenario_name]["tables"][table] = ontology["tables"][table]  
    split_scenarios["cross_scenario"] = json.dumps(ontology, indent=4)  
  
    # Step 1: Generate single-record questions  
    all_questions = {}  
    with ThreadPoolExecutor() as executor:  
        results = executor.map(lambda item: process_scenario_step1_single_record(*item), split_scenarios.items())  
        for scenario_name, questions_data in results:  
            all_questions[scenario_name] = questions_data  
  
    # # Step 2: Generate SQL queries  
    # all_scenarios_output = {}  
    # with ThreadPoolExecutor() as executor:  
    #     results = executor.map(lambda item: process_scenario_step2(item[0], all_questions[item[0]], item[1]), split_scenarios.items())  
    #     for scenario_name, final_data in results:  
    #         all_scenarios_output[scenario_name] = final_data  
  
    # Save results  
    version_num = 1 
    try:  
        with open(f"./all_scenarios_questions_v{version_num}.json", "w") as file:  
            json.dump(all_questions, file, indent=4)  
    except Exception as e:  
        print(f"Error saving data: {e}")  