import os  
import json  
from pathlib import Path  
from dotenv import load_dotenv  
from tenacity import retry, wait_random_exponential, stop_after_attempt, stop_after_delay  
from openai import AzureOpenAI  
from sklearn.model_selection import train_test_split  
from concurrent.futures import ThreadPoolExecutor  
# Load environment variables  
env_path = Path('..') / 'secrets.env'  
load_dotenv(dotenv_path=env_path)  
  
openaikey = os.getenv("AZURE_OPENAI_API_KEY")  
openaiservice = os.getenv("AZURE_OPENAI_ENDPOINT")  
MAX_REC_NUM = 200  
BATCH_SIZE = 20  # Set your desired batch size  
  
# Initialize OpenAI client  
client = AzureOpenAI(api_key=openaikey, api_version=os.getenv("AZURE_OPENAI_API_VERSION"), azure_endpoint=openaiservice)  
  
@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=(stop_after_attempt(10) | stop_after_delay(300)))  
def generate_and_review(schema, prompt_type="single_scenario"):  
    user_message = f"""  
    Use the provided database schema and business metrics definitions, generate 10 diverse and creative business questions, ranging from easy to advanced levels. Each question should be paired with a corresponding SQL query that answers it.  
    ## Database Schema and Business Metrics Definitions  
    {schema}  
    ## Instructions:  
    - Develop a variety of business questions to cover different aspects of the database.  
    - Incorporate JOIN and aggregate functions to enhance the complexity and interest of the questions.  
    - Ensure that the questions and queries are practical and relevant to real business problems.  
    - Do not number the questions.  
    ## Output Format:  
    Output the questions and SQL queries in the following JSON format:  
    {{  
        "questions": ["question1", "question2", ...],  
        "sql_queries": ["query for question1", "query for question2", ...],  
        "difficulty": ["easy", "medium", "advanced", ...]  
    }}  
    """  
    if prompt_type == "cross_scenario":  
        user_message = f"""  
        Use the provided database schema and business metrics definitions from multiple scenarios, generate 10 diverse and creative business questions, ranging from easy to advanced levels. Each question should be paired with a corresponding SQL query that answers it.  
        ## Database Schema and Business Metrics Definitions  
        {schema}  
        ## Instructions:  
        - Develop a variety of business questions to cover different aspects of the database.  
        - Ensure that each question represents more than one scenario (cross scenario question)  
        - Incorporate JOIN and aggregate functions to enhance the complexity and interest of the questions.  
        - Ensure that the questions and queries are practical and relevant to real business problems.  
        - Do not number the questions.  
        ## Output Format:  
        Output the questions and SQL queries in the following JSON format:  
        {{  
            "questions": ["question1", "question2", ...],  
            "sql_queries": ["query for question1", "query for question2", ...],  
            "difficulty": ["easy", "medium", "advanced", ...]  
        }}  
        """  
    response = client.chat.completions.create(  
        model=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),  
        messages=[  
            {"role": "system", "content": "You are a smart AI assistant, you have excellent data analysis and SQL skills. You use SQL ANSI standard"},  
            {"role": "user", "content": user_message},  
        ],  
        response_format={"type": "json_object"},  
        timeout=90,  
    )  
    generated_data = json.loads(response.choices[0].message.content)  
    assert "questions" in generated_data and "sql_queries" in generated_data and "difficulty" in generated_data  
  
    reviewed_questions_and_queries = {  
        "questions": [],  
        "sql_queries": [],  
        "reviews": [],  
        "difficulty": generated_data["difficulty"]  
    }  
  
    for question, query in zip(generated_data["questions"], generated_data["sql_queries"]):  
        review_message = f"""  
        Given the following schema and business metrics definitions, and the following business question and corresponding SQL query, evaluate the correctness of the SQL query for the question. If there are any mistakes, provide the correct SQL query.  
        ## Database Schema and Business Metrics Definitions  
        {schema}  
        ## Business Question and SQL Query  
        Question: {question}  
        SQL Query: {query}  
        Output format: You write the evaluated question and sql query into json format as {{"question": "question", "sql_query": "query", "review": "review"}}.  
        Your output:  
        """  
        response = client.chat.completions.create(  
            model=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),  
            messages=[  
                {"role": "system", "content": "You are a SQL expert and you help review and correct SQL queries written by someone else. You are applying ANSI SQL standard"},  
                {"role": "user", "content": review_message},  
            ],  
            response_format={"type": "json_object"},  
        )  
        review_response = json.loads(response.choices[0].message.content)  
        assert "question" in review_response and "sql_query" in review_response and "review" in review_response  
        reviewed_questions_and_queries["questions"].append(review_response["question"])  
        reviewed_questions_and_queries["sql_queries"].append(review_response["sql_query"])  
        reviewed_questions_and_queries["reviews"].append(review_response["review"])  
  
    return reviewed_questions_and_queries  
  
def deduplicate_questions_and_queries(data):  
    unique_questions = {}  
    for question, query, review, difficulty in zip(data["questions"], data["sql_queries"], data.get("reviews", []), data["difficulty"]):  
        if question not in unique_questions:  
            unique_questions[question] = (query, review, difficulty)  
  
    deduplicated_data = {  
        "questions": list(unique_questions.keys()),  
        "sql_queries": [unique_questions[q][0] for q in unique_questions],  
        "reviews": [unique_questions[q][1] for q in unique_questions],  
        "difficulty": [unique_questions[q][2] for q in unique_questions]  
    }  
    return deduplicated_data  
  
def process_scenario(scenario_name, scenario_data):  
    schema = json.dumps(scenario_data, indent=4)  
    print("Working on scenario:", scenario_name)  
    final_data = {  
        "questions": [],  
        "sql_queries": [],  
        "reviews": [],  
        "difficulty": []  
    }  
    # Process in sequential batches  
    for _ in range(0, MAX_REC_NUM, BATCH_SIZE):  
        batch_data = {  
            "questions": [],  
            "sql_queries": [],  
            "reviews": [],  
            "difficulty": []  
        }  
        with ThreadPoolExecutor() as executor:  
            futures = []  
            for _ in range(BATCH_SIZE):  
                futures.append(executor.submit(generate_and_review, schema))  
            for future in futures:  
                reviewed_data = future.result()  
                batch_data["questions"].extend(reviewed_data["questions"])  
                batch_data["sql_queries"].extend(reviewed_data["sql_queries"])  
                batch_data["reviews"].extend(reviewed_data["reviews"])  
                batch_data["difficulty"].extend(reviewed_data["difficulty"])  
  
        deduplicated_batch = deduplicate_questions_and_queries(batch_data)  
        final_data["questions"].extend(deduplicated_batch["questions"])  
        final_data["sql_queries"].extend(deduplicated_batch["sql_queries"])  
        final_data["reviews"].extend(deduplicated_batch["reviews"])  
        final_data["difficulty"].extend(deduplicated_batch["difficulty"])  
  
    final_data = deduplicate_questions_and_queries(final_data)  
    if len(final_data["questions"]) > MAX_REC_NUM:  
        final_data["questions"] = final_data["questions"][:MAX_REC_NUM]  
        final_data["sql_queries"] = final_data["sql_queries"][:MAX_REC_NUM]  
        final_data["reviews"] = final_data["reviews"][:MAX_REC_NUM]  
        final_data["difficulty"] = final_data["difficulty"][:MAX_REC_NUM]  
  
    return scenario_name, final_data  
  
def main():  
    with open("./data/analytic_graph.json", "r") as file:  
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
  
    all_scenarios_output = {}  
    for scenario_name, scenario_data in split_scenarios.items():  
        scenario_name, final_data = process_scenario(scenario_name, scenario_data)  
        all_scenarios_output[scenario_name] = final_data  
  
    all_scenarios_output = deduplicate_questions_and_queries(all_scenarios_output)  
  
    with open("./data/all_scenarios_questions_and_queries_v6.json", "w") as file:  
        json.dump(all_scenarios_output, file, indent=4)  
  
    with open("./data/all_scenarios_questions_and_queries_v6.json", "r") as file:  
        all_scenarios_output = json.load(file)  
  
    all_data = []  
    for scenario_name, reviewed_data in all_scenarios_output.items():  
        for question, query, difficulty in zip(reviewed_data["questions"], reviewed_data["sql_queries"], reviewed_data["difficulty"]):  
            all_data.append({  
                "scenario": scenario_name,  
                "input": question,  
                "output": query,  
                "difficulty": difficulty  
            })  
  
    train, test = train_test_split(all_data, test_size=0.2, stratify=[item["scenario"] for item in all_data])  
  
    def create_message_format(item, schema, include_context=True):  
        schema_context = f"## Database Schema and Business Metrics Definitions\n{schema}\n\n## Question: " if include_context else ""  
        user_content = f"{schema_context}{item['input']}"  
        return {  
            "messages": [  
                {"role": "system", "content": "You are a smart AI assistant with excellent SQL and data analysis skills. You are querying the MDDX database, what is the SQL query for the following question?"},  
                {"role": "user", "content": user_content},  
                {"role": "assistant", "content": item["output"]}  
            ]  
        }  
  
    with open("./data/train_data_v6.json", "w") as f:  
        json.dump(train, f, indent=4)  
  
    with open("./data/test_data_v6.json", "w") as f:  
        json.dump(test, f, indent=4)  
  
    print("Train and test data saved to ./data/train_data_v6.json and ./data/test_data_v6.json")  
  
    with open("./data/sqltrain_openai_ctx_v6.jsonl", "w") as f:  
        for item in train:  
            schema = json.dumps(split_scenarios[item["scenario"]], indent=4) if item["scenario"] != "cross_scenario" else split_scenarios["cross_scenario"]  
            f.write(json.dumps(create_message_format(item, schema)) + "\n")  
  
    with open("./data/sqltest_openai_ctx_v6.jsonl", "w") as f:  
        for item in test:  
            schema = json.dumps(split_scenarios[item["scenario"]], indent=4) if item["scenario"] != "cross_scenario" else split_scenarios["cross_scenario"]  
            f.write(json.dumps(create_message_format(item, schema)) + "\n")  
  
    with open("./data/sqltrain_openai_v6.jsonl", "w") as f:  
        for item in train:  
            schema = json.dumps(split_scenarios[item["scenario"]], indent=4) if item["scenario"] != "cross_scenario" else split_scenarios["cross_scenario"]  
            f.write(json.dumps(create_message_format(item, schema, False)) + "\n")  
  
    with open("./data/sqltest_openai_v6.jsonl", "w") as f:  
        for item in test:  
            schema = json.dumps(split_scenarios[item["scenario"]], indent=4) if item["scenario"] != "cross_scenario" else split_scenarios["cross_scenario"]  
            f.write(json.dumps(create_message_format(item, schema, False)) + "\n")  
  
if __name__ == "__main__":  
    main()  
