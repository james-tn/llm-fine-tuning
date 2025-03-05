import os  
import json  
from pathlib import Path  
from dotenv import load_dotenv  
from tenacity import retry, wait_random_exponential, stop_after_attempt, stop_after_delay  
from openai import AzureOpenAI  
from difflib import SequenceMatcher  
  
# Load environment variables  
load_dotenv()  
MAX_REC_NUM = 800  
openaikey = os.getenv("AZURE_OPENAI_API_KEY")  
openaiservice = os.getenv("AZURE_OPENAI_ENDPOINT")  
  
# Initialize OpenAI client  
client = AzureOpenAI(  
    api_key=openaikey,  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),  
    azure_endpoint=openaiservice  
)  
  
def load_existing_questions(file_path):  
    """  
    Load existing questions from a JSONL file to ensure new questions are distinct.  
    """  
    existing_questions = []  
    try:  
        with open(file_path, "r") as file:  
            for line in file:  
                data = json.loads(line.strip())  
                if "user" in data:  
                    existing_questions.append(data["user"])  
    except Exception as e:  
        print(f"Error loading existing questions: {e}")  
    return existing_questions  
  
@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=(stop_after_attempt(20) | stop_after_delay(300)))  
def generate_questions(ontology):  
    user_message = f"""  
    Using the provided analytics graph, create at least 20 distinct business questions that can each be answered with a single SQL query.

  
    ## Analytics Graph  
    {ontology}  
  
    ## Instructions:  
    - Develop a wide array of business questions, ranging from basic to advanced complexity.
    - Ensure that each question can be resolved using quantitative data.
    - Focus on questions that are practical and applicable to real-world business scenarios.
    - Frame questions in such a way that they have a clear, singular answer.
    - Base all questions on data from the years 2020 to 2024.
    - Do not include numbering, formulas, or SQL queries in the questions themselves.
    ## Output Format:  
    Output the questions in the following JSON format:  
    {{  
        "questions": ["question1", "question2", ...]  
    }}  
    """  
    response = client.chat.completions.create(  
        model=os.environ.get("AZURE_OPENAI_GPT4_DEPLOYMENT"),  
        messages=[  
            {"role": "system", "content": "You are a smart AI assistant, you excel in generating diverse business questions."},  
            {"role": "user", "content": user_message},  
        ],  
        response_format={"type": "json_object"},  
        timeout=90,  
    )  
    response_message = json.loads(response.choices[0].message.content)  
    assert "questions" in response_message  
    return response_message  
  
def is_similar(question1, question2, threshold=0.9):  
    """  
    Check if two questions are at least `threshold` similar using SequenceMatcher.  
    """  
    similarity = SequenceMatcher(None, question1, question2).ratio()  
    return similarity >= threshold  
  
def deduplicate_questions(questions, existing_questions, threshold=0.9):  
    """  
    Deduplicate questions to ensure no two questions (including existing ones) are more than `threshold` similar.  
    """  
    unique_questions = []  
  
    for question in questions:  
        if all(not is_similar(question, existing_question, threshold) for existing_question in unique_questions) and  all(not is_similar(question, existing_question, threshold) for existing_question in existing_questions):  
            unique_questions.append(question)  
  
    return unique_questions  
  
def process_questions(ontology_data, existing_questions):  
    ontology = json.dumps(ontology_data, indent=4)  
    print("Generating questions...")  
    final_questions = []  
  
    while len(final_questions) < MAX_REC_NUM:  
        generated_data = generate_questions(ontology)  
        final_questions.extend(generated_data["questions"])  
  
        # Deduplication  
        final_questions = deduplicate_questions(final_questions, existing_questions, threshold=0.9)  
  
        if len(final_questions) > MAX_REC_NUM:  
            final_questions = final_questions[:MAX_REC_NUM]  
  
    return {"questions": final_questions}  
  
if __name__ == "__main__":  
    # Load analytics graph  
    with open("./analytic_graph_v2.json", "r") as file:  
        ontology_data = json.load(file)  
  
    # Load existing questions from sql_result_v2.jsonl  
    existing_questions_file = "./sql_result_v2.jsonl"  
    existing_questions = load_existing_questions(existing_questions_file)  
  
    # Generate questions  
    questions_data = process_questions(ontology_data, existing_questions)  
  
    # Save results  
    version_num = 4  
    try:  
        with open(f"./questions_v{version_num}.json", "w") as file:  
            json.dump(questions_data, file, indent=4)  
        print(f"Questions successfully saved to 'questions_v{version_num}.json'")  
    except Exception as e:  
        print(f"Error saving data: {e}")  