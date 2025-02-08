import os  
import json  
from pathlib import Path  
from dotenv import load_dotenv  
from tenacity import retry, wait_random_exponential, stop_after_attempt, stop_after_delay  
from openai import AzureOpenAI  
from concurrent.futures import ThreadPoolExecutor  
from difflib import SequenceMatcher  
  
# Load environment variables  
load_dotenv()  
MAX_REC_NUM = 3000  
openaikey = os.getenv("AZURE_OPENAI_API_KEY")  
openaiservice = os.getenv("AZURE_OPENAI_ENDPOINT")  
  
# Initialize OpenAI client  
client = AzureOpenAI(api_key=openaikey, api_version=os.getenv("AZURE_OPENAI_API_VERSION"), azure_endpoint=openaiservice)  
  
@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=(stop_after_attempt(20) | stop_after_delay(300)))  
def generate_questions(ontology):  
    user_message = f"""  
    Use the provided analytics graph to generate at least 20 diverse business questions.  
  
    ## Analytics Graph  
    {ontology}  
  
    ## Instructions:  
    - Generate a variety of business questions.  
    - Questions should vary in difficulty levels: easy, medium, and advanced.  
    - Questions can involve aggregates, comparisons, trends, or any kind of analytical insights.  
    - Ensure questions are practical and relevant to real-world business use cases.  
    - Examples of questions:  
      - What is the top 1 best-selling product by units sold?  
      - How has the revenue trend changed over the past year?  
      - What is the average purchase size for repeat customers?  
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
            {"role": "system", "content": "You are a smart AI assistant, you excel in generating diverse business questions."},  
            {"role": "user", "content": user_message},  
        ],  
        response_format={"type": "json_object"},  
        timeout=90,  
    )  
    response_message = json.loads(response.choices[0].message.content)  
    assert "questions" in response_message and "difficulty" in response_message  
    return response_message  
  
def is_similar(question1, question2, threshold=0.9):  
    """  
    Check if two questions are at least `threshold` similar using SequenceMatcher.  
    """  
    similarity = SequenceMatcher(None, question1, question2).ratio()  
    return similarity >= threshold  
  
def deduplicate_questions(questions, difficulties, threshold=0.9):  
    """  
    Deduplicate questions to ensure no two questions are more than `threshold` similar.  
    """  
    unique_questions = []  
    unique_difficulties = []  
  
    for i, question in enumerate(questions):  
        if all(not is_similar(question, existing_question, threshold) for existing_question in unique_questions):  
            unique_questions.append(question)  
            unique_difficulties.append(difficulties[i])  
  
    return unique_questions, unique_difficulties  
  
def process_questions(ontology_data):  
    ontology = json.dumps(ontology_data, indent=4)  
    print("Generating questions...")  
    final_data = {"questions": [], "difficulty": []}  
  
    while len(final_data["questions"]) < MAX_REC_NUM:  
        generated_data = generate_questions(ontology)  
        final_data["questions"].extend(generated_data["questions"])  
        final_data["difficulty"].extend(generated_data["difficulty"])  
  
        # Deduplication  
        unique_questions, unique_difficulties = deduplicate_questions(  
            final_data["questions"], final_data["difficulty"], threshold=0.9  
        )  
        final_data = {"questions": unique_questions, "difficulty": unique_difficulties}  
  
        if len(final_data["questions"]) > MAX_REC_NUM:  
            final_data["questions"] = final_data["questions"][:MAX_REC_NUM]  
            final_data["difficulty"] = final_data["difficulty"][:MAX_REC_NUM]  
  
    return final_data  
  
if __name__ == "__main__":  
    # Load analytics graph  
    with open("./analytic_graph.json", "r") as file:  
        ontology_data = json.load(file)  
  
    # Generate questions  
    questions_data = process_questions(ontology_data)  
  
    # Save results  
    version_num = 2
    try:  
        with open(f"./questions_by_difficulty_v{version_num}.json", "w") as file:  
            json.dump(questions_data, file, indent=4)  
        print(f"Questions successfully saved to 'questions_by_difficulty_v{version_num}.json'")  
    except Exception as e:  
        print(f"Error saving data: {e}")  