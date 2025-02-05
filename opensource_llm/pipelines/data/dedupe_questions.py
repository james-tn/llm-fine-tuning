import os  
import json  
from dotenv import load_dotenv  
from openai import AzureOpenAI  
from tenacity import retry, wait_random_exponential, stop_after_attempt, stop_after_delay  
  
# Load environment variables  
load_dotenv()  
  
# Initialize OpenAI client  
openaikey = os.getenv("AZURE_OPENAI_API_KEY")  
openaiservice = os.getenv("AZURE_OPENAI_ENDPOINT")  
client = AzureOpenAI(api_key=openaikey, api_version=os.getenv("AZURE_OPENAI_API_VERSION"), azure_endpoint=openaiservice)  
  
@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=(stop_after_attempt(10) | stop_after_delay(180)))  
def filter_unique_questions(scenario_name, questions_data):  
    """  
    Prompt the LLM to select unique questions from a list of questions for a specific scenario.  
    """  
    user_message = f"""  
    You are an expert in analyzing and improving question quality.  
    Below is a list of business questions for the scenario "{scenario_name}" along with their difficulty levels.  
  
    ## Questions and Difficulties  
    {json.dumps(questions_data, indent=4)}  
  
    ## Instructions:  
    - Filter out duplicate or similar questions. Ensure all remaining questions are unique.  
    - Keep the questions focused, clear, and relevant to business use cases.  
    - Retain the corresponding difficulty levels for each question.  
    - Provide the result in the following JSON format:  
      {{  
          "questions": ["unique_question1", "unique_question2", ...],  
          "difficulty": ["difficulty1", "difficulty2", ...]  
      }}  
    """  
    response = client.chat.completions.create(  
        model=os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT"),  
        messages=[  
            {"role": "system", "content": "You are a smart AI assistant, skilled in refining and filtering questions for uniqueness."},  
            {"role": "user", "content": user_message},  
        ],  
        response_format={"type": "json_object"},  
        timeout=90,  
    )  
    response_message = json.loads(response.choices[0].message.content)  
    assert "questions" in response_message and "difficulty" in response_message  
    return response_message  
  
def process_questions(input_file):  
    """  
    Load questions from the input JSON file, send them to the LLM for filtering, and save the unique questions to a new JSON file.  
    """  
    # Load questions from Step 1 output  
    with open(input_file, "r") as file:  
        all_questions = json.load(file)  
  
    # Output container for unique questions  
    unique_questions_output = {}  
  
    # Process each scenario  
    for scenario_name, questions_data in all_questions.items():  
        print(f"Processing scenario: {scenario_name}")  
        try:  
            unique_questions = filter_unique_questions(scenario_name, questions_data)  
            unique_questions_output[scenario_name] = unique_questions  
        except Exception as e:  
            print(f"Error processing scenario '{scenario_name}': {e}")  
  
    # Save the filtered unique questions to a new JSON file  
    output_file = input_file.replace(".json", "_unique.json")  
    try:  
        with open(output_file, "w") as file:  
            json.dump(unique_questions_output, file, indent=4)  
        print(f"Filtered unique questions saved to {output_file}")  
    except Exception as e:  
        print(f"Error saving unique questions: {e}")  
  
if __name__ == "__main__":  
    # Input file from
    input_file = "./all_scenarios_questions_v1.json"  
  
    # Process the questions and generate unique ones  
    process_questions(input_file)  
