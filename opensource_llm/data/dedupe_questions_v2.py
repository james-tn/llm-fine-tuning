import os  
import json  
from Levenshtein import distance as levenshtein_distance  # Ensure you have python-Levenshtein installed  
  
def is_unique_question(new_question, unique_questions, threshold=0.05):  
    """  
    Check if the new question is unique compared to the list of unique questions  
    based on the edit distance threshold.  
    """  
    for question in unique_questions:  
        max_length = max(len(new_question), len(question))  
        if max_length == 0:  # Avoid division by zero  
            continue  
        edit_distance = levenshtein_distance(new_question, question)  
        if edit_distance / max_length < threshold:  
            return False  
    return True  
  
def filter_unique_questions(questions_data, threshold=0.2):  
    """  
    Filter out questions that are not unique based on edit distance.  
    """  
    unique_questions = []  
    for question in questions_data:  
        if is_unique_question(question, unique_questions, threshold):  
            unique_questions.append(question)  
    return {"questions": unique_questions}  
  
def process_questions(input_file):  
    """  
    Load questions from the input JSON file, filter them based on uniqueness,  
    and save the unique questions to a new JSON file.  
    """  
    # Load questions from input file  
    with open(input_file, "r") as file:  
        data = json.load(file)  
        questions_data = data.get("questions", [])  
  
    # Process the questions  
    print("Processing questions")  
    try:  
        unique_questions = filter_unique_questions(questions_data)  
    except Exception as e:  
        print(f"Error processing questions: {e}")  
        return  
  
    # Save the filtered unique questions to a new JSON file  
    output_file = input_file.replace(".json", "_deduped.json")  
    try:  
        with open(output_file, "w") as file:  
            json.dump(unique_questions, file, indent=4)  
        print(f"Filtered unique questions saved to {output_file}")  
    except Exception as e:  
        print(f"Error saving unique questions: {e}")  
  
if __name__ == "__main__":  
    # Input file path  
    input_file = "./questions_v4.json"  
  
    # Process the questions and generate unique ones  
    process_questions(input_file)  