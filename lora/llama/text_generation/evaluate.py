from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    GenerationConfig,
    pipeline,
    logging,
    
)
from peft import LoraConfig, PeftModel  

import torch
import mlflow
import argparse
import os
import pandas as pd
import json  
import re  
import random
class LAMA2Predict(mlflow.pyfunc.PythonModel):
    pass


def load_model(model, tokenizer):
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """

  
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    scoring_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    return scoring_pipeline, tokenizer


def run(scoring_pipeline, tokenizer, input_text):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the pipeline
    method and return the result back
    """
    gen_config = GenerationConfig(max_new_tokens=180, temperature= 0.1, stop_strings=["### End"] )
    result = scoring_pipeline(input_text, generation_config=gen_config,tokenizer=tokenizer)
    return result


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()


    # add arguments
    parser.add_argument("--chat_model", type=str, default="False")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_examples", type=int, default=1000)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--trained_model", type=str, default="trained_model")
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--evaluated_model", type=str)
    parser.add_argument("--test_dataset", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args
  
def process_jsonl(file_path):  
    with open(file_path, 'r') as file:  
        return [json.loads(line) for line in file]  
  
def extract_input(record):  
    output_index = record.index('### Output:')  
    return record[:output_index + len('### Output:')]  
  
def extract_output(record):  
    output_index = record.index('### Output:') + len('### Output:')  
  
    try:  
        # Try to find the end index using "### End"  
        end_index = record.index('### End', output_index)  
    except ValueError:  
        # If "### End" is not found, set end_index to 16 characters after "### Output:"  
        end_index = output_index + 35  
  
    return record[output_index:end_index]  
  
def extract_codes(output_text):  
    # Update the regex to match numbers with optional dots  
    match = re.search(r'\b((?:\d+\.)*\d+)\b', output_text)  
    if match:  
        # Return the matched number as a string without dots  
        return match.group(1).replace('.', '')    
def run_scoring_pipeline(scoring_pipeline, tokenizer, batch):  
    return run(scoring_pipeline, tokenizer, batch)
def extract_codes1(output_text):  
    # Split the text into lines  
    lines = output_text.split('\n')  
      
    # Initialize variables to store the extracted numbers  
    chapter, heading, subheading = None, None, None  
      
    # Iterate over each line to find the relevant codes  
    for line in lines:  
        if 'Chapter:' in line:  
            # Extract the chapter number, removing any non-numeric characters  
            chapter_match = re.search(r'Chapter:\s*([\d.]+)', line)  
            if chapter_match:  
                chapter = re.sub(r'\D', '', chapter_match.group(1))  
          
        elif 'Heading:' in line:  
            # Extract the heading number, removing any non-numeric characters  
            heading_match = re.search(r'Heading:\s*([\d.]+)', line)  
            if heading_match:  
                heading = re.sub(r'\D', '', heading_match.group(1))  
          
        elif 'Subheading:' in line:  
            # Extract the subheading number, removing any non-numeric characters  
            subheading_match = re.search(r'Subheading:\s*([\d.]+)', line)  
            if subheading_match:  
                subheading = re.sub(r'\D', '', subheading_match.group(1))  
      
    return chapter, heading, subheading  
  
def calculate_accuracy(true_values, predicted_values):  
    correct_chapter = sum(1 for true, pred in zip(true_values, predicted_values) if true[:2] == pred[:2])  
    correct_heading = sum(1 for true, pred in zip(true_values, predicted_values) if true[:4] == pred[:4])  
    correct_subheading = sum(1 for true, pred in zip(true_values, predicted_values) if true == pred)  
    total = len(true_values)  
  
    return {  
        'chapter_accuracy': correct_chapter / total,  
        'heading_accuracy': correct_heading / total,  
        'subheading_accuracy': correct_subheading / total  
    }  
  
def evaluate(scoring_pipeline, tokenizer, file_path, batch_size=5):  
    records = process_jsonl(file_path)  
    # Randomly sample 1000 records  
    sampled_records = random.sample(records, min(100, len(records)))  

    inputs = [extract_input(record['record']) for record in sampled_records]  
    true_outputs = [extract_codes(extract_output(record['record'])) for record in sampled_records]
      
    predicted_outputs = []  
    for i in range(0, len(inputs), batch_size):  
        batch = inputs[i:i + batch_size]  
        batch_results = run_scoring_pipeline(scoring_pipeline, tokenizer,batch)  
        for result in batch_results:  
            text_result = result[0]['generated_text']
            predicted_outputs.append(extract_codes(extract_output(text_result)))  
    print("predicted_outputs", predicted_outputs)
    accuracy = calculate_accuracy(true_outputs, predicted_outputs) 
    # Log the accuracy metrics with MLflow  
    with mlflow.start_run():  
        mlflow.log_metric("chapter_accuracy", accuracy['chapter_accuracy'])  
        mlflow.log_metric("heading_accuracy", accuracy['heading_accuracy'])  
        mlflow.log_metric("subheading_accuracy", accuracy['subheading_accuracy'])  
      
    print("Accuracies:", accuracy)  
 
  

def main(args):
    trained_model = args.trained_model
    model_name = args.model_name
    test_dataset = args.test_dataset
    model_dir = args.model_dir  
    evaluated_model = args.evaluated_model

    chat_model = args.chat_model
    model_artifact_path = "mlflow_model_folder"

    print("content of model path", os.listdir(trained_model))
    
    base_model = AutoModelForCausalLM.from_pretrained(  
        os.path.join(model_dir, "data", "model") ,  
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto"  


    )  
    model = PeftModel.from_pretrained(base_model, os.path.join(trained_model, "lora"))
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(  
         os.path.join(trained_model, "lora"),  
        local_files_only=True,  
        device_map="auto"  
    )  


    scoring_pipeline, tokenizer = load_model(model, tokenizer)
    print("pipeline loaded")
    evaluate(scoring_pipeline, tokenizer, test_dataset) 
    print("model evaluated")
    model.save_pretrained(evaluated_model)  
    # # Save the tokenizer  
    tokenizer.save_pretrained(evaluated_model)  

    with mlflow.start_run() as run:
            os.environ["AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"] = "1800" #give time for model to be registered
            mlflow.pyfunc.log_model(artifacts={"model": evaluated_model}, artifact_path=model_artifact_path, python_model=LAMA2Predict())
            model_uri = f"runs:/{run.info.run_id}/{model_artifact_path}"
            mlflow.register_model(model_uri, name = model_name,await_registration_for=1800)


if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)