from vllm import LLM, SamplingParams  
import mlflow  
import argparse  
import os  
import pandas as pd  
import json  
import re  
import random  
import logging  
  
class LAMA2Predict(mlflow.pyfunc.PythonModel):  
    pass  
  
def load_model_vllm(model_path):  
    """  
    Load the vLLM model for inference.  
    """  
    llm = LLM(model=model_path)  
    return llm  
  
def run_vllm(llm, input_texts, max_new_tokens=180, temperature=0.1, stop_strings=["### End"]):  
    """  
    Run the vLLM model inference on the input texts.  
    """  
    sampling_params = SamplingParams(  
        max_tokens=max_new_tokens,  
        temperature=temperature,  
        stop=stop_strings,  
    )  
    outputs = llm.generate(input_texts, sampling_params)  
    return outputs  
  
def str2bool(v):  
    if isinstance(v, bool):  
        return v  
    if v.lower() in ('yes', 'true', 't', 'y', '1'):  
        return True  
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):  
        return False  
    else:  
        raise argparse.ArgumentTypeError('Boolean value expected.')  
  
def parse_args():  
    # setup arg parser  
    parser = argparse.ArgumentParser()  
  
    # add arguments  
    parser.add_argument("--chat_model", type=str, default="False")  
    parser.add_argument("--epochs", type=int, default=1)  
    parser.add_argument("--num_examples", type=int, default=500)  
    parser.add_argument("--model_name", type=str)  
    parser.add_argument("--trained_model", type=str, default="trained_model")  
    parser.add_argument("--model_dir", type=str)  
    parser.add_argument("--mlflow_artifact_dir", type=str)  
    parser.add_argument("--evaluated_model", type=str)  
    parser.add_argument("--test_dataset", type=str)  
    parser.add_argument("--verbosity", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help="Set the logging verbosity level.")  
  
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
    else:  
        return "-1"  
  
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
  
def evaluate(llm, file_path, batch_size=5, max_new_tokens=180, temperature=0.1, stop_strings=["### End"]):  
    records = process_jsonl(file_path)  
    # Randomly sample records  
    random.seed(42)  
    sampled_records = random.sample(records, min(args.num_examples, len(records)))  
  
    inputs = [extract_input(record['record']) for record in sampled_records]  
    true_outputs = [extract_codes(extract_output(record['record'])) for record in sampled_records]  
  
    predicted_outputs = []  
    for i in range(0, len(inputs), batch_size):  
        batch = inputs[i:i + batch_size]  
        batch_results = run_vllm(llm, batch, max_new_tokens=max_new_tokens, temperature=temperature, stop_strings=stop_strings)  
        for result in batch_results:  
            # Extract the generated text for each input  
            text_result = result.outputs[0].text  
            predicted_outputs.append(extract_codes(extract_output(text_result)))  
  
    logging.info("Predicted outputs: %s", predicted_outputs)  
    accuracy = calculate_accuracy(true_outputs, predicted_outputs)  
  
    # Log the accuracy metrics with MLflow  
    with mlflow.start_run():  
        mlflow.log_metric("chapter_accuracy", accuracy['chapter_accuracy'])  
        mlflow.log_metric("heading_accuracy", accuracy['heading_accuracy'])  
        mlflow.log_metric("subheading_accuracy", accuracy['subheading_accuracy'])  
        logging.info("Accuracies: %s", accuracy)  
  
def main(args):  
    trained_model = args.trained_model  
    model_name = args.model_name  
    test_dataset = args.test_dataset  
    evaluated_model = args.evaluated_model  
    trained_model_path = os.path.join(trained_model, "model")  
    logging.info("Content of model path: %s", os.listdir(trained_model_path))  

    logging.info("Loading vLLM model from: %s", trained_model_path)  
    llm = load_model_vllm(trained_model_path)  
  
    logging.info("Model loaded successfully. Starting evaluation.")  
    evaluate(llm, test_dataset)  
    logging.info("Model evaluation completed.")  
  
    # Save the model directory for future use (optional, depends on your workflow)  
    os.makedirs(evaluated_model, exist_ok=True)  
    logging.info("Model saved in: %s", evaluated_model)  
  
    # Log the model with MLflow  
    with mlflow.start_run() as run:  
        os.environ["AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"] = "1800"  # Increase timeout for model registration  
        mlflow.pyfunc.log_model(artifacts={"model": evaluated_model}, artifact_path="vllm_model_folder", python_model=LAMA2Predict())  
        model_uri = f"runs:/{run.info.run_id}/vllm_model_folder"  
        mlflow.register_model(model_uri, name=model_name, await_registration_for=1800)  
  
if __name__ == "__main__":  
    args = parse_args()  
    logging.basicConfig(level=getattr(logging, args.verbosity.upper()))  
    main(args)  