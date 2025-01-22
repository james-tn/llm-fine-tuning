from vllm import LLM, SamplingParams  
from transformers import AutoTokenizer, AutoModelForCausalLM  
from peft import PeftModel  
import torch  
import mlflow  
import argparse  
import os  
import json  
import re  
import random  
import logging  
import shutil  # For temporary directory handling  
import gc  

  
  
class LAMA2Predict(mlflow.pyfunc.PythonModel):  
    pass  
  
  
def load_model(model_dir, trained_model_path,merged_model_dir,  use_lora):  
    """  
    Load the model based on whether LoRA is used or the model is fully fine-tuned.  
    Merge LoRA adapters into the base model if necessary.  
    """  
    num_gpus = torch.cuda.device_count()  # Detect the number of available GPUs  
    tensor_parallel_size = max(1, num_gpus)  # Use at least 1 GPU  
    logging.info(f"Detected {num_gpus} GPU(s). Using tensor_parallel_size={tensor_parallel_size}.")  
  
    if use_lora:  
        # Load the base model  
        base_model_path = os.path.join(model_dir, "data", "model")  
        logging.info(f"Loading base model from: {base_model_path}")  
        model = AutoModelForCausalLM.from_pretrained(  
            base_model_path,  
            torch_dtype=torch.bfloat16,  # Use bfloat16 for A100/H100  
            device_map="auto"  
        )  

        # Load the LoRA adapter  
        logging.info(f"Loading LoRA adapter from: {trained_model_path}")  
        model = PeftModel.from_pretrained(model, trained_model_path)  
  
        # Merge LoRA adapter with the base model  
        logging.info("Merging LoRA adapter with base model...")  
        model = model.merge_and_unload()  
  
        # Save the merged model to a temporary directory  
          
        model.save_pretrained(merged_model_dir)  
        tokenizer = AutoTokenizer.from_pretrained(  
            trained_model_path,  
            local_files_only=True,  
            device_map="auto"  
        )  
        tokenizer.save_pretrained(merged_model_dir) 
        del model  
        del tokenizer
        gc.collect()  
        torch.cuda.empty_cache() 


        logging.info(f"Merged model saved to temporary directory: {merged_model_dir}") 
  
        model_path = merged_model_dir  
    else:  
        # Load the fully fine-tuned model  
        logging.info(f"Loading fully fine-tuned model from: {trained_model_path}")  
        model_path = trained_model_path  
  
    # Initialize the vLLM model  
    vllm = LLM(  
        model=model_path, 
        tensor_parallel_size=tensor_parallel_size,  
        dtype="bfloat16",  # Use bfloat16 for A100/H100 GPUs  
        max_model_len = 1000,
        enforce_eager = True
    )  
  
    return vllm  
  
  
def run(vllm, input_texts):  
    """  
    Perform the actual scoring/prediction using vLLM's LLM API.  
    """  
    sampling_params = SamplingParams(  
        max_tokens=180,  
        temperature=0.1,  
        stop=["### End"]  # Stop strings for generation  
    )  
    results = vllm.generate(input_texts, sampling_params)  
    return [result.outputs[0].text for result in results]  
  
  
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
  
  
def evaluate(vllm, file_path, batch_size=5):  
    records = process_jsonl(file_path)  
    # Randomly sample records  
    random.seed(42)  
    sampled_records = random.sample(records, min(args.num_examples, len(records)))  
  
    inputs = [extract_input(record['record']) for record in sampled_records]  
    true_outputs = [extract_codes(extract_output(record['record'])) for record in sampled_records]  
  
    predicted_outputs = []  
    for i in range(0, len(inputs), batch_size):  
        batch = inputs[i:i + batch_size]  
        batch_results = run(vllm, batch)  
        for text_result in batch_results:  
            predicted_outputs.append(extract_codes(extract_output(text_result)))  
  
    logging.info("10 predicted outputs: %s", predicted_outputs[:10])  
    accuracy = calculate_accuracy(true_outputs, predicted_outputs)  
  
    # Log the accuracy metrics with MLflow  
    with mlflow.start_run():  
        mlflow.log_metric("chapter_accuracy", accuracy['chapter_accuracy'])  
        mlflow.log_metric("heading_accuracy", accuracy['heading_accuracy'])  
        mlflow.log_metric("subheading_accuracy", accuracy['subheading_accuracy'])  
        logging.info("Accuracies: %s", accuracy)  
  
    return accuracy  
  
  
def main(args):  
    # Load the model and tokenizer  
    trained_model_path = os.path.join(args.trained_model, "model")  
    vllm = load_model(args.model_dir, trained_model_path,args.evaluated_model, args.use_lora)  
    logging.info("vLLM LLM loaded")  
  
    # Evaluate the model  
    accuracy = evaluate(vllm, args.test_dataset)  
    logging.info("Model evaluated")  
  
    # Save the tokenizer and register the model with MLflow  
    with mlflow.start_run() as run:  
        os.environ["AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"] = "1800"  # Give time for model to be registered  
        mlflow.pyfunc.log_model(  
            artifacts={"model": args.evaluated_model},  
            artifact_path=args.mlflow_artifact_dir,  
            python_model=LAMA2Predict()  
        )  
        model_uri = f"runs:/{run.info.run_id}/{args.mlflow_artifact_dir}"  
        mlflow.register_model(model_uri, name=args.model_name, await_registration_for=1800)  
  
  
if __name__ == "__main__":  
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
        parser = argparse.ArgumentParser()  
        parser.add_argument("--chat_model", type=str, default="False")  
        parser.add_argument("--epochs", type=int, default=1)  
        parser.add_argument("--num_examples", type=int, default=500)  
        parser.add_argument("--model_name", type=str)  
        parser.add_argument("--trained_model", type=str, default="trained_model")  
        parser.add_argument("--mlflow_artifact_dir", type=str)  
        parser.add_argument("--model_dir", type=str)  
        parser.add_argument("--evaluated_model", type=str)  
        parser.add_argument("--test_dataset", type=str)  
        parser.add_argument("--use_lora", type=str2bool, nargs='?', const=True, default=True)  
        parser.add_argument("--verbosity", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')  
        return parser.parse_args()  
  
    args = parse_args()  
    logging.basicConfig(level=getattr(logging, args.verbosity.upper()))  
    main(args)  