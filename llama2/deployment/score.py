import os
import logging
import json
import numpy
import joblib
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    
)

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global scoring_pipeline
    model_name="llama2_13b_fine_tuned"

  
    device_map = {"": 0}
    artifact_path = f"{model_name}/artifacts/trained_model"
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), artifact_path
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        device_map=device_map
    )
    scoring_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    print("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the pipeline
    method and return the result back
    """
    print("model 1: request received")
    model_input = json.loads(raw_data)["data"]
    texts = model_input['text']
    max_length = model_input['max_length']
    result = scoring_pipeline(texts, max_new_tokens=max_length)
    return result

