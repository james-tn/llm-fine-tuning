import os
import logging
import json
import numpy
import joblib
import torch
import pandas as pd
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

from transformers.pipelines import ConversationalPipeline, Conversation

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model, tokenizer
    # model_name="Llama-2-70b-chat"

  
    device_map = "auto"
    artifact_path=  "mlflow_model_folder/data/model"

    # artifact_path = f"{model_name}/artifacts/trained_model"
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
    print("Init complete")

def predict(data, model, tokenizer, **kwargs):

    TEMPERATURE_KEY = "temperature"
    MAX_GEN_LEN_KEY = "max_gen_len"
    DO_SAMPLE_KEY = "do_sample"
    MAX_NEW_TOKENS_KEY = "max_new_tokens"
    MAX_LENGTH_KEY = "max_length"
    TOP_P_KEY = "top_p"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    
    if isinstance(data, pd.DataFrame):
        data = data[data.columns[0]].tolist()

    addn_args = kwargs.get("addn_args", {})
    max_gen_len = addn_args.pop(MAX_GEN_LEN_KEY, 256)
    addn_args[MAX_NEW_TOKENS_KEY] = addn_args.get(MAX_NEW_TOKENS_KEY, max_gen_len)
    addn_args[MAX_LENGTH_KEY] = addn_args.get(MAX_LENGTH_KEY, 4096)
    addn_args[TEMPERATURE_KEY] = addn_args.get(TEMPERATURE_KEY, 0.9)
    addn_args[TOP_P_KEY] = addn_args.get(TOP_P_KEY, 0.6)
    addn_args[DO_SAMPLE_KEY] = addn_args.get(DO_SAMPLE_KEY, True)

    model.eval()
    conv_arr = data
    # validations
    assert len(conv_arr) > 0
    assert conv_arr[-1]["role"] == "user"
    next_turn = "system" if conv_arr[0]["role"] == "system" else "user"
    # Build conversation
    conversation = Conversation()
    conversation_agent = ConversationalPipeline(model=model, tokenizer=tokenizer)
    for i, conv in enumerate(conv_arr):
        if conv["role"] == "system":
            assert next_turn == "system", "System prompts can only be set at the start of the conversation"
            next_turn = "user"
            conversation.add_user_input(B_SYS + conv_arr[0]["content"].strip() + E_SYS)
            conversation.mark_processed()
        if conv["role"] == "assistant":
            assert next_turn == "assistant", "Invalid Turn. Expected user input"
            next_turn = "user"
            conversation.append_response(conv["content"].strip())
        elif conv["role"] == "user":
            assert next_turn == "user", "Invalid Turn. Expected assistant input"
            next_turn = "assistant"
            conversation.add_user_input(conv["content"].strip())
            if i != len(conv_arr[0:]) - 1:
                conversation.mark_processed()
    result = conversation_agent(conversation, use_cache=True, **addn_args)
    return {'output': result.generated_responses[-1]}

def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the pipeline
    method and return the result back
    """
    print("model 1: request received")
    model_input = json.loads(raw_data)["data"]
    content = model_input['text']
    max_length = model_input['max_length']
    # gen_config = GenerationConfig(max_new_tokens=max_length, temperature= 0.8)
    # content = f"<s>[INST]\n{instruction}\n\n### Input:\n{input}\n[/INST]"
    model_input = pd.DataFrame({"input":[
                        {
                            "role": "user",
                            "content": content,
                        },
                    ],})
    result= predict(model_input,model, tokenizer)
    # result = scoring_pipeline(texts, generation_config=gen_config)
    return result

