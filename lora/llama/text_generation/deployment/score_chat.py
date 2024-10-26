import os
import json
import mlflow

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
import torch
from transformers.pipelines import ConversationalPipeline, Conversation

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model, tokenizer
    model_name = os.getenv("AZUREML_MODEL_DIR").split('/')[-2]
    # loading_path =os.path.join(os.getenv("AZUREML_MODEL_DIR"), model_name)
    # print("loading_path is ", loading_path)
    # print("inside model loading path", os.listdir(loading_path))
    # model = mlflow.pyfunc.load_model(loading_path)

    # print("model directory is ", os.getenv("AZUREML_MODEL_DIR"))  

  
    device_map = "auto"
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
    print("Init complete")

def predict(data, model, tokenizer, **kwargs):

    TEMPERATURE_KEY = "temperature"
    MAX_GEN_LEN_KEY = "max_gen_len"
    DO_SAMPLE_KEY = "do_sample"
    MAX_NEW_TOKENS_KEY = "max_new_tokens"
    MAX_LENGTH_KEY = "max_length"
    TOP_P_KEY = "top_p"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    
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
    return result

def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the pipeline
    method and return the result back
    """
    input = json.loads(raw_data)["data"]
    contents = input['text']
    max_gen_len = input['max_gen_len']
    temperature = input['temperature']
    model_input =[]
    
    PROMPT_DICT_CHAT ="<s>[INST]\n{context}\n\n### Question:\n{input}\n[/INST]"
    for content in contents:
        example = {"context":"You are querying the sales database, what is the SQL query for the following question?","input":content}
        
        prompt = PROMPT_DICT_CHAT.format(input=example["input"], context=example["context"])
        prompt = {"role": "user","content": prompt} 
        model_input.append(prompt)
    result = predict(model_input, model, tokenizer, max_gen_len=max_gen_len, temperature=temperature)
    return {'output': result.generated_responses[-1]}
