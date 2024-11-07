import os
import torch
import math
from datasets import load_dataset
import argparse
import mlflow
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    TrainerCallback
    
)
from transformers.integrations import MLflowCallback

from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import json
from datasets import Dataset

#Function to load model for scoring 


class MlflowLoggingCallback(TrainerCallback):  
    def on_log(self, args, state, control, logs=None, **kwargs):  
        # Log metrics to MLflow  
        if logs is not None:  
            mlflow.log_metrics(logs, step=state.global_step)  
            mlflow.log_metric('epoch', state.epoch)  

class LAMA2Predict(mlflow.pyfunc.PythonModel):
    pass



def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()


    # add arguments
    parser.add_argument("--chat_model", type=str, default="False")
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_examples", type=int, default=1000)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--trained_model", type=str, default="trained_model")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--mounted_data_file", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args

def launch_main(path):
    main(model_dir=path)
def main(args):
    PATH = args.model_dir
    num_examples = args.num_examples
    trained_model = args.trained_model
    model_name = args.model_name
    chat_model = args.chat_model
    dataset_path = args.dataset_path
    mounted_data_file = args.mounted_data_file

    print("trained model path", trained_model)
    print("Model dir: ", os.listdir(os.path.join(PATH,"data", "model")) )

    rank = int(os.environ.get('RANK'))
    print("rank ", rank)

    
    # The instruction dataset to use

    # Fine-tuned model name
    new_model = "fine_tuned_model"

    ################################################################################
    # QLoRA parameters
    ################################################################################

    # LoRA attention dimension
    lora_r = 64

    # Alpha parameter for LoRA scaling
    lora_alpha = 16

    # Dropout probability for LoRA layers
    lora_dropout = 0.1

    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    ################################################################################
    # TrainingArguments parameters
    ################################################################################

    # Output directory where the model predictions and checkpoints will be stored
    output_dir = "./results"

    # Number of training epochs
    num_train_epochs = args.epochs

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = False

    # Batch size per GPU for training
    per_device_train_batch_size = 1

    # Batch size per GPU for evaluation
    per_device_eval_batch_size = 1

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 1

    # Enable gradient checkpointing
    gradient_checkpointing = True

    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3

    # Initial learning rate (AdamW optimizer)
    learning_rate = 2e-4

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001

    # Optimizer to use
    optim = "paged_adamw_32bit"

    # Learning rate schedule
    lr_scheduler_type = "cosine"

    # Number of training steps (overrides num_train_epochs)
    max_steps = -1

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True

    # Save checkpoint every X updates steps
    save_steps = 0

    # Log every X updates steps
    logging_steps = 25

    ################################################################################
    # SFT parameters
    ################################################################################

    # Maximum sequence length to use
    max_seq_length = None

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = False


    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}
    print("device map ", device_map)
    list_data_dict = pd.read_json(dataset_path, lines=True).to_dict(orient="records")
    list_data_dict_l = pd.read_json(mounted_data_file, lines=True).to_dict(orient="records")
    print("listed datafile, ", len(list_data_dict_l))
    list_data_dict= list_data_dict[:num_examples]
    contexts = [item["context"] for item in list_data_dict] 
    inputs = [item["input"] for item in list_data_dict] 
    outputs = [item["output"] for item in list_data_dict]
    dataset = Dataset.from_dict({"context":contexts, "input": inputs, "output":outputs})   


    PROMPT_DICT = {
        "prompt_no_context": (
            "### Question:\n{input}\n\n### Response:{output}"
        ),
        "prompt_context": (
            "\n{context}\n\n### Question:\n{input}\n\n### Response:{output}"
        ),

    }
    
    PROMPT_DICT_CHAT = {
        "prompt_context": (
            "<s>[INST]\n{context}\n\n### Question:\n{input}\n[/INST]\n### Response:{output}\n</s>"
        ),
        "prompt_no_context": (
            "<s>[INST]\n{input}\n[/INST]\n### Response:{output}\n</s>"
        ),
    }
    if chat_model=="True":
        print("Training for Chat Model")
        PROMPT_DICT = PROMPT_DICT_CHAT


    def formatting_prompts_func(example):
        # example_no_context = PROMPT_DICT["prompt_no_context"]
        example_input = PROMPT_DICT["prompt_context"]
        output_texts = []
        for i in range(len(example['input'])):
            text =  example_input.format_map({"context":example["context"][i], "input":example["input"][i], "output":example["output"][i]})
            output_texts.append(text)
        return output_texts

        






    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(PATH,"data", "model"),
                low_cpu_mem_usage=True,
        local_files_only=True,
        quantization_config=bnb_config,
        device_map=device_map
    )



    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(PATH,"data", "model"),
        
        local_files_only=True,
        device_map=device_map
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",

                        "embed_tokens", "lm_head",], # Add for continual pretraining

    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        evaluation_strategy="epoch",
        # report_to="tensorboard"
    )
    training_arguments.ddp_find_unused_parameters = False
    response_template_with_context = "\n### Response:"  # We added context here: "\n". This is enough for this tokenizer
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`

    # data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    split_ds = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split_ds["train"]
    eval_ds = split_ds["test"]
    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        # dataset_text_field="text",
        callbacks=[MlflowLoggingCallback()],

        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
        # data_collator=data_collator,
        formatting_func=formatting_prompts_func,

    )
    trainer.remove_callback(MLflowCallback)

    
    with mlflow.start_run() as run:
        trainer.train()
        eval_results = trainer.evaluate()
        mlflow.log_metric("perplexity",math.exp(eval_results['eval_loss']))
        trainer.model.save_pretrained(new_model)
        model_output_dir = "trained_model"
        model_artifact_path = "mlflow_model_folder"
        # Empty VRAM
        del model
        # del pipe
        del trainer
        import gc
        gc.collect()
        if rank==0:

            # Reload model in FP16 and merge it with LoRA weights
            base_model = AutoModelForCausalLM.from_pretrained(
                os.path.join(PATH,"data", "model"),
                        low_cpu_mem_usage=True,
                local_files_only=True,
                return_dict=True,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
            model = PeftModel.from_pretrained(base_model, new_model)
            model = model.merge_and_unload()

            # Reload tokenizer to save it
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(PATH,"data", "model"),
                local_files_only=True,
                device_map=device_map
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            model.save_pretrained(model_output_dir+"/data/model")
            tokenizer.save_pretrained(model_output_dir+"/data/tokenizer")
            os.environ["AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"] = "1800" #give time for model to be registered
            mlflow.pyfunc.log_model(code_paths=[model_output_dir], artifact_path=model_artifact_path, python_model=LAMA2Predict())
            model_uri = f"runs:/{run.info.run_id}/{model_artifact_path}/artifacts/code/{model_output_dir}"
            mlflow.register_model(model_uri, name = model_name,await_registration_for=1800)



if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)