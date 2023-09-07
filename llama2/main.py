import os
import torch
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
    
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import json
from datasets import Dataset

#Function to load model for scoring 

pipeline_artifact_name = "pipeline"


class LAMA2Predict(mlflow.pyfunc.PythonModel):
  def __init__(self, model_name):
    self.model_name = model_name
  def load_context(self, context):
    device_map = {"": 0}
    artifact_path = f"{self.model_name}/artifacts/trained_model"
    model = AutoModelForCausalLM.from_pretrained(
        artifact_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        artifact_path,
        local_files_only=True,
        device_map=device_map
    )
    self.pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    
  def predict(self, context, model_input): 
    texts = model_input['text']
    max_length = model_input['max_length']
    result = self.pipeline(texts, max_new_tokens=max_length)
    return result

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()


    # add arguments
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_examples", type=int, default=1000)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--trained_model", type=str, default="trained_model")
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
    per_device_train_batch_size = 4

    # Batch size per GPU for evaluation
    per_device_eval_batch_size = 4

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

    # Load the entire model on the GPU 0
    # device_map = {"": 0}
    # device_map={'':torch.cuda.current_device()}
    # print("device map 1", device_map)
    # device_map = "auto"
    # device_map={'':torch.xpu.current_device()}
    # Load dataset (you can process it here)

    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}
    print("device map ", device_map)


    with open("data/alpaca_data.json", "r") as file:    
        list_data_dict = json.load(file)
    list_data_dict= list_data_dict[:num_examples]
    instructions = [item["instruction"] for item in list_data_dict]
    inputs = [item["input"] for item in list_data_dict]
    outputs = [item["output"] for item in list_data_dict]


    dataset = Dataset.from_dict({"instruction":instructions, "input": inputs, "output":outputs})   

    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:{output}"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:{output}"
        ),
    }

    def formatting_prompts_func(example):
        example_input, example_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        output_texts = []
        for i in range(len(example['input'])):
            if example['input'][i] == "":
                text =  example_no_input.format_map({"instruction":example["instruction"][i], "output":example["output"][i]})
            else:
                text = example_input.format_map({"instruction":example["instruction"][i], "input":example["input"][i], "output":example["output"][i]})

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

    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
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
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
        # data_collator=data_collator,
        formatting_func=formatting_prompts_func,

    )
    
    with mlflow.start_run() as run:
        trainer.train()
        import math
        eval_results = trainer.evaluate()
        mlflow.log_metric("perplexity",math.exp(eval_results['eval_loss']))
        trainer.model.save_pretrained(new_model)
        model_output_dir = "trained_model"
        model_artifact_path = model_name
        # Empty VRAM
        del model
        # del pipe
        del trainer
        import gc
        gc.collect()
        gc.collect()
        if rank==0:

            # Reload model in FP16 and merge it with LoRA weights
            base_model = AutoModelForCausalLM.from_pretrained(
                os.path.join(PATH,"data", "model"),
                local_files_only=True,
                low_cpu_mem_usage=True,
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
            model.save_pretrained(model_output_dir)
            tokenizer.save_pretrained(model_output_dir)
            mlflow.pyfunc.log_model(artifacts={pipeline_artifact_name: model_output_dir}, artifact_path=model_artifact_path, python_model=LAMA2Predict(model_name))
            model_uri = f"runs:/{run.info.run_id}/{model_artifact_path}"
            mlflow.register_model(model_uri, name = model_name,await_registration_for=1800)



if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)