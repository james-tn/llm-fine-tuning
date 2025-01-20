import mlflow
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
import torch
from transformers.integrations import MLflowCallback
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset
import os
import argparse
import math
import shutil


from transformers import TrainerCallback  
import mlflow  

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()


    # add arguments
    parser.add_argument("--mounted_data_folder", type=str)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--trained_model", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args


class LAMA2Predict(mlflow.pyfunc.PythonModel):
    #place holder. Need to implement later
    pass


class MlflowLoggingCallback(TrainerCallback):  
    def on_log(self, args, state, control, logs=None, **kwargs):  
        # Log metrics to MLflow  
        if logs is not None:  
            mlflow.log_metrics(logs, step=state.global_step)  
            mlflow.log_metric('epoch', state.epoch)  



def main(args):
    model_name = args.model_name
    learning_rate = args.learning_rate
    print("DATASET_MOUNT_CACHE_SIZE ", os.environ.get("DATASET_MOUNT_CACHE_SIZE"))
    print("content of the folder ", os.listdir(args.mounted_data_folder))
    trained_model = args.trained_model
    #save your train model to this folder to persist to job storage in cloud
    print("content of the trained_model folder ", os.listdir(args.trained_model))

    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
        "unsloth/llama-3-8b-Instruct-bnb-4bit",
        "unsloth/llama-3-70b-bnb-4bit",
        "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/mistral-7b-bnb-4bit",
        "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
    ] # More models at https://huggingface.co/unsloth

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3.3-70B-instruct-bnb-4bit", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",

                        "embed_tokens", "lm_head",], # Add for continual pretraining
        lora_alpha = 32,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = True,   # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    
    )

    # Wikipedia provides a title and an article text.
    # Use https://translate.google.com!
    _wikipedia_prompt = """Wikipedia Article
    ### Title: {}

    ### Article:
    {}"""
    # becomes:
    wikipedia_prompt = """위키피디아 기사
    ### 제목: {}

    ### 기사:
    {}"""

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_prompts_func(examples):
        titles = examples["title"]
        texts  = examples["text"]
        outputs = []
        for title, text in zip(titles, texts):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = wikipedia_prompt.format(title, text) + EOS_TOKEN
            outputs.append(text)
        return { "text" : outputs, }
    pass


    dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split = "train",)

    # We select 1% of the data to make training faster!
    dataset = dataset.train_test_split(train_size = 0.01)["train"]
    dataset = dataset.train_test_split(train_size = 0.9)
    train_dataset= dataset["train"]
    eval_dataset =dataset["test"]
    train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True,)


    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        callbacks=[MlflowLoggingCallback()],
        args = UnslothTrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 8,

            # Use warmup_ratio and num_train_epochs for longer runs!
            max_steps = 10,
            warmup_steps = 10,
            # warmup_ratio = 0.1,
            # num_train_epochs = 1,

            # Select a 2 to 10x smaller learning rate for the embedding matrices!
            learning_rate = learning_rate,
            embedding_learning_rate = 1e-5,

            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )
    # Define the directory path  
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")  
  
    def list_and_clean_directory(directory):  
        if not os.path.exists(directory):  
            print(f"The directory {directory} does not exist.")  
            return  
        
        # List all files and directories  
        for root, dirs, files in os.walk(directory):  
            for name in files:  
                file_path = os.path.join(root, name)  
                print(f"File: {file_path}")  
            
            for name in dirs:  
                dir_path = os.path.join(root, name)  
                print(f"Directory: {dir_path}")  
        
        # Remove all files and directories without confirmation  
        shutil.rmtree(directory)  
        print(f"All files and directories in {directory} have been removed.")  
    
    # Execute the function  
    list_and_clean_directory(cache_dir)  
    

    #@title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    trainer.remove_callback(MLflowCallback)
    model_artifact_path = "mlflow_model_folder"
    with mlflow.start_run() as run:
        trainer_stats = trainer.train()
        list_and_clean_directory("outputs")  

        # eval_results = trainer.evaluate()
        # mlflow.log_metric("perplexity",math.exp(eval_results['eval_loss']))
        # os.makedirs("data", exist_ok=True)  
        os.makedirs(args.trained_model+'/_unsloth_temporary_saved_buffers', exist_ok=True)

        model.save_pretrained_merged(args.trained_model+"/model", save_method = "merged_16bit",maximum_memory_usage = 0.4, temporary_location = args.trained_model+'/_unsloth_temporary_saved_buffers')
        print("done saving model!")
        tokenizer.save_pretrained(args.trained_model+"/tokenizer")


if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)