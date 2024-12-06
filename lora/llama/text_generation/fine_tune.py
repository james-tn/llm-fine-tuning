import os  
import torch  
import math  
import argparse  
import mlflow  
import pandas as pd  
import glob  

from datasets import Dataset  
from transformers import (  
    AutoModelForCausalLM,  
    AutoTokenizer,  
    BitsAndBytesConfig,  
    TrainingArguments,  
    TrainerCallback  
)  
import shutil  

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint, load_state_dict_from_zero_checkpoint  
import subprocess
from transformers.integrations import MLflowCallback  
from peft import LoraConfig, PeftModel  
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM  
  
def str2bool(v):  
    if isinstance(v, bool):  
        return v  
    if v.lower() in ('yes', 'true', 't', 'y', '1'):  
        return True  
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):  
        return False  
    else:  
        raise argparse.ArgumentTypeError('Boolean value expected.')  
  
def get_latest_checkpoint_tag(checkpoint_dir):  
    """  
    Retrieves the latest checkpoint tag by sorting the directory names.  
  
    :param checkpoint_dir: The path to the directory containing checkpoint subdirectories.  
    :return: The tag of the latest checkpoint.  
    """  
    # List all subdirectories in the checkpoint directory  
    subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]  
      
    if not subdirs:  
        raise FileNotFoundError("No checkpoint directories found in the specified directory.")  
      
    # Sort the directory names  
    subdirs.sort()  
    print("content of checkpoint_dir ", subdirs)
    print("content inside latest checkpoint ", os.listdir(os.path.join(checkpoint_dir, subdirs[-1])))
    # list the content of the directries under subdirs[-1]
    sub_subdirs = [d for d in os.listdir(os.path.join(checkpoint_dir, subdirs[-1])) if os.path.isdir(os.path.join(checkpoint_dir, subdirs[-1], d))]
    for sub_subdir in sub_subdirs:
        print("sub sub inside latest checkpoint ", os.listdir(os.path.join(checkpoint_dir, subdirs[-1], sub_subdir)))
   
  
    # Return the last directory in the sorted list, which should be the latest  
    latest_tag = subdirs[-1]  
    return latest_tag  
  
class MlflowLoggingCallback(TrainerCallback):  
    def on_log(self, args, state, control, logs=None, **kwargs):  
        if logs is not None:  
            mlflow.log_metrics(logs, step=state.global_step)  
            mlflow.log_metric('epoch', state.epoch)  
  
class EarlyStoppingCallback(TrainerCallback):  
    def __init__(self, early_stopping_patience=1, early_stopping_threshold=0.0):  
        self.early_stopping_patience = early_stopping_patience  
        self.early_stopping_threshold = early_stopping_threshold  
        self.best_metric = None  
        self.patience_counter = 0  
  
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  
        if metrics is not None and "eval_loss" in metrics:  
            current_metric = metrics["eval_loss"]  
            if self.best_metric is None or current_metric < self.best_metric - self.early_stopping_threshold:  
                self.best_metric = current_metric  
                self.patience_counter = 0  
                control.should_save = True  
                print(f"New best model found with eval_loss: {current_metric}")  
            else:  
                self.patience_counter += 1  
                print(f"No improvement in eval_loss for {self.patience_counter} evaluations.")  
                if self.patience_counter >= self.early_stopping_patience:  
                    control.should_training_stop = True  
                    print("Early stopping triggered.")  
  
def parse_args():  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--model_dir", type=str)  
    parser.add_argument("--epochs", type=int, default=1)  
    parser.add_argument("--max_steps", type=int, default=1000)  
    parser.add_argument("--trained_model", type=str, default="trained_model")  
    parser.add_argument("--ckp_dir", type=str, default="ckp_dir")  
    parser.add_argument("--train_dataset", type=str)  
    parser.add_argument("--val_dataset", type=str)  
    parser.add_argument("--train_batch_size", type=int, default=5)  
    parser.add_argument("--val_batch_size", type=int, default=5)  
    parser.add_argument("--lora_r", type=int, default=256)  
    parser.add_argument("--lora_alpha", type=int, default=16)  
    parser.add_argument("--lora_dropout", type=float, default=0.1)  
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16")  
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")  
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)  
    parser.add_argument("--max_seq_length", type=int, default=1280)  
    parser.add_argument("--max_grad_norm", type=float, default=0.3)  
    parser.add_argument("--learning_rate", type=float, default=2e-4)  
    parser.add_argument("--weight_decay", type=float, default=0.001)  
    parser.add_argument("--early_stopping_patience", type=int, default=3)  
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)  
    parser.add_argument("--chat_model", type=str2bool, nargs='?', const=True, default=False)  
    parser.add_argument("--deepspeed", type=str2bool, nargs='?', const=True, default=False)  
    parser.add_argument("--use_4bit", type=str2bool, nargs='?', const=True, default=True)  
    parser.add_argument("--use_nested_quant", type=str2bool, nargs='?', const=True, default=False)  
    parser.add_argument("--fp16", type=str2bool, nargs='?', const=True, default=False)  
    parser.add_argument("--bf16", type=str2bool, nargs='?', const=True, default=True)  
    parser.add_argument("--gradient_checkpointing", type=str2bool, nargs='?', const=True, default=True)  
    parser.add_argument("--packing", type=str2bool, nargs='?', const=True, default=False)  
  
    return parser.parse_args()  
  
def main(args):  
    PATH = args.model_dir  
    trained_model = args.trained_model  
    ckp_dir = args.ckp_dir
    chat_model = args.chat_model  
    deepspeed = args.deepspeed  
    train_dataset = args.train_dataset  
    val_dataset = args.val_dataset  
    rank = int(os.environ.get('RANK', 0))  
  
    print("rank ", rank)  
  
    new_model = "fine_tuned_model"  
    lora_r = args.lora_r  
    lora_alpha = args.lora_alpha  
    lora_dropout = args.lora_dropout  
    use_4bit = args.use_4bit  
    bnb_4bit_compute_dtype = args.bnb_4bit_compute_dtype  
    bnb_4bit_quant_type = args.bnb_4bit_quant_type  
    use_nested_quant = args.use_nested_quant  
    fp16 = args.fp16  
    bf16 = args.bf16  
    per_device_train_batch_size = args.train_batch_size  
    per_device_eval_batch_size = args.val_batch_size  
    gradient_accumulation_steps = args.gradient_accumulation_steps  
    gradient_checkpointing = args.gradient_checkpointing  
    num_train_epochs = args.epochs  
  
    output_dir = ckp_dir  
    max_grad_norm = args.max_grad_norm  
    learning_rate = args.learning_rate  
    weight_decay = args.weight_decay  
    optim = "paged_adamw_32bit"  
    lr_scheduler_type = "constant"  
    max_steps = args.max_steps  
    warmup_ratio = 0.03  
    group_by_length = True  
    save_steps=8  # Adjust based on your needs  
    logging_steps = 5  
    max_seq_length = args.max_seq_length  
    packing = args.packing  
  
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))  
    device_map = {'': local_rank}  
    # print("device map ", device_map)  
  
    train_data_dict = pd.read_json(train_dataset, lines=True).to_dict(orient="records")  
    train_records = [item["record"] for item in train_data_dict]  
    train_ds = Dataset.from_dict({"record": train_records})  
  
    val_data_dict = pd.read_json(val_dataset, lines=True).to_dict(orient="records")  
    val_records = [item["record"] for item in val_data_dict]  
    val_ds = Dataset.from_dict({"record": val_records})  
  
    def formatting_prompts_func(example):  
        return example['record']  
  
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)  
    bnb_config = BitsAndBytesConfig(  
        load_in_4bit=use_4bit,  
        bnb_4bit_quant_type=bnb_4bit_quant_type,  
        bnb_4bit_compute_dtype=compute_dtype,  
        bnb_4bit_use_double_quant=use_nested_quant,  
    )  
  
    model = AutoModelForCausalLM.from_pretrained(  
        os.path.join(PATH, "data", "model"),  
        low_cpu_mem_usage=True,  
        local_files_only=True,  
        quantization_config=bnb_config,  
        device_map=device_map  
    )  
  
    model.config.use_cache = False  
    model.config.pretraining_tp = 1  
  
    tokenizer = AutoTokenizer.from_pretrained(  
        os.path.join(PATH, "data", "model"),  
        local_files_only=True,  
        device_map=device_map  
    )  
    tokenizer.pad_token = tokenizer.eos_token  
    tokenizer.padding_side = "right"  
  
    peft_config = LoraConfig(  
        lora_alpha=lora_alpha,  
        lora_dropout=lora_dropout,  
        r=lora_r,  
        bias="none",  
        task_type="CAUSAL_LM",  
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],  
    )  
  
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
        deepspeed="deepspeed_configs/ds_config.json" if deepspeed else None,  
    )  
    training_arguments.ddp_find_unused_parameters = False  
    response_template = "### Output:"  
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)  
  
    early_stopping_callback = EarlyStoppingCallback(  
        early_stopping_patience=args.early_stopping_patience,  
        early_stopping_threshold=args.early_stopping_threshold  
    )  
  
    trainer = SFTTrainer(  
        model=model,  
        train_dataset=train_ds,  
        eval_dataset=val_ds,  
        peft_config=peft_config,  
        callbacks=[MlflowLoggingCallback()],  
        max_seq_length=max_seq_length,  
        tokenizer=tokenizer,  
        args=training_arguments,  
        packing=packing,  
        formatting_func=formatting_prompts_func,  
    )  
    trainer.remove_callback(MLflowCallback)  
  
    with mlflow.start_run() as run:  
        trainer.train() 
        print("done with training")
        dest_path = os.path.join(trained_model, 'lora')
        os.makedirs(dest_path, exist_ok=True)
 
  
        if rank == 0:  
            # Clear memory  
            if not deepspeed:  
                # Save the model
                trainer.model.save_pretrained(dest_path)
                tokenizer.save_pretrained(dest_path)    
                return

            del model  
            del trainer  
            import gc  
            gc.collect()  
            ckp_tag = get_latest_checkpoint_tag(ckp_dir)
            
            ckp_dir = os.path.join(ckp_dir, ckp_tag)
            subprocess.run(
                ['python', 'zero_to_fp32.py', '.', ckp_dir],
                cwd=ckp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            #copy from the converted_fp32 to the trained_model
            # Copy all bin files from the converted_fp32 to the trained_model  
            bin_files_pattern = os.path.join(ckp_dir, "pytorch_model-*.bin")  
            for fp32_output_path in glob.glob(bin_files_pattern):  
                shutil.copy(fp32_output_path, dest_path)  
            # Copy config.json and tokenizer files to dest_path
            model_files = ['adapter_config.json','adapter_model.safetensors','tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']

            for file_name in model_files:
                src_file = os.path.join(ckp_dir, file_name)
                if os.path.exists(src_file):
                    shutil.copy(src_file, dest_path)
                else:
                    print(f"Warning: {file_name} not found in {ckp_dir}")

                

        else:  
            print("at rank ", rank)  
  
if __name__ == "__main__":  
    args = parse_args()  
    main(args)  