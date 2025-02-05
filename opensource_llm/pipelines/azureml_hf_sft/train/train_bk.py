import os  
import torch  
import math  
import argparse  
import mlflow  
import pandas as pd  
import glob  
from datasets import Dataset  
import transformers
import packaging.version
from transformers import (  
    AutoModelForCausalLM,  
    AutoTokenizer,  
    BitsAndBytesConfig,  
    TrainingArguments,  
    TrainerCallback,  
    Trainer  
)  
import shutil  
import logging  
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint, load_state_dict_from_zero_checkpoint  
import subprocess  
from transformers.integrations import MLflowCallback, is_deepspeed_zero3_enabled  
from peft import LoraConfig, PeftModel  
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig  
  
  
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
    """Retrieves the latest checkpoint tag by sorting the directory names."""  
    subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]  
    if not subdirs:  
        raise FileNotFoundError("No checkpoint directories found in the specified directory.")  
    subdirs.sort()  
    logging.info("content of checkpoint_dir: %s", subdirs)  
    logging.info("content inside latest checkpoint: %s", os.listdir(os.path.join(checkpoint_dir, subdirs[-1])))  
    sub_subdirs = [d for d in os.listdir(os.path.join(checkpoint_dir, subdirs[-1])) if os.path.isdir(os.path.join(checkpoint_dir, subdirs[-1], d))]  
    for sub_subdir in sub_subdirs:  
        logging.info("sub sub inside latest checkpoint: %s", os.listdir(os.path.join(checkpoint_dir, subdirs[-1], sub_subdir)))  
    latest_tag = subdirs[-1]  
    return latest_tag  
  
  
class MlflowLoggingCallback(TrainerCallback):  
    def on_log(self, args, state, control, logs=None, **kwargs):  
        if logs is not None:  
            try:  
                mlflow.log_metrics(logs, step=state.global_step)  
                if hasattr(state, 'epoch'):  
                    mlflow.log_metric('epoch', state.epoch)  
            except Exception as e:  
                logging.error("Failed to log metrics to mlflow", exc_info=True)  
  
  
class EarlyStoppingCallback(TrainerCallback):  
    def __init__(self, early_stopping_patience=3, early_stopping_threshold=0.0):  
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
                logging.info(f"New best model found with eval_loss: {current_metric}")  
            else:  
                self.patience_counter += 1  
                logging.info(f"No improvement in eval_loss for {self.patience_counter} evaluations.")  
                if self.patience_counter >= self.early_stopping_patience:  
                    control.should_training_stop = True  
                    logging.info("Early stopping triggered.")  
  
  
def parse_args():  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--model_dir", type=str)  
    parser.add_argument("--epochs", type=int, default=1)  
    parser.add_argument("--max_steps", type=int)  
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
    parser.add_argument("--bnb_4bit_quant_storage_dtype", type=str, default="uint8")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)  
    parser.add_argument("--max_seq_length", type=int, default=1280)  
    parser.add_argument("--max_grad_norm", type=float, default=0.3)  
    parser.add_argument("--learning_rate", type=float, default=2e-4)  
    parser.add_argument("--weight_decay", type=float, default=0.001)  
    parser.add_argument("--early_stopping_patience", type=int, default=3)  
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)  
    parser.add_argument("--chat_model", type=str2bool, nargs='?', const=True, default=False)  
    parser.add_argument("--use_4bit_quantization", type=str2bool, nargs='?', const=True, default=True) 
    parser.add_argument("--use_8bit_quantization", type=str2bool, nargs='?', const=True, default=False) 
    parser.add_argument("--enable_deepspeed", type=str2bool, nargs='?', const=True, default=False)  
    parser.add_argument("--use_nested_quant", type=str2bool, nargs='?', const=True, default=False)  
    parser.add_argument("--fp16", type=str2bool, nargs='?', const=True, default=False)  
    parser.add_argument("--bf16", type=str2bool, nargs='?', const=True, default=True)  
    parser.add_argument("--gradient_checkpointing", type=str2bool, nargs='?', const=True, default=True)  
    parser.add_argument("--packing", type=str2bool, nargs='?', const=True, default=False)  
    parser.add_argument("--use_flash_attn", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--use_lora", type=str2bool, nargs='?', const=True, default=True, help="Use LoRA for parameter-efficient fine-tuning.")  
    parser.add_argument("--deepspeed_config", type=str, default="configs/ds_config.json", help="Path to deepspeed config file.")  
    parser.add_argument("--target_modules", nargs='+', default=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"], help="Target modules for LoRA.")  
    parser.add_argument("--enable_quantization", type=str2bool, nargs='?', const=True, default=False, help="Enable quantization for model loading.")  
    parser.add_argument("--verbosity", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help="Set the logging verbosity level.")  
    return parser.parse_args()  
  
  
def main(args):  
    print("args.bf16", args.bf16)
    logging.basicConfig(level=getattr(logging, args.verbosity.upper()))  
  
    PATH = args.model_dir  
    trained_model = args.trained_model  
    ckp_dir = args.ckp_dir  
    logging.info("deepspeed enabled: %s", args.enable_deepspeed)  
    train_dataset = args.train_dataset  
    val_dataset = args.val_dataset  
    rank = int(os.environ.get('RANK', 0))  
    logging.info("rank: %d", rank)  
  
    new_model = "fine_tuned_model"  
    per_device_train_batch_size = args.train_batch_size  
    per_device_eval_batch_size = args.val_batch_size  
    gradient_accumulation_steps = 16  
    num_train_epochs = args.epochs  
  
    output_dir = ckp_dir  
    max_grad_norm = args.max_grad_norm  
    learning_rate = args.learning_rate  
    weight_decay = args.weight_decay  
    optim = "paged_adamw_32bit"  
    lr_scheduler_type = "linear"  
    max_steps = args.max_steps  
    warmup_ratio = 0.03  
    group_by_length = True  
    save_steps = 200  
    logging_steps = 10  
    packing = args.packing  
  
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))  
    device_map = {'': local_rank}  
    logging.info("device map: %s", device_map)  
  
    train_data_dict = pd.read_json(train_dataset, lines=True).to_dict(orient="records")  
    train_records = [item["record"] for item in train_data_dict]  
    train_ds = Dataset.from_dict({"record": train_records})  
  
    val_data_dict = pd.read_json(val_dataset, lines=True).to_dict(orient="records")  
    val_records = [item["record"] for item in val_data_dict]  
    val_ds = Dataset.from_dict({"record": val_records})  
  
    def formatting_prompts_func(example):  
        return example['record']  
    load_best_model_at_the_end = None if (args.enable_deepspeed and args.use_lora) else True
    logging.info("load_best_model_at_the_end  %s", load_best_model_at_the_end)

    bnb_config = None
    quant_storage_dtype = None  
    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)
    torch_dtype = (
    quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
)   
    # Define training arguments  
    load_best_model_at_the_end = None if (args.enable_deepspeed and args.use_lora) else True
    logging.info("load_best_model_at_the_end  %s", load_best_model_at_the_end)
    training_arguments = SFTConfig(  
        output_dir=output_dir,  
        num_train_epochs=num_train_epochs,  
        per_device_train_batch_size=per_device_train_batch_size,  
        per_device_eval_batch_size=per_device_eval_batch_size,  
        gradient_accumulation_steps=gradient_accumulation_steps,  
        optim=optim,  
        save_steps=save_steps if max_steps != -1 else None,  
        save_strategy="steps" if max_steps != -1 else "epoch",  
        logging_steps=logging_steps,  
        learning_rate=learning_rate,  
        weight_decay=weight_decay,  
        fp16=args.fp16,
        bf16=args.bf16,  
        max_grad_norm=max_grad_norm,  
        max_steps=max_steps if max_steps else None,  
        warmup_ratio=warmup_ratio,  
        group_by_length=group_by_length,  
        lr_scheduler_type=lr_scheduler_type,  
        deepspeed=args.deepspeed_config if args.enable_deepspeed else None,  
        load_best_model_at_end=load_best_model_at_the_end,  
        eval_steps=logging_steps * 20,  
        eval_strategy="epoch" if max_steps == -1 else "steps",  
        max_seq_length=args.max_seq_length,  
        packing=args.packing,
        ddp_timeout= 7200
    )  
    if args.enable_quantization and args.use_lora:  
        logging.info("Using quantization for model loading.")  
  
  
    print("torch_dtype ", torch_dtype)
    uses_fsdp = os.environ.get("ACCELERATE_USE_FSDP", "none").lower() == "true"
    print("is_deepspeed_zero3_enabled(): ", is_deepspeed_zero3_enabled())  
    print("uses_fsdp  ", uses_fsdp)

    # Load the model  

    model = AutoModelForCausalLM.from_pretrained(  
        os.path.join(PATH, "data", "model"),  
        quantization_config=bnb_config if args.enable_quantization and args.use_lora else None,  
        device_map=device_map if ((not is_deepspeed_zero3_enabled()) and (not uses_fsdp)) else None,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            torch_dtype=torch_dtype,
    )  
    # Apply LoRA if specified
    peft_config = None  
    if args.use_lora:  
        print("Using LoRA for parameter-efficient fine-tuning, target modules:", args.target_modules)
        peft_config = LoraConfig(  
            lora_alpha=args.lora_alpha,  
            lora_dropout=args.lora_dropout,  
            r=args.lora_r,  
            bias="none",  
            task_type="CAUSAL_LM",  
            target_modules=args.target_modules,  
        )  
        model = PeftModel(model, peft_config)  
    else:
        print("Training using full model's weight.")
  
    model.config.use_cache = False  
    model.config.pretraining_tp = 1  
  
    tokenizer = AutoTokenizer.from_pretrained(  
        os.path.join(PATH, "data", "model"),  
        local_files_only=True,  
        device_map=device_map  
    )  
    tokenizer.pad_token = tokenizer.eos_token  
    # make embedding resizing configurable?
    # Transformers 4.46.0+ defaults uses mean_resizing by default, which fails with QLoRA + FSDP because the
    # embedding could be on meta device, therefore, we set mean_resizing=False in that case (i.e. the status quo
    # ante). See https://github.com/huggingface/accelerate/issues/1620.
    uses_transformers_4_46 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.46.0")

    if (bnb_config is not None) and uses_fsdp and uses_transformers_4_46:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)
    else:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    early_stopping_callback = EarlyStoppingCallback(  
        early_stopping_patience=args.early_stopping_patience,  
        early_stopping_threshold=args.early_stopping_threshold  
    )  
  
    trainer = SFTTrainer(  
        model=model,  
        train_dataset=train_ds,  
        eval_dataset=val_ds,  
        callbacks=[MlflowLoggingCallback(), early_stopping_callback],  
        tokenizer=tokenizer,  
        args=training_arguments,  
        formatting_func=formatting_prompts_func,  
    )  
  
    trainer.remove_callback(MLflowCallback)  
  
    with mlflow.start_run() as run:  
        trainer.train()  
        logging.info("done with training")  
        dest_path = os.path.join(trained_model, 'model')  
        os.makedirs(dest_path, exist_ok=True)  
        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
            trainer.save_model(dest_path)
            tokenizer.save_pretrained(dest_path)  
            return  
  
        if rank == 0:  

            if not args.enable_deepspeed:  
                trainer.model.save_pretrained(dest_path)  
                tokenizer.save_pretrained(dest_path)  
                return  
  
            del model  
            del trainer  
            import gc  
            gc.collect()  
            ckp_tag = get_latest_checkpoint_tag(ckp_dir)  
  
            ckp_dir = os.path.join(ckp_dir, ckp_tag)  
            try:  
                subprocess.run(  
                    ['python', 'zero_to_fp32.py', '.', ckp_dir],  
                    cwd=ckp_dir,  
                    stdout=subprocess.PIPE,  
                    stderr=subprocess.PIPE,  
                    text=True,  
                    check=True  
                )  
            except subprocess.CalledProcessError as e:  
                logging.error("Failed to convert zero to fp32: %s", e.stderr)  

            except Exception as e:  
                logging.error("Failed to convert zero to fp32", exc_info=True)  
  
            bin_files_pattern = os.path.join(ckp_dir, "pytorch_model-*.bin")  
            for fp32_output_path in glob.glob(bin_files_pattern):  
                shutil.copy(fp32_output_path, dest_path)  
  
            model_files = ['config.json','adapter_config.json', 'adapter_model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'pytorch_model.bin.index.json']  
            for file_name in model_files:  
                src_file = os.path.join(ckp_dir, file_name)  
                if os.path.exists(src_file):  
                    shutil.copy(src_file, dest_path)  
                else:  
                    logging.info(f"Warning: {file_name} not found in {ckp_dir}")  
  
        else:  
            logging.info("at rank: %d", rank)  
  
if __name__ == "__main__":  
    args = parse_args()  
    main(args)  