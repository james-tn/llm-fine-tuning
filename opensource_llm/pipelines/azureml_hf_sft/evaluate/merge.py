import os  
import logging  
import torch  
import gc  
import argparse  
from transformers import AutoTokenizer, AutoModelForCausalLM  
from peft import PeftModel  
  
def merge_lora_and_save(model_dir, trained_model_path, merged_model_dir, use_lora):  
    """  
    Merge LoRA adapters into the base model and save the merged model to the output path.  
    If use_lora is False, do nothing and return.  
    """  
    if not use_lora:  
        logging.info("use_lora is set to False. Skipping LoRA merging.")  
        return  
  
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
  
    # Save the merged model  
    logging.info(f"Saving merged model to: {merged_model_dir}")  
    model.save_pretrained(merged_model_dir)  
    tokenizer = AutoTokenizer.from_pretrained(  
        trained_model_path,  
        local_files_only=True,  
        device_map="auto"  
    )  
    tokenizer.save_pretrained(merged_model_dir)  
  
    # Cleanup  
    del model  
    del tokenizer  
    gc.collect()  
    torch.cuda.empty_cache()  
  
    logging.info("Merged model saved successfully.")  
  
def main():  
    """  
    Main method to parse arguments and execute the LoRA merging logic.  
    """  
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into a base model and save the merged model.")  
      
    # Define required arguments  
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the base model directory.")  
    parser.add_argument("--trained_model_path", type=str, required=True, help="Path to the trained LoRA adapter.")  
    parser.add_argument("--merged_model_dir", type=str, required=True, help="Directory to save the merged model.")  
    parser.add_argument("--use_lora", type=bool, required=True, help="Whether to use LoRA merging (True or False).")  
      
    # Parse the arguments  
    args = parser.parse_args()  
      
    # Configure logging  
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")  
    trained_model_path = os.path.join(args.trained_model_path, "model")  

    # Execute the LoRA merging logic  
    merge_lora_and_save(  
        model_dir=args.model_dir,  
        trained_model_path=trained_model_path,  
        merged_model_dir=args.merged_model_dir,  
        use_lora=args.use_lora  
    )  
  
if __name__ == "__main__":  
    main()  