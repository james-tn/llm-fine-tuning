# llm-fine-tuning
Examples of fine-tuning LLMs and deployment using Azure ML distributed compute (Multiple GPUs & Multiple nodes)


Instruction:
1. Setup your Azure ML CLI v2: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public
2. Make sure you have A100 GPU SKU (NCadsA100 or NDadsA100) series
3. Go to llma2 folder Run the training script: ``` az ml job create finetune_pipeline.yml```
This job fine-tunes pretrained models (LLAMA2-7B, LLAMA2_13B or LLAMA2-70B, including Chat models) from Azure ML's model registry using Hugging Face's SFT library. Azure ML distributed DL infrastructure allow easy scaling out for large scale training.
The fine-tuned model is registered in MLFLow format to Azure ML.
Use the test.ipynb notebook to test the fine-tuned model.
Checkout ```finetine_pipeline.yml``` for training parameters.

Credit: 
This repo uses training data from from https://github.com/tatsu-lab/stanford_alpaca/tree/main
