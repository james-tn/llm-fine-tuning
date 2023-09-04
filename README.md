# llm-fine-tuning
Examples of fine-tuning LLMs and deployment using Azure ML distributed compute


Instruction:
1. Setup your Azure ML CLI v2: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public
2. Make sure you have A100 GPU SKU (NCadsA100 or NDadsA100) series
3. Go to llma2 folder Run the training script: ``` az ml job create finetune_pipeline.yml```

Credit: 
This repo uses training data from from https://github.com/tatsu-lab/stanford_alpaca/tree/main
