# Fine-Tuning SQL Query Generation for Business Analytics  
  
This repository provides a complete, end-to-end solution for fine-tuning language models to generate SQL queries for business analytics. The project covers the full lifecycle—from data generation based on an ontology, through fine-tuning on domain-specific examples, deployment of the customized model, and evaluation of the deployed model.  
  
## Overview  
  
In today’s competitive business environment, analyzing data quickly and accurately is essential. This project demonstrates how you can:  
  
1. **Generate training data**: Leverage a knowledge graph/ontology to auto-generate diverse business questions and corresponding SQL queries.  
2. **Fine-tune a base model**: Use Azure OpenAI to fine-tune a pre-trained GPT model on these domain-specific SQL examples.  
3. **Deploy the fine-tuned model**: Deploy the customized model using Azure’s control plane API.  
4. **Evaluate performance**: Assess model accuracy by comparing generated SQL outputs (with or without reasoning) to ground truth answers.  
  
Users may choose to work with pre-generated data (in the `data/` folder) or run the full data-generation pipeline.  
  
---  
  
## Business Scenario  
  
The underlying business scenario focuses on business analytics for a typical retail or logistics company. An ontology file (e.g., `analytic_graph.json`) defines key business concepts such as:  
  
- **Gross Profit**  
- **Freight to Sales Ratio**  
- **Late Shipment Rate**  
- **Regional Freight Distribution**  
- **Multi-Item Order Percentage**  
- **Business Account**, and many more.  
  
Each concept includes details like its description, formulas, associated tables, and time dependencies. This ontology drives:  
  
1. **Business question generation**: Create a set of unique, well-formed questions that are answerable by a single SQL query.  
2. **SQL query generation**: Generate correct SQL queries that answer the generated questions using the provided database schema (the Northwind database in this example).  
  
---  
  
## Repository Structure & Pipeline  
  
Below is a high-level overview of the key components:  
  
### **Data Generation (`data_gen_program.py`)**  
  
1. **Question Generation**: Uses the GPT-based question generation client to create unique questions based on the ontology (`analytic_graph.json`).  
2. **SQL Query Generation**: Uses a shared prompt (derived from the deepseek prompt) to generate SQL queries with either the `o3` model or the `deepseek` model. Optionally, the model can include reasoning.  
3. **Packaging & Splitting**: Wraps every question–SQL pair into an OpenAI-style conversation record and splits the dataset into 80% training, 10% testing, and 10% inference JSONL files.  
  
### **Fine-Tuning (`ft_job_and_deploy.py`)**  
  
1. **File Upload**: Uploads the training (e.g., `training_set.jsonl`) and validation (e.g., `validation_set.jsonl`) files to Azure OpenAI.  
2. **Job Creation**: Starts a fine-tuning job on a base model (for example, `gpt-35-turbo-0125`).  
3. **Monitoring**: Polls the fine-tuning job status, lists events and checkpoints, and downloads the results (e.g., `results.csv`) for analysis.  
  
### **Deployment**  
  
Once the fine-tuning job succeeds, a deployment script uses the Azure management (control plane) API to deploy the fine-tuned model. This deployment step takes a fine-tuned model ID (or checkpoint ID) and creates a new endpoint (e.g., `gpt-35-turbo-ft`) for inference.  
  
### **Evaluation**  
  
1. **Compares the deployed models’ generated SQL queries with ground truth SQL.**  
2. **Supports both deployments with reasoning and without reasoning.**  
3. **Uses a two-stage evaluation**:  
   - Execute both predicted and ground truth SQL queries on a local SQLite database (Northwind).  
   - Apply a custom logic and even call a “SQL judge” service (via another Azure OpenAI deployment) to determine if the predicted results are adequate.  
4. **Accuracy is computed over the test set.**  
  
---  
  
## How to Use the Repository  
  
### **Option 1: Use Pre-generated Data**  
  
If you prefer not to run the data-generation pipeline yourself, you can use the pre-generated training, testing, and inference files available in the `data/` folder. Then proceed directly to fine-tuning and evaluation.  
  
---  
  
### **Option 2: Generate Your Own Data**  
  
#### Configure Environment Variables:  
Create a `.env` file with all necessary keys and endpoints. You will need:  
  
- Azure OpenAI endpoints and API keys for both the question generation and SQL generation (e.g., `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, and additional model-specific variables like `AZURE_OPENAI_O3_API_KEY`, etc.).  
- Settings for the SQL Judge deployment (`SQL_JUDGE_DEPLOYMENT`, `SQL_JUDGE_ENDPOINT`, `SQL_JUDGE_API_KEY`, `SQL_JUDGE_API_VERSION`).  
  
#### Run Data Generation:  
Execute the data generation script:  
  
```bash  
python data_gen_program.py --num_questions 50 --sql_model deepseek --include_reasoning  
```
This script will generate unique business questions, create corresponding SQL statements (with your chosen model and reasoning flag), and split them into training, testing, and inference JSONL files.  
  
#### Review the Generated Data:  
Check the generated files (e.g., `open_ai_data_train.jsonl`, `open_ai_data_test.jsonl`, `open_ai_data_inference.jsonl`) to ensure everything is as expected.  
  
---  
  
### **Fine-Tuning the Model**  
  
#### Upload Files & Create Fine-Tuning Job:  
Run the fine-tuning script:  
  
```bash  
python ft_job_and_deploy.py 
``` 
This script:  
  
1. Uploads your `training_set.jsonl` and `validation_set.jsonl` files.  
2. Creates a fine-tuning job with the chosen base model.  
3. Monitors the job until it completes (or fails).  
4. Downloads the fine-tuning result file (e.g., `results.csv`).  
  
#### Deploy the Fine-Tuned Model:  
If the job succeeds, the script automatically deploys your fine-tuned model. Make sure your management token (e.g., `AZURE_MANAGEMENT_TOKEN`) is set properly in your environment.  
  
---  
  
### **Evaluating the Deployed Model**  
  
Use the provided evaluation code to assess the performance of your deployed models. The evaluation code:  
  
1. Loads the test dataset (from, e.g., `sql_result_test.jsonl`).  
2. Queries the deployed model(s) (with and without reasoning).  
3. Compares the predicted SQL query results with the ground truth by executing the SQL and, if needed, using a judge service.  
4. Computes and prints the accuracy for each deployment.  
  
To run evaluation:  
  
```bash  
python evaluate_fine_tuned_model.py  
```
## Pre-run Test Results  
  
During our internal tests, we observed the following accuracy results (accuracy is defined as the percentage of test items where the SQL result met the evaluation criteria):  
  
- **Accuracy for `gpt-4o-mini-2024-07-18-no_reasoning`**: 75.40%  
- **Accuracy for `gpt-4o`**: 71.43%  
- **Accuracy for `gpt-4o-mini-2024-07-18-reasoning`**: 78.57%  
- **Accuracy for `gpt-4o-mini`**: 69.05%  
  
These results indicate a reasonable performance with slight variations based on the model deployment and whether reasoning was used.  
  
---  
  
## Conclusion  
  
This repository provides a complete practical example for:  
  
1. Generating fine-tuning data for domain-specific SQL query generation.  
2. Fine-tuning an Azure OpenAI model.  
3. Deploying a customized model using the Azure management API.  
4. Evaluating the performance of the deployed model.  
  
Whether you use the pre-generated data or generate your own from the provided ontology, this pipeline demonstrates how to build, fine-tune, and assess a customized SQL query generator for modern business analytics.  
  
Feel free to contribute, adjust the hyperparameters for fine-tuning, or extend the evaluation logic as needed.  
  
**Happy fine-tuning!**  