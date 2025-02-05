import os  
import pandas as pd  
import json  
import yaml  
from sklearn.model_selection import train_test_split  
from openai import AzureOpenAI  
from dotenv import load_dotenv  
  
# Load environment variables  
load_dotenv()  

from utils import create_aml_dataset  
  
# Load additional data files  
chapters_dict = json.load(open('data/hs_chapters.json'))
  
hscodes_df = pd.read_csv('data/hscodes.csv')  
hscodes_df['hscode'] = hscodes_df['hscode'].astype(str)  # Ensure it's a string for matching  
hscodes_dict = hscodes_df.set_index('hscode')['description'].to_dict()  
  
# Load dataset  
df = pd.read_csv('data/09.tsv', sep='\t')  
  
# Sample records  
def sample_records(df, num_records=10000, random_state=42):  
    return df.sample(n=min(num_records, len(df)), random_state=random_state)  
  
sampled_df = sample_records(df)  
sampled_df['HS6 actual'] = sampled_df['HS6 actual'].astype(str)

  
# Load prompt template  
with open('data_processing/osllm_prompt_template.yaml', 'r') as file:  
    prompt_template = yaml.safe_load(file)  
  
system_prompt = prompt_template['system_prompt']  
input_template = prompt_template['input_template']
output_template = prompt_template['output_template']
  
# Generate assistant answers  
def generate_assistant_answer(row):  
    hs6_actual = row['HS6 actual'].zfill(6)  # Ensure it's 6 digits  
  
    chapter_code = hs6_actual[:2]  
      
    heading_code = hs6_actual[:4]  # First 4 digits  
  
  
    assistant_answer = (  
        f"## Chapter: {chapter_code}\n"  
        f"#### Heading: {heading_code}\n"  
        f"#### Subheading: {hs6_actual}\n"  
    )  
    return output_template.format(answer=assistant_answer)  
  
# Generate JSONL data with labeled answers  
def generate_jsonl_data_with_labels(df, system_prompt, prompt_template):  
    jsonl_data = []  
    original_indices = []  # Store original indices here  
    for index, row in df.iterrows():  
        input = prompt_template['input_template'].format(  
            item_description=row['item name'],  
            category=row['category path']  
        )  
        assistant_answer = generate_assistant_answer(row)  
        if assistant_answer is None:
            continue

        # Create a list of prompt template elements
        
        parts = [system_prompt, prompt_template['instruction'], input,  assistant_answer] 

        # Join prompt template elements into a single string to create the prompt template
        formatted_prompt = "\n\n".join(parts)

        jsonl_data.append({"record":formatted_prompt})  
        original_indices.append(index)  # Store the original index  
  
    return jsonl_data, original_indices  
  
jsonl_data, original_indices = generate_jsonl_data_with_labels(sampled_df, system_prompt, prompt_template)  
  
# Split data into train, validation, and test sets  
train_data, remaining_data, train_indices, remaining_indices = train_test_split(  
    jsonl_data, original_indices, test_size=0.2, random_state=42  
)  
  
validation_data, test_data, validation_indices, test_indices = train_test_split(  
    remaining_data, remaining_indices, test_size=0.5, random_state=42  
)  
  
# Save JSONL data  
def save_jsonl(data, filename):  
    with open(filename, 'w') as f:  
        for record in data:  
            json.dump(record, f)  
            f.write('\n')  
  
save_jsonl(train_data, 'data/ossllm_train_data_short.jsonl')  
save_jsonl(validation_data, 'data/ossllm_validation_data_short.jsonl')  
save_jsonl(test_data, 'data/ossllm_test_data_short.jsonl')  
  
# Create mapping file  
def create_mapping_file(original_indices, output_path):  
    mapping = [{"jsonl_index": i, "original_index": idx} for i, idx in enumerate(original_indices)]  
    with open(output_path, 'w') as f:  
        json.dump(mapping, f, indent=2)  
  
create_mapping_file(train_indices, 'data/ossllm_mapping_to_train_short.json')  
create_mapping_file(validation_indices, 'data/ossllm_mapping_to_validation_short.json')  
create_mapping_file(test_indices, 'data/ossllm_mapping_to_test_short.json')  
  
# Save sampled DataFrame to CSV  
sampled_df.to_csv('data/ossllm_sample_data_short.csv', index=True)  
# Create the dataset in AML
create_aml_dataset('data/ossllm_train_data_short.jsonl', 'hscode_train_short_ds')
create_aml_dataset('data/ossllm_validation_data_short.jsonl', 'hscode_val_short_ds')
create_aml_dataset('data/ossllm_test_data_short.jsonl', 'hscode_test_short_ds')
