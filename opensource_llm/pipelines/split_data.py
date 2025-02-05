import json  
import random  
from collections import defaultdict  
  
# Load the JSONL data  
input_file = "sql_results.jsonl"  
with open(input_file, "r") as f:  
    data = [json.loads(line.strip()) for line in f]  
  
# Organize data by scenario for stratification  
scenario_buckets = defaultdict(list)  
for entry in data:  
    scenario_buckets[entry["scenario"]].append(entry)  
  
# Perform stratified split  
train_data = []  
test_data = []  
test_ratio = 0.2  # 20% test data  
  
for scenario, entries in scenario_buckets.items():  
    # Shuffle entries within each scenario bucket for randomness  
    random.shuffle(entries)  
      
    # Calculate split index  
    split_index = int(len(entries) * (1 - test_ratio))  
      
    # Split into train and test  
    train_data.extend(entries[:split_index])  
    test_data.extend(entries[split_index:])  
  
# Save the train and test data into separate JSONL files  
train_file = "sql_result_train.jsonl"  
test_file = "sql_result_test.jsonl"  
  
with open(train_file, "w") as train_f:  
    for entry in train_data:  
        train_f.write(json.dumps(entry) + "\n")  
  
with open(test_file, "w") as test_f:  
    for entry in test_data:  
        test_f.write(json.dumps(entry) + "\n")  
  
print(f"Data split complete. Train file: {train_file}, Test file: {test_file}")  