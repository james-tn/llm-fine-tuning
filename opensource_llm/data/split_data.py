import json  
import random  
  
# Set the file paths  
input_file = 'sql_result_v5.jsonl'  
train_file = 'sql_result_train_v5.jsonl'  
test_file = 'sql_result_test_v5.jsonl'  
  
# Define the split ratio  
train_ratio = 0.8  # 80% training, 20% testing  
  
# Read the data from the input file  
data = []  
with open(input_file, 'r') as f:  
    for line in f:  
        data.append(json.loads(line.strip()))  
  
# Shuffle the data randomly  
random.seed(42)  # Set a seed for reproducibility  
random.shuffle(data)  
  
# Split the data into training and testing sets  
split_index = int(len(data) * train_ratio)  
train_data = data[:split_index]  
test_data = data[split_index:]  
  
# Write the training data to a new file  
with open(train_file, 'w') as f:  
    for record in train_data:  
        f.write(json.dumps(record) + '\n')  
  
# Write the testing data to a new file  
with open(test_file, 'w') as f:  
    for record in test_data:  
        f.write(json.dumps(record) + '\n')  
  
print(f"Data split complete. Training data: {len(train_data)} records, Testing data: {len(test_data)} records.")  