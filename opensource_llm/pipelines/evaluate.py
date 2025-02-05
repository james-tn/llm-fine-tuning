import json
import sqlite3
from pathlib import Path
import mlflow
from vllm import LLM, SamplingParams
import regex as re
import torch

def extract_sql_from_response(text):
    """Extracts SQL query from markdown code blocks"""
    pattern = r"```sql(.*?)```"
    matches = re.search(pattern, text, re.DOTALL)
    return matches.group(1).strip() if matches else text.strip()

def compare_sql_results(predicted_text, ground_truth_query, db_path):
    # Extract SQL from both predicted and ground truth
    predicted_sql = extract_sql_from_response(predicted_text)
    ground_truth_sql = extract_sql_from_response(ground_truth_query)
    
    def execute_query(sql):
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                result = cursor.fetchall()
                return sorted([str(item) for sublist in result for item in sublist])
        except Exception as e:
            print(f"Error executing SQL: {e}\nQuery: {sql}")
            return None
    
    pred_result = execute_query(predicted_sql)
    truth_result = execute_query(ground_truth_sql)
    
    return {
        "predicted_sql": predicted_sql,
        "ground_truth_sql": ground_truth_sql,
        "results_match": pred_result == truth_result if pred_result and truth_result else False,
        "execution_success": bool(pred_result and truth_result)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--db_path", type=str, default="northwind.db")
    parser.add_argument("--results_path", type=str, default="results.json")
    args = parser.parse_args()

    # Initialize LLM
    llm = LLM(
        model=args.model_dir,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype="bfloat16"
    )

    # Load test data
    with open(args.test_dataset) as f:
        test_data = [json.loads(line) for line in f]

    # Evaluation parameters
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=512,
        stop=["<|im_end|>", "\n"]
    )

    # Run evaluation
    # Modified evaluation loop
    results = []
    for item in test_data:
        # Generate response
        generated = llm.generate([prompt], sampling_params)[0]
        full_response = generated.outputs[0].text
        
        # Compare results
        comparison = compare_sql_results(
            full_response,
            item["sql_result"],
            str(Path(args.model_dir)/args.db_path)
        )
        
        results.append({
            "question": item["user"],
            "full_response": full_response,
            **comparison
        })
    
    # Calculate metrics
    successful_executions = [r for r in results if r["execution_success"]]
    accuracy = sum(r["results_match"] for r in successful_executions) / len(successful_executions) if successful_executions else 0
    
    print(f"Valid Execution Rate: {len(successful_executions)/len(results):.2%}")
    print(f"Accuracy (Valid Executions): {accuracy:.2%}")
    
    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_metrics({
            "execution_success_rate": len(successful_executions)/len(results),
            "accuracy": accuracy
        })
        mlflow.log_artifact(args.results_path)

if __name__ == "__main__":

    main()