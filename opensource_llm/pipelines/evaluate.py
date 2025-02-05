import json
import sqlite3
from pathlib import Path
import mlflow
from vllm import LLM, SamplingParams

def execute_sql(query, db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return sorted(map(str, [item for sublist in result for item in sublist]))
    except Exception as e:
        print(f"Error executing query: {e}")
        return None

def compare_queries(predicted, ground_truth, db_path):
    pred_result = execute_sql(predicted, db_path)
    truth_result = execute_sql(ground_truth, db_path)
    
    if pred_result and truth_result:
        return pred_result == truth_result
    return False

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
    results = []
    for item in test_data:
        prompt = f"<|im_start|>system
Generate SQL for: {item['user']}<|im_end|>\n<|im_start|>assistant
"
        generated = llm.generate([prompt], sampling_params)[0]
        predicted_sql = generated.outputs[0].text.split("<|im_end|>")[0].strip()
        
        is_correct = compare_queries(
            predicted_sql,
            item["sql_result"],
            Path(args.model_dir)/args.db_path
        )
        
        results.append({
            "question": item["user"],
            "predicted": predicted_sql,
            "ground_truth": item["sql_result"],
            "correct": is_correct
        })

    # Calculate metrics
    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"Execution Accuracy: {accuracy:.2%}")

    # Log results
    with open(args.results_path, "w") as f:
        json.dump({"accuracy": accuracy, "details": results}, f)

    # MLflow logging
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_artifact(args.results_path)
        mlflow.log_params({
            "model": args.model_dir,
            "test_set_size": len(results)
        })

if __name__ == "__main__":

    main()