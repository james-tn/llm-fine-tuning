import mlflow
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
import torch
from transformers.integrations import MLflowCallback
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset
import os
import argparse
import math



from transformers import TrainerCallback  
import mlflow  

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()


    # add arguments
    parser.add_argument("--input_model", type=str)
    # parser.add_argument("--learning_rate", type=float, default=5e-5)
    # parser.add_argument("--model_name", type=str)
    parser.add_argument("--trained_model", type=str, default="trained_model")

    # parse args
    args = parser.parse_args()

    # return args
    return args


class LAMA2Predict(mlflow.pyfunc.PythonModel):
    #place holder. Need to implement later
    pass


class MlflowLoggingCallback(TrainerCallback):  
    def on_log(self, args, state, control, logs=None, **kwargs):  
        # Log metrics to MLflow  
        if logs is not None:  
            mlflow.log_metrics(logs, step=state.global_step)  
            mlflow.log_metric('epoch', state.epoch)  



def main(args):
    # model_name = args.model_name
    # learning_rate = args.learning_rate
    print("content of the folder ", os.listdir(args.input_model))
    #code to perform fining....
    # os.environ["AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"] = "1800" #give time for model to be registered
    # mlflow.pyfunc.log_model(artifacts={"data":"data"}, artifact_path=model_artifact_path, python_model=LAMA2Predict())
    # model_uri = f"runs:/{run.info.run_id}/{model_artifact_path}"
    # mlflow.register_model(model_uri, name = model_name,await_registration_for=1800)




if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)