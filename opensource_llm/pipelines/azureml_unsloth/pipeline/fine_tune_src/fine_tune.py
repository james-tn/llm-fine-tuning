import os
import mlflow
from transformers import TrainingArguments
import torch
from transformers.integrations import MLflowCallback
from transformers import TrainingArguments
from datasets import load_dataset
import os
import argparse
import math
import vllm
from vllm import LLM, SamplingParams


from transformers import TrainerCallback  
import mlflow  
def get_num_devices():
    return torch.cuda.device_count()
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
    #loading the model
    print("num GPU devices ", get_num_devices())

    llm = vllm.LLM(model = args.input_model+"/model", dtype = "bfloat16",tokenizer = args.input_model+"/tokenizer", max_model_len = 3000, 
                        tensor_parallel_size = get_num_devices(), 
                        )

    # os.environ["AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"] = "1800" #give time for model to be registered
    # mlflow.pyfunc.log_model(artifacts={"data":"data"}, artifact_path=model_artifact_path, python_model=LAMA2Predict())
    # model_uri = f"runs:/{run.info.run_id}/{model_artifact_path}"
    # mlflow.register_model(model_uri, name = model_name,await_registration_for=1800)

    # Sample prompts.
    prompts = [  
        "Hello, my name is",  
        "The president of the United States is",  
        "The capital of France is",  
        "The future of AI is",  
        "In the year 2050, technological advancements will have transformed the way we",  
        "The most significant challenge facing global leaders today is addressing",  
        "The historical significance of the Great Wall of China lies in its ability to",  
        "One of the most profound impacts of climate change is the alteration of",  
        "The development of renewable energy sources is crucial for",  
        "Throughout history, the role of women in shaping societies has been",  
        "The intricate process of photosynthesis in plants allows them to convert",  
        "In the realm of space exploration, the discovery of water on Mars suggests",  
        "The primary function of the human immune system is to protect the body from",  
        "The cultural and economic significance of the Silk Road in ancient times was",  
        "Artificial intelligence has the potential to revolutionize industries by",  
        "The underlying principles of quantum mechanics challenge our understanding of",  
        "The biodiversity of the Amazon Rainforest is critical for maintaining",  
        "The importance of mental health awareness in modern society cannot be",  
        "The evolution of language over centuries reflects the dynamic nature of",  
        "One of the greatest mysteries of the universe is the existence of",  
        "The architectural marvels of ancient Egypt, including the pyramids, were constructed to",  
        "The digital age has dramatically altered the way we communicate by introducing",  
        "In the field of medicine, the discovery of antibiotics has been pivotal in",  
        "The philosophical debates about the nature of consciousness often revolve around",  
        "The economic implications of globalization have led to increased",  
        "The role of education in fostering critical thinking and innovation is",  
        "The impact of social media on interpersonal relationships has been",  
        "The principles of democracy are grounded in the belief that citizens should",  
        "The exploration of the deep ocean has revealed a diverse array of",  
        "The significance of the moon landing in 1969 was a testament to human",  
        "The intricate relationship between genetics and environment in shaping behavior is",  
        "The conservation of endangered species is vital for preserving",  
        "The philosophical concept of time has been debated by scholars for",  
        "The development of autonomous vehicles presents both opportunities and challenges in"  
    ]      # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create an LLM.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")




if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)