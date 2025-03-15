FROM vllm/vllm-openai:latest  
  
# Set environment variables  
ENV MODEL_NAME Qwen/Qwen2.5-VL-7B-Instruct  
  
# Install required Python libraries  
RUN pip install git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775 \  
    && pip install accelerate \  
    && pip install qwen-vl-utils \  
    && pip install 'vllm>0.7.2'  
  
# Execute vLLM server entrypoint with the specified model and arguments  
ENTRYPOINT python3 -m vllm.entrypoints.openai.api_server --model $MODEL_NAME $VLLM_ARGS  