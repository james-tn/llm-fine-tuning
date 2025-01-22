# Distributed Fine-Tuning Techniques  
  
This repository provides an overview of distributed fine-tuning techniques for scaling large language models (LLMs) on Azure Machine Learning (Azure ML). It includes configurations, tools, and best practices for leveraging multi-GPU and multi-node setups.  
  
## Key Techniques  
### 1. Distributed Data Parallel (DDP)  
- **Purpose**: Replicates models across GPUs, synchronizes gradients.  
- **Azure ML Example**:  
  ```yaml  
  $schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json  
  type: command  
  compute: azureml:nc12s-cluster  
  distribution:  
    type: pytorch  
    process_count_per_instance: 1  
  command: >  
    accelerate launch --num_processes 4 --num_machines 2  
### 2. Model Parallelism & Memory Optimization  
  
Training large language models (LLMs) often exceeds the memory capacity of a single GPU. To address this, advanced frameworks like Fully Sharded Data Parallel (FSDP) and DeepSpeed are used to distribute computation and memory efficiently.  
  
#### Key Techniques  
- **FSDP** (Fully Sharded Data Parallel):  
  - Shards model parameters, gradients, and optimizer states across GPUs.  
  - Dynamically loads/unloads parameters to handle large models.  
  - Seamless integration with PyTorch.  
- **DeepSpeed**:  
  - **ZeRO Optimization**: Shards optimizer states, gradients, and model parameters in stages (ZeRO-1, ZeRO-2, ZeRO-3).  
  - Supports hybrid parallelism (data + tensor parallelism).  
  - Includes memory-saving features like INT8 quantization and offloading to CPU/NVMe.  
  
#### Memory Optimization Techniques  
- **Gradient Checkpointing**: Reduces memory by recomputing activations during backpropagation.  
- **Mixed Precision Training**: Uses FP16/BF16 instead of FP32 to save memory and accelerate training.  
- **Quantization (DeepSpeed Exclusive)**: Reduces memory/compute needs with INT8 precision.  
- **Offloading (DeepSpeed Exclusive)**: Moves model states and optimizer states to CPU or NVMe storage.  
  
#### Example Configurations  
- **DeepSpeed Configuration**:  
  ```yaml  
  $schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json  
  type: command  
  command: >  
    accelerate launch --config_file "configs/deepspeed_config_zero3.yaml

- **FSDP Configuration:**: 
  ```yaml  
  $schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json  
  type: command  
  command: >  
    accelerate launch --config_file "configs/fsdp_config.yaml

  

### Experiments: Scaling LLM Training with FSDP and DeepSpeed  
  
We conducted experiments on Azure ML using various distributed training configurations to fine-tune large language models. The goal was to evaluate the impact of different frameworks, hardware setups, and optimization techniques.  
  
#### Experiment Configurations  
| Model                  | Hardware Configuration                        | Framework | Fine-Tuning Method | Batch Size | Optimization Techniques                  |  
|------------------------|-----------------------------------------------|-----------|--------------------|------------|------------------------------------------|  
| Mistral-7B             | 2 nodes of NC12_v2 (4 GPUs, NVIDIA V100, 16 GB) | DDP       | LoRA               | 5          | Gradient checkpointing, mixed precision  |  
| LLAMA3_8B_Instruct     | 1 node of NC48_A100 (2 GPUs, NVIDIA A100, 40 GB) | FSDP      | Full-weight tuning | 5          | FSDP, mixed precision, gradient checkpointing |  
| Phi4 (14B)             | 1 node of NC96_A100 (4 GPUs, NVIDIA A100, 80 GB) | FSDP      | Full-weight tuning | 2          | DeepSpeed, mixed precision, gradient checkpointing |  
| LLAMA3.3-70B-Instruct  | 4 nodes of ND40_v2 (8 GPUs, NVIDIA V100, 32 GB)  | FSDP      | LoRA               | 1          | FSDP, mixed precision, gradient checkpointing |  
| LLAMA3.3-70B-Instruct  | 1 node of NC80_H100 (2 GPUs, NVIDIA H100, 80 GB) | FSDP      | LoRA               | 1          | FSDP, mixed precision, gradient checkpointing |  
  
#### Key Observations  
1. **Memory Utilization**:  
   - **FSDP**: Sharded parameters and gradients reduced peak memory usage, enabling training of large models (e.g., 70B parameters).  
   - **DeepSpeed ZeRO-3**: Further reduced memory requirements by offloading optimizer states to CPU memory.  
2. **Optimization Techniques**:  
   - **Mixed Precision**: Accelerated training and reduced memory consumption across all models.  
   - **Gradient Checkpointing**: Essential for larger models to fit within GPU memory.  
   - **LoRA**: Efficient fine-tuning for models like Mistral-7B and LLAMA3.3-70B.  
3. **Scaling Efficiency**:  
   - Distributed setups achieved near-linear scaling in throughput as nodes increased.  
  
#### Highlights  
- Fine-tuning LLAMA3.3-70B-Instruct using **LoRA** is possible on 2 nodes of ND40_v2 (32 V100 GPUs) with **FSDP**.  
- Full-weight fine-tuning of a 14B parameter model (Phi4) is feasible on a single NC96_A100 machine with **DeepSpeed ZeRO-3**. 
  
### Special Note for Using `accelerate launch` in Azure ML  
  
When using the `accelerate launch` command for distributed training in Azure ML, the configuration must account for the specific way `accelerate` handles distributed processes across nodes and GPUs.  
  
- **`process_count_per_instance` Must Be Set to 1**:    
  The `accelerate launch` command internally manages the distribution of processes across nodes and GPUs. Therefore, `process_count_per_instance` must remain set to `1` in the Azure ML YAML configuration.  
  
- **Parameters for `accelerate launch`**:  
  - **`--num_processes`**: Specifies the total number of GPUs across all nodes.    
    Example:    
    - If each node has 4 GPUs and there are 2 nodes, set `--num_processes 8`.    
    - In the example, `--num_processes 4` assumes 2 nodes with 2 GPUs each.  
  - **`--num_machines`**: Specifies the total number of nodes (e.g., 2 in this case).  
  - **`--machine_rank`**: Indicates the rank (index) of the current node in the cluster. Azure ML automatically sets this value via the `$NODE_RANK` environment variable.  
  - **`--main_process_ip`**: The IP address of the master node. Azure ML automatically sets this value via the `$MASTER_ADDR` environment variable.  
  - **`--main_process_port`**: The communication port for the master process. Azure ML automatically sets this via the `$MASTER_PORT` environment variable.  
  
## References  
- This code is adapted for Azure ML from [Hugging Face's repo](https://github.com/huggingface/peft/tree/main/examples/sft)
