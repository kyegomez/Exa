import torch
from exa.utils import gpu_ops

def run_gpu_ops():
    world_size, rank = gpu_ops.get_world_size_rank()
    print(f"World size: {world_size}, Rank: {rank}")

    num_gpus = gpu_ops.get_num_gpus_available()
    print(f"Number of GPUs available: {num_gpus}")
    
    
    # Calculate available memory on every gpu
    total_memory = gpu_ops.calculate_total_memory_across_gpus()
    print(f"Total memory across all GPUs: {total_memory} GB")
    
    
    # Calculate available memory on every gpu
    total_available_memory = gpu_ops.calculate_total_available_memory_across_gpus()
    print(f"Total available memory across all GPUs: {total_available_memory} GB")

    if torch.cuda.is_available():
        available_memory = gpu_ops.calculate_available_memory()
        print(f"Available memory: {available_memory}")

        model = torch.nn.Linear(10, 10)  # A simple model for testing
        model_memory_consumption = gpu_ops.calculate_model_memory_consumption(model)
        print(f"Model memory consumption: {model_memory_consumption}")

        available_memory_after_load = gpu_ops.available_memory_after_model_load(model)
        print(f"Available memory after model load: {available_memory_after_load}")
    else:
        print("CUDA is not available")

if __name__ == "__main__":
    run_gpu_ops()