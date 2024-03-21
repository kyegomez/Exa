import torch
from exa.utils import gpu_ops
from loguru import logger


def cluster_init():
    world_size, rank = gpu_ops.get_world_size_rank()
    logger.info(f"World size: {world_size}, Rank: {rank}")

    num_gpus = gpu_ops.get_num_gpus_available()
    logger.info(f"Number of GPUs available: {num_gpus}")

    # Calculate available memory on every gpu
    total_memory = gpu_ops.calculate_total_memory_across_gpus()
    logger.info(f"Total memory across all GPUs: {total_memory} GB")

    # Calculate available memory on every gpu
    total_available_memory = (
        gpu_ops.calculate_total_available_memory_across_gpus()
    )
    logger.info(
        "Total available memory across all GPUs:"
        f" {total_available_memory} GB"
    )

    if torch.cuda.is_available():
        available_memory = gpu_ops.calculate_available_memory()
        logger.info(f"Available memory: {available_memory}")

        model = torch.nn.Linear(10, 10)  # A simple model for testing
        model_memory_consumption = (
            gpu_ops.calculate_model_memory_consumption(model)
        )
        logger.info(
            f"Model memory consumption: {model_memory_consumption}"
        )

        available_memory_after_load = (
            gpu_ops.available_memory_after_model_load(model)
        )
        logger.info(
            "Available memory after model load:"
            f" {available_memory_after_load}"
        )
    else:
        logger.info("CUDA is not available")
