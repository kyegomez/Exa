import torch
from torch import distributed as dist
from torch.cuda import (
    memory_allocated,
    get_device_properties,
    memory_reserved,
)


def get_world_size_rank():
    """
    Calculate the world size and rank in a distributed environment.
    """
    if not dist.is_initialized():
        # Assuming a default setup if not running in a distributed context
        world_size = (
            torch.cuda.device_count()
        )  # Adjusted to consider the number of GPUs as world size
        rank = 0
    else:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    return world_size, rank


def get_num_gpus_available():
    """
    Return the number of GPUs available in the system.
    """
    return torch.cuda.device_count()


def calculate_available_memory(gpu_id=0):
    """
    Calculates the available memory on a specified GPU in GB.
    """
    if (
        not torch.cuda.is_available()
        or gpu_id >= get_num_gpus_available()
    ):
        raise RuntimeError(
            "CUDA is not available or the specified GPU ID is out of"
            " bounds."
        )

    torch.cuda.set_device(gpu_id)
    total_memory = get_device_properties(gpu_id).total_memory
    allocated_memory = memory_allocated(gpu_id)
    cached_memory = memory_reserved(gpu_id)
    available_memory_bytes = total_memory - (
        allocated_memory + cached_memory
    )
    return available_memory_bytes / (1024**3)  # Convert to GB


def calculate_total_memory_across_gpus():
    """
    Calculates the total memory across all GPUs in GB.
    Returns:
        total_memory_gb (float): Total memory across all GPUs in gigabytes.
    """
    total_memory_gb = 0.0
    for gpu_id in range(get_num_gpus_available()):
        total_memory = get_device_properties(gpu_id).total_memory
        total_memory_gb += total_memory / (
            1024**3
        )  # Convert bytes to GB
    return total_memory_gb


def calculate_total_available_memory_across_gpus():
    """
    Calculates the total available memory across all GPUs in GB.
    Returns:
        total_available_memory_gb (float): Total available memory across all GPUs in gigabytes.
    """
    total_available_memory_gb = 0.0
    for gpu_id in range(get_num_gpus_available()):
        available_memory = calculate_available_memory(gpu_id)
        total_available_memory_gb += available_memory  # Assuming available_memory is already in GB
    return total_available_memory_gb


def calculate_model_memory_consumption(model, gpu_id=0):
    """
    Calculates the memory consumption of a model on a specific GPU by temporarily moving it to the GPU.
    Args:
        model (torch.nn.Module): The model to calculate memory consumption for.
        gpu_id (int): GPU ID to use for the calculation.
    Returns:
        model_memory_consumption (int): Memory consumption in bytes.
    """
    next(model.parameters()).device
    model.to(f"cuda:{gpu_id}")
    torch.cuda.synchronize(gpu_id)

    allocated_memory_before = memory_allocated(gpu_id)
    torch.cuda.synchronize(gpu_id)
    allocated_memory_after = memory_allocated(gpu_id)
    model_memory_consumption = (
        allocated_memory_after - allocated_memory_before
    )

    return model_memory_consumption


def available_memory_after_model_load(model, gpu_id=0):
    """
    Calculates available GPU memory after loading a model onto it.
    Args:
        model (torch.nn.Module): The model to be loaded onto the GPU.
        gpu_id (int): GPU ID where the model will be loaded.
    Returns:
        available_memory_after_load (int): Available memory in bytes on the GPU after loading the model.
    """
    model_memory_consumption = calculate_model_memory_consumption(
        model, gpu_id
    )
    available_memory_before_load = calculate_available_memory(gpu_id)
    available_memory_after_load = (
        available_memory_before_load - model_memory_consumption
    )
    return available_memory_after_load
