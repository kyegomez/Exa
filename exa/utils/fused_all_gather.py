import torch
import torch.distributed as dist
from typing import List
from torch import Tensor


# Fused all_gather operations
def fused_all_gather_v1(tensor_list: List[Tensor], tensor: Tensor):
    """
    Fused all_gather operation optimized for speed. Version 1 focuses on minimizing communication overhead.

    Args:
    - tensor_list (List[torch.Tensor]): List to store the gathered tensors from all processes.
    - tensor (torch.Tensor): Tensor to be gathered across all processes.
    """
    dist.get_rank()
    world_size = dist.get_world_size()

    # Ensuring tensor_list can hold tensors from all processes
    if len(tensor_list) != world_size:
        raise ValueError(
            "tensor_list must have length equal to the world size."
        )

    tensor_shape = tensor.size()
    flat_tensor = tensor.flatten()

    # Allocate a flat buffer for the gathered tensors
    buffer_size = flat_tensor.numel() * world_size
    buffer = torch.empty(
        buffer_size, dtype=tensor.dtype, device=tensor.device
    )

    # Allgather operation
    dist.all_gather(tensor_list, flat_tensor)

    # Concatenating into a single buffer then splitting ensures minimal communication overhead
    for i, gathered_tensor in enumerate(tensor_list):
        buffer[
            i * flat_tensor.numel() : (i + 1) * flat_tensor.numel()
        ] = gathered_tensor.flatten()

    # Reshape the flattened tensors back to their original shape
    for i in range(world_size):
        start_index = i * flat_tensor.numel()
        end_index = (i + 1) * flat_tensor.numel()
        tensor_list[i] = buffer[start_index:end_index].view(
            tensor_shape
        )


# Fused all_gather operations
def fused_all_gather_v2(tensor_list: List[Tensor], tensor: Tensor):
    """
    Fused all_gather operation optimized for speed. Version 2 focuses on reducing memory footprint.

    Args:
    - tensor_list (List[torch.Tensor]): List to store the gathered tensors from all processes.
    - tensor (torch.Tensor): Tensor to be gathered across all processes.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Ensuring tensor_list can hold tensors from all processes
    if len(tensor_list) != world_size:
        raise ValueError(
            "tensor_list must have length equal to the world size."
        )

    tensor_shape = tensor.size()
    num_elements = tensor.numel()

    # Use torch.cuda.comm.gather if available for lower memory footprint
    if tensor.is_cuda:
        gathered = torch.cuda.comm.gather(tensor, destination=0)
        if rank == 0:
            for i, t in enumerate(gathered):
                tensor_list[i] = t.view(tensor_shape)
    else:
        flat_tensor = tensor.flatten()
        torch.empty(
            num_elements * world_size,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dist.all_gather(tensor_list, flat_tensor)

        # Directly split the buffer to reduce memory footprint
        for i, gathered_tensor in enumerate(tensor_list):
            tensor_list[i] = gathered_tensor.view(tensor_shape)

        # For non-zero ranks or when not using CUDA, fall back to CPU-based gathering and reshaping
        if not tensor.is_cuda or rank != 0:
            for i in range(world_size):
                i * num_elements
                (i + 1) * num_elements
                tensor_list
