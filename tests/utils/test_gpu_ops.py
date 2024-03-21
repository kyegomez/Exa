import pytest
import torch
from exa.utils import gpu_ops

def test_get_world_size_rank():
    world_size, rank = gpu_ops.get_world_size_rank()
    assert world_size == 1
    assert rank == 0

def test_get_num_gpus_available():
    num_gpus = gpu_ops.get_num_gpus_available()
    assert isinstance(num_gpus, int)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_calculate_available_memory():
    available_memory = gpu_ops.calculate_available_memory()
    assert isinstance(available_memory, int)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_calculate_model_memory_consumption():
    model = torch.nn.Linear(10, 10)  # A simple model for testing
    model_memory_consumption = gpu_ops.calculate_model_memory_consumption(model)
    assert isinstance(model_memory_consumption, int)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_available_memory_after_model_load():
    model = torch.nn.Linear(10, 10)  # A simple model for testing
    available_memory_after_load = gpu_ops.available_memory_after_model_load(model)
    assert isinstance(available_memory_after_load, int)