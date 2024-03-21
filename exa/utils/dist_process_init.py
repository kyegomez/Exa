from dotenv import load_dotenv
import os
import torch
from torch import distributed as dist
from exa.utils.gpu_ops import get_world_size_rank
from loguru import logger


# Load environment variables
load_dotenv()


def initialize_distributed():
    """
    Dynamically initializes the PyTorch distributed process group,
    allowing for communication between the processes initiated.
    """
    if not dist.is_initialized():
        # Automatically selects a backend based on the environment and cuda availability
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        logger.info(
            f"Initializing process group with backend: {backend}"
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        world_size, rank = get_world_size_rank()

        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)

        dist.init_process_group(backend)

    logger.info(
        f"Initialized process group with {world_size} processes. Rank"
        f" {rank} ready to go!"
    )

    # Add checks for initialization errors
    if not dist.is_initialized():
        logger.error(
            "Failed to initialize the distributed process group."
        )
        raise RuntimeError(
            "Failed to initialize the distributed process group."
        )

    # Add logging for environment variables
    logger.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
    logger.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
    logger.debug(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}")
    logger.debug(f"RANK: {os.environ['RANK']}")
