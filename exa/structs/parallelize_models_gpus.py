import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger


def setup_distributed_environment(
    default_port: int = 12355, default_master_addr: str = "localhost"
):
    """Set up the distributed environment variables with fallbacks."""
    # Default to localhost and a common port if not specified
    master_addr = os.getenv("MASTER_ADDR", default_master_addr)
    master_port = os.getenv("MASTER_PORT", default_port)

    # Ensure the MASTER_ADDR and MASTER_PORT are set for the current process
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    print(
        f"Using MASTER_ADDR={master_addr} and"
        f" MASTER_PORT={master_port}"
    )


def initialize_process_group(backend="nccl"):
    """Initialize the distributed environment."""
    # Here, no need to manually set rank and world_size as they should be handled by the launch utility
    try:
        logger.info("Initializing process group with NCCL backend")
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend)
        else:
            raise RuntimeError(
                "RANK and WORLD_SIZE environment variables need to be"
                " set."
            )
    except Exception as e:
        logger.error(f"Error initializing process group: {e}")
        raise


def prepare_model_for_ddp_inference(model):
    """Prepare and wrap the model for DDP execution in inference mode, considering bitsandbytes models."""
    try:
        model.eval()  # Ensure the model is in eval mode

        logger.info("Preparing model for DDP inference")
        if (
            torch.cuda.is_available()
            and torch.cuda.device_count() > 1
        ):
            logger.info("Multiple GPUs detected. Setting up DDP.")
            setup_distributed_environment()

            # Initialize the process group
            logger.info(
                "Initializing process group with NCCL backend"
            )
            initialize_process_group()

            rank = dist.get_rank()
            world_size = dist.get_world_size()

            logger.info(
                f"Rank {rank}/{world_size} - Preparing model for DDP"
                " inference"
            )

            model = DDP(
                model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False,
            )
        else:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model.to(device)
            print("Single GPU/CPU detected. Proceeding without DDP.")

        return model
    except Exception as e:
        logger.error(f"Error preparing model for DDP inference: {e}")
        raise
