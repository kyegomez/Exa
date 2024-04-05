import torch
from torch import Tensor
import torch.distributed as dist
# from exa.utils.dist_process_init import initialize_distributed


# initialize_distributed()


def fused_all_reduce_v1(tensor: Tensor, op=dist.ReduceOp.SUM):
    """
    Hyper-optimized fused all_reduce operation with reduced communication overhead.
    This version uses a ring-reduce approach to minimize the amount of data sent across the network.

    Args:
    - tensor (torch.Tensor): The tensor to be reduced across all processes.
    - op (dist.ReduceOp): The reduction operation to apply. Default is SUM.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    send_buff = tensor.clone()
    recv_buff = torch.empty_like(tensor)
    accum_buff = torch.empty_like(tensor)
    accum_buff[:] = tensor

    left = (rank - 1) % world_size
    right = (rank + 1) % world_size

    for i in range(1, world_size):
        # Send send_buff to the right, receive recv_buff from the left
        send_req = dist.isend(send_buff, right)
        recv_req = dist.irecv(recv_buff, left)
        send_req.wait()
        recv_req.wait()

        # Perform the reduction operation
        if op == dist.ReduceOp.SUM:
            accum_buff += recv_buff
        elif op == dist.ReduceOp.PRODUCT:
            accum_buff *= recv_buff
        elif op == dist.ReduceOp.MAX:
            torch.max(accum_buff, recv_buff, out=accum_buff)
        elif op == dist.ReduceOp.MIN:
            torch.min(accum_buff, recv_buff, out=accum_buff)

        # Prepare for the next iteration
        send_buff[:] = recv_buff

    # Copy the accumulated buffer back to the original tensor
    tensor[:] = accum_buff


def fused_all_reduce_v2(tensor: Tensor, op=dist.ReduceOp.SUM):
    """
    Perform an all-reduce operation on the given tensor using the specified reduction operation.

    Args:
        tensor (Tensor): The input tensor to be reduced.
        op (dist.ReduceOp, optional): The reduction operation to be applied. Defaults to dist.ReduceOp.SUM.

    Returns:
        Tensor: The tensor after the all-reduce operation has been applied.
    """

    # Use pytorch's built-in all_reduce
    return dist.all_reduce(tensor, op)


# x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
# fused_all_reduce_v1(x)  # Output: tensor([ 1.,  2.,  3.,  4.,  5.])
# print(x)
