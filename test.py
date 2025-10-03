#!/usr/bin/env python3
"""
Minimal NCCL test: runs an all_reduce across all ranks.

Usage (single node with N GPUs):
    torchrun --nproc_per_node=<NUM_GPUS> nccl_test.py

Usage (multi-node, 2 nodes × 4 GPUs each):
    # On node 0 (head):
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
        --master_addr=<HEAD_NODE_IP> --master_port=29500 nccl_test.py

    # On node 1:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
        --master_addr=<HEAD_NODE_IP> --master_port=29500 nccl_test.py
"""

import os
import torch
import torch.distributed as dist

def main():
    # torchrun automatically sets RANK, LOCAL_RANK, WORLD_SIZE
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Set device for this process
    torch.cuda.set_device(local_rank)

    # Init NCCL process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Create a tensor = rank + 1
    x = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
    dist.all_reduce(x)

    print(f"Rank {rank}/{world_size} on cuda:{local_rank}: tensor after all_reduce = {x.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

