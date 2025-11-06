import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional, Callable


def setup_ddp_environment(rank: int, world_size: int, master_addr: str = "localhost", master_port: str = "12355", visible_gpus: Optional[list] = None):
    """
    Setup environment variables for DDP initialization.
    
    Args:
        rank: Global rank of the process
        world_size: Total number of processes
        master_addr: Address of the master node
        master_port: Port for communication
        visible_gpus: List of GPU IDs to use (e.g., [0, 2, 3])
    """
    os.environ['RANK'] = str(rank)
    
    # Calculate local_rank based on visible GPUs
    if visible_gpus:
        # Map rank to visible GPUs
        gpus_per_node = len(visible_gpus)
        local_rank = rank % gpus_per_node
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, visible_gpus))
        print(f"Rank {rank}: Using GPUs {visible_gpus}, local_rank={local_rank}")
    else:
        # Use default mapping
        local_rank = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        os.environ['LOCAL_RANK'] = str(local_rank)
    
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port


def is_ddp_available() -> bool:
    """
    Check if DDP environment is properly configured.
    
    Returns:
        bool: True if all required environment variables are set
    """
    required_vars = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    available = all(var in os.environ for var in required_vars)
    
    if not available:
        missing_vars = [var for var in required_vars if var not in os.environ]
        print(f"DDP not available - missing environment variables: {missing_vars}")
        print("For DDP multi-process training, set these variables or use torchrun")
    
    return available


def get_ddp_info() -> dict:
    """
    Get DDP configuration information from environment.
    
    Returns:
        dict: DDP configuration or None if not available
    """
    if not is_ddp_available():
        return None
    
    return {
        'rank': int(os.environ['RANK']),
        'local_rank': int(os.environ['LOCAL_RANK']),
        'world_size': int(os.environ['WORLD_SIZE']),
        'master_addr': os.environ['MASTER_ADDR'],
        'master_port': os.environ['MASTER_PORT']
    }


def cleanup_ddp_on_error(func: Callable):
    """
    Decorator to ensure DDP cleanup on errors.
    
    Args:
        func: Function to wrap with DDP error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if dist.is_initialized():
                print(f"Error in {func.__name__}: {e}")
                print("Cleaning up DDP...")
                dist.destroy_process_group()
            raise e
    return wrapper


class DDPErrorHandler:
    """
    Context manager for DDP error handling and cleanup.
    """
    
    def __init__(self, client_id: Optional[int] = None):
        self.client_id = client_id
        self.process_group_initialized = False
    
    def __enter__(self):
        self.process_group_initialized = dist.is_initialized()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            client_info = f" (Client {self.client_id})" if self.client_id is not None else ""
            print(f"DDP Error{client_info}: {exc_type.__name__}: {exc_val}")
            
            # Cleanup if we initialized the process group
            if self.process_group_initialized and dist.is_initialized():
                try:
                    dist.destroy_process_group()
                    print(f"DDP process group cleaned up{client_info}")
                except Exception as cleanup_error:
                    print(f"Error during DDP cleanup{client_info}: {cleanup_error}")
        
        # Return False to propagate the exception
        return False


def run_federated_ddp(
    world_size: int,
    federated_function: Callable,
    args,
    master_addr: str = "localhost",
    master_port: str = "12355",
    visible_gpus: Optional[list] = None
):
    """
    Launch federated learning with DDP support.
    
    Args:
        world_size: Number of processes to spawn
        federated_function: Function to run (should accept rank and args)
        args: Arguments to pass to federated function
        master_addr: Master node address
        master_port: Master node port
        visible_gpus: List of GPU IDs to use (e.g., [0, 2, 3])
    """
    def worker(rank):
        # Setup environment for this process
        setup_ddp_environment(rank, world_size, master_addr, master_port, visible_gpus)
        
        # Add DDP-specific arguments
        args.ddp_enabled = True
        args.ddp_world_size = world_size
        args.ddp_backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        args.ddp_visible_gpus = visible_gpus
        
        # Run the federated learning function
        with DDPErrorHandler(client_id=rank):
            federated_function(args)
    
    if world_size == 1:
        # Single process mode
        args.ddp_enabled = False
        if visible_gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, visible_gpus))
        federated_function(args)
    else:
        # Multi-process DDP mode
        mp.spawn(worker, args=(), nprocs=world_size, join=True)


def verify_ddp_setup():
    """
    Verify that DDP setup is working correctly.
    
    Returns:
        bool: True if DDP verification passes
    """
    try:
        if not torch.cuda.is_available():
            print("CUDA not available - DDP will use CPU backend")
            return True
        
        if not is_ddp_available():
            print("DDP environment variables not set")
            return False
        
        ddp_info = get_ddp_info()
        print(f"DDP Environment verified:")
        print(f"  - Rank: {ddp_info['rank']}")
        print(f"  - Local Rank: {ddp_info['local_rank']}")
        print(f"  - World Size: {ddp_info['world_size']}")
        print(f"  - Master: {ddp_info['master_addr']}:{ddp_info['master_port']}")
        
        return True
        
    except Exception as e:
        print(f"DDP verification failed: {e}")
        return False


def parse_gpu_selection(gpu_str: str) -> list:
    """
    Parse GPU selection string into list of GPU IDs.
    
    Args:
        gpu_str: String like "0,2,3" or "0-3" or "all"
        
    Returns:
        List of GPU IDs to use
    """
    if not gpu_str or gpu_str.lower() == 'all':
        return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    
    gpu_ids = []
    for part in gpu_str.split(','):
        part = part.strip()
        if '-' in part:
            # Range like "0-3"
            start, end = map(int, part.split('-'))
            gpu_ids.extend(range(start, end + 1))
        else:
            # Single GPU ID
            gpu_ids.append(int(part))
    
    # Validate GPU IDs
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        gpu_ids = [gpu for gpu in gpu_ids if 0 <= gpu < available_gpus]
    
    return gpu_ids


def get_optimal_gpu_distribution(num_processes: int, available_gpus: list) -> dict:
    """
    Distribute processes optimally across available GPUs.
    
    Args:
        num_processes: Number of DDP processes to create
        available_gpus: List of available GPU IDs
        
    Returns:
        dict: Mapping from rank to GPU ID
    """
    if not available_gpus:
        return {}
    
    distribution = {}
    for rank in range(num_processes):
        gpu_idx = rank % len(available_gpus)
        distribution[rank] = available_gpus[gpu_idx]
    
    return distribution


def initialize_ddp_args(args):
    """
    Add DDP-specific arguments to args if they don't exist.
    
    Args:
        args: Argument object to modify
    """
    # Set default DDP arguments if not present
    if not hasattr(args, 'ddp_enabled'):
        # Only enable DDP if environment is properly configured
        ddp_available = is_ddp_available()
        args.ddp_enabled = ddp_available and int(os.environ.get('WORLD_SIZE', 1)) > 1
    
    if not hasattr(args, 'ddp_backend'):
        args.ddp_backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    
    if not hasattr(args, 'ddp_world_size'):
        args.ddp_world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if not hasattr(args, 'ddp_visible_gpus'):
        args.ddp_visible_gpus = None
    
    # Parse GPU selection if provided
    if hasattr(args, 'gpu_ids') and args.gpu_ids:
        args.ddp_visible_gpus = parse_gpu_selection(args.gpu_ids)
    
    # Log DDP status
    if args.ddp_enabled:
        print(f"DDP enabled: world_size={args.ddp_world_size}, backend={args.ddp_backend}")
        if args.ddp_visible_gpus:
            print(f"DDP visible GPUs: {args.ddp_visible_gpus}")
    else:
        print("DDP disabled - running in single process mode")
    
    return args