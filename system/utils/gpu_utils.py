import torch
import subprocess
import re
import os
from typing import List, Dict, Optional


def get_gpu_info() -> List[Dict]:
    """
    Get detailed information about available GPUs.
    
    Returns:
        List of dictionaries with GPU information
    """
    if not torch.cuda.is_available():
        return []
    
    gpu_info = []
    
    try:
        # Get GPU information using nvidia-smi
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 6:
                    gpu_info.append({
                        'id': int(parts[0]),
                        'name': parts[1],
                        'memory_total': int(parts[2]),
                        'memory_used': int(parts[3]),
                        'memory_free': int(parts[4]),
                        'utilization': int(parts[5])
                    })
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        # Fallback to basic PyTorch info
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                'id': i,
                'name': props.name,
                'memory_total': props.total_memory // 1024**2,  # MB
                'memory_used': 0,
                'memory_free': props.total_memory // 1024**2,
                'utilization': 0
            })
    
    return gpu_info


def select_optimal_gpus(num_processes: int, min_memory_mb: int = 4000, max_utilization: int = 50) -> List[int]:
    """
    Select optimal GPUs based on memory and utilization criteria.
    
    Args:
        num_processes: Number of processes that need GPUs
        min_memory_mb: Minimum free memory required (MB)
        max_utilization: Maximum utilization threshold (%)
        
    Returns:
        List of optimal GPU IDs
    """
    gpu_info = get_gpu_info()
    
    if not gpu_info:
        return []
    
    # Filter GPUs based on criteria
    suitable_gpus = []
    for gpu in gpu_info:
        if gpu['memory_free'] >= min_memory_mb and gpu['utilization'] <= max_utilization:
            suitable_gpus.append(gpu)
    
    # Sort by utilization (ascending) then by free memory (descending)
    suitable_gpus.sort(key=lambda x: (x['utilization'], -x['memory_free']))
    
    # Select top GPUs
    selected_gpu_ids = [gpu['id'] for gpu in suitable_gpus[:num_processes]]
    
    return selected_gpu_ids


def print_gpu_status():
    """Print current GPU status in a formatted table."""
    gpu_info = get_gpu_info()
    
    if not gpu_info:
        print("No CUDA-capable GPUs found")
        return
    
    print("\n=== GPU Status ===")
    print(f"{'ID':>3} {'Name':<20} {'Memory (MB)':<15} {'Utilization':<12} {'Status'}")
    print("-" * 65)
    
    for gpu in gpu_info:
        memory_str = f"{gpu['memory_used']}/{gpu['memory_total']}"
        util_str = f"{gpu['utilization']}%"
        
        # Determine status
        if gpu['utilization'] > 80:
            status = "BUSY"
        elif gpu['memory_free'] < 2000:
            status = "LOW MEM"
        else:
            status = "AVAILABLE"
        
        print(f"{gpu['id']:>3} {gpu['name']:<20} {memory_str:<15} {util_str:<12} {status}")
    
    print()


def validate_gpu_selection(gpu_ids: List[int]) -> bool:
    """
    Validate that selected GPU IDs are available and accessible.
    
    Args:
        gpu_ids: List of GPU IDs to validate
        
    Returns:
        bool: True if all GPUs are valid and accessible
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False
    
    available_gpus = torch.cuda.device_count()
    
    for gpu_id in gpu_ids:
        if gpu_id < 0 or gpu_id >= available_gpus:
            print(f"Invalid GPU ID {gpu_id}. Available GPUs: 0-{available_gpus-1}")
            return False
        
        try:
            # Test if we can access the GPU
            torch.cuda.set_device(gpu_id)
            _ = torch.cuda.get_device_properties(gpu_id)
        except RuntimeError as e:
            print(f"Cannot access GPU {gpu_id}: {e}")
            return False
    
    return True


def recommend_gpu_configuration(num_processes: int) -> Dict:
    """
    Recommend optimal GPU configuration for given number of processes.
    
    Args:
        num_processes: Number of DDP processes needed
        
    Returns:
        Dictionary with recommended configuration
    """
    gpu_info = get_gpu_info()
    
    if not gpu_info:
        return {
            'recommended_gpus': [],
            'processes_per_gpu': 1,
            'total_processes': 0,
            'warning': 'No CUDA GPUs available'
        }
    
    # Select optimal GPUs
    optimal_gpus = select_optimal_gpus(num_processes)
    
    if len(optimal_gpus) >= num_processes:
        # Enough GPUs for 1 process per GPU
        return {
            'recommended_gpus': optimal_gpus[:num_processes],
            'processes_per_gpu': 1,
            'total_processes': num_processes,
            'warning': None
        }
    elif len(optimal_gpus) > 0:
        # Need to share GPUs
        processes_per_gpu = (num_processes + len(optimal_gpus) - 1) // len(optimal_gpus)
        return {
            'recommended_gpus': optimal_gpus,
            'processes_per_gpu': processes_per_gpu,
            'total_processes': len(optimal_gpus) * processes_per_gpu,
            'warning': f'Sharing {processes_per_gpu} processes per GPU'
        }
    else:
        # No suitable GPUs found
        return {
            'recommended_gpus': [],
            'processes_per_gpu': 1,
            'total_processes': 0,
            'warning': 'No suitable GPUs found (check memory/utilization)'
        }


def set_cuda_visible_devices(gpu_ids: List[int]):
    """
    Set CUDA_VISIBLE_DEVICES environment variable.
    
    Args:
        gpu_ids: List of GPU IDs to make visible
    """
    if gpu_ids:
        gpu_str = ','.join(map(str, gpu_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
        print(f"CUDA_VISIBLE_DEVICES set to: {gpu_str}")
    else:
        print("No GPUs specified, using default CUDA_VISIBLE_DEVICES")