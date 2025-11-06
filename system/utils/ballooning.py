"""
GPU Memory Ballooning System

This module provides functionality to allocate and deallocate GPU memory
to prevent other processes from using it and causing OOM (Out Of Memory) errors.

The ballooning system works by allocating dummy tensors on specified GPUs,
effectively "reserving" memory that won't be used by other processes.

Usage:
    # Single GPU
    balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=4000)
    balloon.inflate()  # Allocate memory
    # ... do your work ...
    balloon.deflate()  # Release memory

    # Multiple GPUs
    multi_balloon = MultiGPUMemoryBalloon(gpu_ids=[0, 1, 2], reserve_mb_per_gpu=2000)
    multi_balloon.inflate_all()
    # ... do your work ...
    multi_balloon.deflate_all()

    # Context manager (automatic cleanup)
    with GPUMemoryBalloon(gpu_id=0, reserve_mb=4000):
        # Memory is reserved here
        pass
    # Memory automatically released
"""

import torch
import logging
from typing import List, Dict, Optional, Union
from contextlib import contextmanager


# Setup logging
logger = logging.getLogger(__name__)


def parse_memory_spec(memory_spec: Union[int, str, float],
                      total_memory_mb: Optional[int] = None,
                      free_memory_mb: Optional[int] = None) -> int:
    """
    Parse flexible memory specification and convert to MB.

    Args:
        memory_spec: Memory specification in various formats:
            - int: Amount in MB (e.g., 2000 = 2000 MB)
            - str with 'GB': Amount in GB (e.g., "2GB", "2.5GB")
            - str with 'MB': Amount in MB (e.g., "2000MB")
            - str with '%': Percentage of free memory (e.g., "50%", "75%")
            - float 0.0-1.0: Fraction of free memory (e.g., 0.5 = 50%)
        total_memory_mb: Total GPU memory in MB (required for some conversions)
        free_memory_mb: Free GPU memory in MB (required for percentage/fraction)

    Returns:
        int: Amount in MB

    Examples:
        >>> parse_memory_spec(2000)  # 2000 MB
        2000
        >>> parse_memory_spec("2GB")  # 2 GB
        2048
        >>> parse_memory_spec("2.5GB")  # 2.5 GB
        2560
        >>> parse_memory_spec("50%", free_memory_mb=10000)  # 50% of 10GB
        5000
        >>> parse_memory_spec(0.7, free_memory_mb=10000)  # 70% of 10GB
        7000
    """
    # Integer: direct MB
    if isinstance(memory_spec, int):
        return memory_spec

    # Float: fraction of free memory
    if isinstance(memory_spec, float):
        if not (0.0 < memory_spec <= 1.0):
            raise ValueError("Float memory_spec must be between 0.0 and 1.0 (fraction)")
        if free_memory_mb is None:
            raise ValueError("free_memory_mb required for fractional memory_spec")
        return int(free_memory_mb * memory_spec)

    # String: parse various formats
    if isinstance(memory_spec, str):
        memory_spec = memory_spec.strip().upper()

        # GB format
        if memory_spec.endswith('GB'):
            try:
                gb_value = float(memory_spec[:-2].strip())
                return int(gb_value * 1024)  # Convert GB to MB
            except ValueError:
                raise ValueError(f"Invalid GB format: {memory_spec}")

        # MB format
        if memory_spec.endswith('MB'):
            try:
                mb_value = float(memory_spec[:-2].strip())
                return int(mb_value)
            except ValueError:
                raise ValueError(f"Invalid MB format: {memory_spec}")

        # Percentage format
        if memory_spec.endswith('%'):
            try:
                percentage = float(memory_spec[:-1].strip())
                if not (0.0 < percentage <= 100.0):
                    raise ValueError("Percentage must be between 0 and 100")
                if free_memory_mb is None:
                    raise ValueError("free_memory_mb required for percentage memory_spec")
                return int(free_memory_mb * (percentage / 100.0))
            except ValueError as e:
                raise ValueError(f"Invalid percentage format: {memory_spec}") from e

        # Try parsing as plain number (MB)
        try:
            return int(float(memory_spec))
        except ValueError:
            raise ValueError(f"Invalid memory specification: {memory_spec}")

    raise ValueError(f"Unsupported memory_spec type: {type(memory_spec)}")


class GPUMemoryBalloon:
    """
    Memory ballooning for a single GPU.

    Allocates dummy tensors to reserve GPU memory and prevent OOM from other processes.
    """

    def __init__(self, gpu_id: int, reserve_mb: Optional[int] = None,
                 reserve_fraction: Optional[float] = None, chunk_size_mb: int = 100):
        """
        Initialize GPU memory balloon.

        Args:
            gpu_id: GPU device ID
            reserve_mb: Amount of memory to reserve in MB (exclusive with reserve_fraction)
            reserve_fraction: Fraction of free memory to reserve (0.0-1.0, exclusive with reserve_mb)
            chunk_size_mb: Size of each allocation chunk in MB (for safer allocation)

        Raises:
            ValueError: If both or neither reserve_mb and reserve_fraction are specified
            RuntimeError: If CUDA is not available or GPU is invalid
        """
        if (reserve_mb is None) == (reserve_fraction is None):
            raise ValueError("Specify exactly one of reserve_mb or reserve_fraction")

        if reserve_fraction is not None and not (0.0 < reserve_fraction <= 1.0):
            raise ValueError("reserve_fraction must be between 0.0 and 1.0")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise RuntimeError(f"Invalid GPU ID {gpu_id}. Available: 0-{torch.cuda.device_count()-1}")

        self.gpu_id = gpu_id
        self.reserve_mb = reserve_mb
        self.reserve_fraction = reserve_fraction
        self.chunk_size_mb = chunk_size_mb
        self.balloons: List[torch.Tensor] = []
        self.total_allocated_mb = 0
        self.is_inflated = False

        logger.info(f"Initialized GPUMemoryBalloon for GPU {gpu_id}")

    def _get_free_memory_mb(self) -> int:
        """Get free memory on the GPU in MB."""
        torch.cuda.set_device(self.gpu_id)
        free_mb = torch.cuda.mem_get_info(self.gpu_id)[0] // (1024 ** 2)
        return free_mb

    def _get_total_memory_mb(self) -> int:
        """Get total memory on the GPU in MB."""
        torch.cuda.set_device(self.gpu_id)
        total_mb = torch.cuda.mem_get_info(self.gpu_id)[1] // (1024 ** 2)
        return total_mb

    def _calculate_target_mb(self) -> int:
        """Calculate target memory to reserve based on configuration."""
        if self.reserve_mb is not None:
            return self.reserve_mb
        else:
            free_mb = self._get_free_memory_mb()
            target_mb = int(free_mb * self.reserve_fraction)
            logger.info(f"GPU {self.gpu_id}: Free memory: {free_mb} MB, "
                       f"Reserving {self.reserve_fraction*100:.1f}% = {target_mb} MB")
            return target_mb

    def inflate(self, verbose: bool = True) -> int:
        """
        Allocate memory on the GPU (inflate the balloon).

        Args:
            verbose: If True, log allocation details

        Returns:
            int: Amount of memory allocated in MB

        Raises:
            RuntimeError: If already inflated or allocation fails
        """
        if self.is_inflated:
            raise RuntimeError(f"Balloon on GPU {self.gpu_id} is already inflated")

        torch.cuda.set_device(self.gpu_id)

        # Get initial state
        free_mb_before = self._get_free_memory_mb()
        total_mb = self._get_total_memory_mb()
        target_mb = self._calculate_target_mb()

        if verbose:
            logger.info(f"GPU {self.gpu_id}: Total memory: {total_mb} MB, "
                       f"Free before: {free_mb_before} MB, Target reserve: {target_mb} MB")

        if target_mb > free_mb_before:
            logger.warning(f"GPU {self.gpu_id}: Requested {target_mb} MB but only {free_mb_before} MB available. "
                          f"Allocating {free_mb_before} MB instead.")
            target_mb = max(0, free_mb_before - 100)  # Leave 100 MB buffer

        # Allocate memory in chunks
        allocated_mb = 0
        try:
            while allocated_mb < target_mb:
                remaining_mb = target_mb - allocated_mb
                chunk_mb = min(self.chunk_size_mb, remaining_mb)

                # Allocate chunk (float32 = 4 bytes per element)
                elements = (chunk_mb * 1024 * 1024) // 4
                chunk_tensor = torch.empty(elements, dtype=torch.float32, device=f'cuda:{self.gpu_id}')

                self.balloons.append(chunk_tensor)
                allocated_mb += chunk_mb

                if verbose and allocated_mb % 500 == 0:
                    logger.info(f"GPU {self.gpu_id}: Allocated {allocated_mb}/{target_mb} MB")

            self.total_allocated_mb = allocated_mb
            self.is_inflated = True

            free_mb_after = self._get_free_memory_mb()

            if verbose:
                logger.info(f"GPU {self.gpu_id}: Successfully allocated {allocated_mb} MB. "
                           f"Free memory: {free_mb_before} MB -> {free_mb_after} MB")

            return allocated_mb

        except RuntimeError as e:
            # Clean up on failure
            logger.error(f"GPU {self.gpu_id}: Failed to allocate memory: {e}")
            self.deflate()
            raise RuntimeError(f"Failed to allocate {target_mb} MB on GPU {self.gpu_id}: {e}")

    def deflate(self, verbose: bool = True) -> int:
        """
        Deallocate memory on the GPU (deflate the balloon).

        Args:
            verbose: If True, log deallocation details

        Returns:
            int: Amount of memory freed in MB
        """
        if not self.balloons:
            if verbose:
                logger.info(f"GPU {self.gpu_id}: No memory to release")
            return 0

        torch.cuda.set_device(self.gpu_id)
        free_mb_before = self._get_free_memory_mb()

        # Clear all balloon tensors
        freed_mb = self.total_allocated_mb
        self.balloons.clear()
        torch.cuda.empty_cache()

        self.total_allocated_mb = 0
        self.is_inflated = False

        free_mb_after = self._get_free_memory_mb()

        if verbose:
            logger.info(f"GPU {self.gpu_id}: Released {freed_mb} MB. "
                       f"Free memory: {free_mb_before} MB -> {free_mb_after} MB")

        return freed_mb

    def inflate_if_not_inflated(self, verbose: bool = True) -> int:
        """
        Inflate balloon only if not already inflated.

        Args:
            verbose: If True, log allocation details

        Returns:
            int: Amount of memory allocated in MB (0 if already inflated)
        """
        if self.is_inflated:
            if verbose:
                logger.info(f"GPU {self.gpu_id}: Balloon already inflated, skipping")
            return 0
        return self.inflate(verbose=verbose)

    def deflate_if_inflated(self, verbose: bool = True) -> int:
        """
        Deflate balloon only if currently inflated.

        Args:
            verbose: If True, log deallocation details

        Returns:
            int: Amount of memory freed in MB (0 if not inflated)
        """
        if not self.is_inflated:
            if verbose:
                logger.info(f"GPU {self.gpu_id}: Balloon not inflated, skipping")
            return 0
        return self.deflate(verbose=verbose)

    def resize_balloon(self, new_reserve_mb: Optional[int] = None,
                      new_reserve_fraction: Optional[float] = None,
                      verbose: bool = True) -> int:
        """
        Resize the balloon by deflating and re-inflating with new size.

        Args:
            new_reserve_mb: New amount of memory to reserve in MB
            new_reserve_fraction: New fraction of free memory to reserve
            verbose: If True, log resize details

        Returns:
            int: Amount of memory allocated in MB after resize

        Raises:
            ValueError: If both or neither parameters are specified
        """
        if (new_reserve_mb is None) == (new_reserve_fraction is None):
            raise ValueError("Specify exactly one of new_reserve_mb or new_reserve_fraction")

        # Deflate if currently inflated
        if self.is_inflated:
            self.deflate(verbose=verbose)

        # Update configuration
        if new_reserve_mb is not None:
            self.reserve_mb = new_reserve_mb
            self.reserve_fraction = None
        else:
            self.reserve_mb = None
            self.reserve_fraction = new_reserve_fraction

        # Inflate with new size
        return self.inflate(verbose=verbose)

    def set_chunk_size(self, chunk_size_mb: int):
        """
        Set the chunk size for memory allocation.

        Args:
            chunk_size_mb: New chunk size in MB

        Note:
            This will only affect future inflate() calls, not current allocations.
            To apply immediately, call resize_balloon() after changing chunk size.
        """
        if chunk_size_mb <= 0:
            raise ValueError("chunk_size_mb must be positive")
        self.chunk_size_mb = chunk_size_mb
        logger.info(f"GPU {self.gpu_id}: Chunk size set to {chunk_size_mb} MB")

    def allocate_memory(self, amount: Union[int, str, float], verbose: bool = True) -> int:
        """
        Allocate specific amount of memory with flexible specification.

        This is a more flexible alternative to inflate() that accepts various formats.

        Args:
            amount: Memory to allocate, supports:
                - int: MB (e.g., 2000 = 2GB)
                - str: "2GB", "2048MB", "50%", etc.
                - float: fraction 0.0-1.0 (e.g., 0.5 = 50% of free)
            verbose: If True, log allocation details

        Returns:
            int: Amount of memory allocated in MB

        Examples:
            >>> balloon.allocate_memory("2GB")     # Allocate 2 GB
            >>> balloon.allocate_memory(2048)      # Allocate 2048 MB
            >>> balloon.allocate_memory("50%")     # Allocate 50% of free memory
            >>> balloon.allocate_memory(0.7)       # Allocate 70% of free memory

        Raises:
            RuntimeError: If already inflated
        """
        if self.is_inflated:
            raise RuntimeError(f"Balloon on GPU {self.gpu_id} is already inflated. "
                             f"Use allocate_additional() to add more memory.")

        # Parse amount to MB
        free_mb = self._get_free_memory_mb()
        total_mb = self._get_total_memory_mb()
        target_mb = parse_memory_spec(amount, total_memory_mb=total_mb, free_memory_mb=free_mb)

        # Temporarily set reserve_mb and inflate
        old_reserve_mb = self.reserve_mb
        old_reserve_fraction = self.reserve_fraction

        self.reserve_mb = target_mb
        self.reserve_fraction = None

        try:
            result = self.inflate(verbose=verbose)
            return result
        finally:
            # Restore original settings
            self.reserve_mb = old_reserve_mb
            self.reserve_fraction = old_reserve_fraction

    def allocate_additional(self, amount: Union[int, str, float], verbose: bool = True) -> int:
        """
        Allocate additional memory on top of existing allocation.

        Args:
            amount: Additional memory to allocate, supports:
                - int: MB (e.g., 1000 = 1GB more)
                - str: "1GB", "1024MB", "20%", etc.
                - float: fraction 0.0-1.0 (e.g., 0.2 = 20% more of current free)
            verbose: If True, log allocation details

        Returns:
            int: Amount of additional memory allocated in MB

        Examples:
            >>> balloon.allocate_memory("2GB")         # Initial: 2GB
            >>> balloon.allocate_additional("1GB")     # Now: 3GB total
            >>> balloon.allocate_additional("50%")     # Add 50% of remaining free
        """
        torch.cuda.set_device(self.gpu_id)

        # Parse amount to MB
        free_mb = self._get_free_memory_mb()
        total_mb = self._get_total_memory_mb()
        additional_mb = parse_memory_spec(amount, total_memory_mb=total_mb, free_memory_mb=free_mb)

        if verbose:
            logger.info(f"GPU {self.gpu_id}: Allocating additional {additional_mb} MB "
                       f"(current: {self.total_allocated_mb} MB)")

        if additional_mb > free_mb:
            logger.warning(f"GPU {self.gpu_id}: Requested {additional_mb} MB but only {free_mb} MB available. "
                          f"Allocating {max(0, free_mb - 100)} MB instead.")
            additional_mb = max(0, free_mb - 100)

        # Allocate additional chunks
        allocated_mb = 0
        try:
            while allocated_mb < additional_mb:
                remaining_mb = additional_mb - allocated_mb
                chunk_mb = min(self.chunk_size_mb, remaining_mb)

                # Allocate chunk
                elements = (chunk_mb * 1024 * 1024) // 4
                chunk_tensor = torch.empty(elements, dtype=torch.float32, device=f'cuda:{self.gpu_id}')

                self.balloons.append(chunk_tensor)
                allocated_mb += chunk_mb
                self.total_allocated_mb += chunk_mb

            self.is_inflated = True

            if verbose:
                logger.info(f"GPU {self.gpu_id}: Successfully allocated additional {allocated_mb} MB. "
                           f"Total: {self.total_allocated_mb} MB")

            return allocated_mb

        except RuntimeError as e:
            logger.error(f"GPU {self.gpu_id}: Failed to allocate additional memory: {e}")
            raise RuntimeError(f"Failed to allocate additional {additional_mb} MB on GPU {self.gpu_id}: {e}")

    def allocate_to_target(self, target: Union[int, str, float], verbose: bool = True) -> int:
        """
        Allocate or deallocate to reach a target total allocation.

        Args:
            target: Target total memory allocation, supports:
                - int: MB (e.g., 5000 = 5GB total)
                - str: "5GB", "5000MB", "80%", etc.
                - float: fraction 0.0-1.0 (e.g., 0.8 = 80% of free)
            verbose: If True, log allocation details

        Returns:
            int: Amount of memory change in MB (positive = allocated, negative = freed)

        Examples:
            >>> balloon.allocate_memory("2GB")       # Current: 2GB
            >>> balloon.allocate_to_target("5GB")    # Allocates 3GB more -> Total: 5GB
            >>> balloon.allocate_to_target("1GB")    # Frees 4GB -> Total: 1GB
        """
        torch.cuda.set_device(self.gpu_id)

        # Parse target to MB
        free_mb = self._get_free_memory_mb()
        total_mb = self._get_total_memory_mb()
        target_mb = parse_memory_spec(target, total_memory_mb=total_mb, free_memory_mb=free_mb)

        current_mb = self.total_allocated_mb
        difference_mb = target_mb - current_mb

        if verbose:
            logger.info(f"GPU {self.gpu_id}: Current allocation: {current_mb} MB, "
                       f"Target: {target_mb} MB, Difference: {difference_mb:+d} MB")

        if difference_mb > 0:
            # Need to allocate more
            if not self.is_inflated:
                # First allocation
                return self.allocate_memory(target_mb, verbose=verbose)
            else:
                # Additional allocation
                return self.allocate_additional(difference_mb, verbose=verbose)
        elif difference_mb < 0:
            # Need to deallocate (resize down)
            if target_mb <= 0:
                self.deflate(verbose=verbose)
                return difference_mb
            else:
                self.resize_balloon(new_reserve_mb=target_mb, verbose=verbose)
                return difference_mb
        else:
            # Already at target
            if verbose:
                logger.info(f"GPU {self.gpu_id}: Already at target allocation {target_mb} MB")
            return 0

    def get_status(self) -> Dict:
        """
        Get current status of the balloon.

        Returns:
            Dictionary with status information
        """
        torch.cuda.set_device(self.gpu_id)
        return {
            'gpu_id': self.gpu_id,
            'is_inflated': self.is_inflated,
            'allocated_mb': self.total_allocated_mb,
            'num_chunks': len(self.balloons),
            'free_memory_mb': self._get_free_memory_mb(),
            'total_memory_mb': self._get_total_memory_mb()
        }

    def __enter__(self):
        """Context manager entry - inflate the balloon."""
        self.inflate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - deflate the balloon."""
        self.deflate()
        return False

    def __del__(self):
        """Destructor - ensure memory is released."""
        # Check if object was fully initialized
        if hasattr(self, 'is_inflated') and self.is_inflated:
            try:
                self.deflate(verbose=False)
            except:
                pass


class MultiGPUMemoryBalloon:
    """
    Memory ballooning for multiple GPUs.

    Manages multiple GPUMemoryBalloon instances across different GPUs.
    """

    def __init__(self, gpu_ids: List[int], reserve_mb_per_gpu: Optional[Union[int, str, List[Union[int, str]]]] = None,
                 reserve_fraction_per_gpu: Optional[Union[float, str, List[Union[float, str]]]] = None,
                 chunk_size_mb: int = 100):
        """
        Initialize multi-GPU memory balloon.

        Args:
            gpu_ids: List of GPU device IDs
            reserve_mb_per_gpu: Memory to reserve per GPU in MB (int/str for same on all, list for per-GPU)
                               Supports: int, "2GB", "2048MB", ["2GB", "1GB", ...], etc.
            reserve_fraction_per_gpu: Fraction of memory to reserve per GPU (float/str for same, list for per-GPU)
                                     Supports: float, "50%", [0.5, 0.7, ...], ["50%", "70%", ...], etc.
            chunk_size_mb: Size of each allocation chunk in MB

        Raises:
            ValueError: If invalid configuration
        """
        if not gpu_ids:
            raise ValueError("gpu_ids cannot be empty")

        self.gpu_ids = gpu_ids
        self.balloons: Dict[int, GPUMemoryBalloon] = {}

        # Normalize reserve parameters
        if reserve_mb_per_gpu is not None:
            if isinstance(reserve_mb_per_gpu, (int, str)):
                # Single value for all GPUs - parse it first if string
                if isinstance(reserve_mb_per_gpu, str):
                    # Get a sample GPU to determine memory for parsing
                    torch.cuda.set_device(gpu_ids[0])
                    free_mb = torch.cuda.mem_get_info(gpu_ids[0])[0] // (1024 ** 2)
                    total_mb = torch.cuda.mem_get_info(gpu_ids[0])[1] // (1024 ** 2)
                    parsed_mb = parse_memory_spec(reserve_mb_per_gpu, total_memory_mb=total_mb, free_memory_mb=free_mb)
                    reserve_mb_list = [parsed_mb] * len(gpu_ids)
                else:
                    reserve_mb_list = [reserve_mb_per_gpu] * len(gpu_ids)
            else:
                # List of values - parse each if string
                reserve_mb_list = []
                for i, spec in enumerate(reserve_mb_per_gpu):
                    if isinstance(spec, str):
                        torch.cuda.set_device(gpu_ids[i])
                        free_mb = torch.cuda.mem_get_info(gpu_ids[i])[0] // (1024 ** 2)
                        total_mb = torch.cuda.mem_get_info(gpu_ids[i])[1] // (1024 ** 2)
                        reserve_mb_list.append(parse_memory_spec(spec, total_memory_mb=total_mb, free_memory_mb=free_mb))
                    else:
                        reserve_mb_list.append(spec)
            reserve_fraction_list = [None] * len(gpu_ids)
        elif reserve_fraction_per_gpu is not None:
            if isinstance(reserve_fraction_per_gpu, (float, str)):
                # Single value for all GPUs - parse if string
                if isinstance(reserve_fraction_per_gpu, str):
                    # Parse percentage string (e.g., "50%")
                    if reserve_fraction_per_gpu.strip().endswith('%'):
                        percentage = float(reserve_fraction_per_gpu.strip()[:-1])
                        reserve_fraction_list = [percentage / 100.0] * len(gpu_ids)
                    else:
                        reserve_fraction_list = [float(reserve_fraction_per_gpu)] * len(gpu_ids)
                else:
                    reserve_fraction_list = [reserve_fraction_per_gpu] * len(gpu_ids)
            else:
                # List of values
                reserve_fraction_list = []
                for spec in reserve_fraction_per_gpu:
                    if isinstance(spec, str) and spec.strip().endswith('%'):
                        percentage = float(spec.strip()[:-1])
                        reserve_fraction_list.append(percentage / 100.0)
                    else:
                        reserve_fraction_list.append(float(spec) if isinstance(spec, str) else spec)
            reserve_mb_list = [None] * len(gpu_ids)
        else:
            raise ValueError("Specify either reserve_mb_per_gpu or reserve_fraction_per_gpu")

        # Create balloons for each GPU
        for i, gpu_id in enumerate(gpu_ids):
            self.balloons[gpu_id] = GPUMemoryBalloon(
                gpu_id=gpu_id,
                reserve_mb=reserve_mb_list[i],
                reserve_fraction=reserve_fraction_list[i],
                chunk_size_mb=chunk_size_mb
            )

        logger.info(f"Initialized MultiGPUMemoryBalloon for GPUs: {gpu_ids}")

    def inflate_all(self, verbose: bool = True) -> Dict[int, int]:
        """
        Inflate balloons on all GPUs.

        Args:
            verbose: If True, log allocation details

        Returns:
            Dictionary mapping GPU ID to allocated memory in MB
        """
        results = {}
        for gpu_id, balloon in self.balloons.items():
            try:
                allocated = balloon.inflate(verbose=verbose)
                results[gpu_id] = allocated
            except Exception as e:
                logger.error(f"Failed to inflate balloon on GPU {gpu_id}: {e}")
                # Continue with other GPUs
                results[gpu_id] = 0

        return results

    def deflate_all(self, verbose: bool = True) -> Dict[int, int]:
        """
        Deflate balloons on all GPUs.

        Args:
            verbose: If True, log deallocation details

        Returns:
            Dictionary mapping GPU ID to freed memory in MB
        """
        results = {}
        for gpu_id, balloon in self.balloons.items():
            try:
                freed = balloon.deflate(verbose=verbose)
                results[gpu_id] = freed
            except Exception as e:
                logger.error(f"Failed to deflate balloon on GPU {gpu_id}: {e}")
                results[gpu_id] = 0

        return results

    def inflate_gpu(self, gpu_id: int, verbose: bool = True) -> int:
        """Inflate balloon on a specific GPU."""
        if gpu_id not in self.balloons:
            raise ValueError(f"GPU {gpu_id} not in managed GPUs: {self.gpu_ids}")
        return self.balloons[gpu_id].inflate(verbose=verbose)

    def deflate_gpu(self, gpu_id: int, verbose: bool = True) -> int:
        """Deflate balloon on a specific GPU."""
        if gpu_id not in self.balloons:
            raise ValueError(f"GPU {gpu_id} not in managed GPUs: {self.gpu_ids}")
        return self.balloons[gpu_id].deflate(verbose=verbose)

    def get_status_all(self) -> Dict[int, Dict]:
        """Get status for all GPUs."""
        return {gpu_id: balloon.get_status() for gpu_id, balloon in self.balloons.items()}

    def __enter__(self):
        """Context manager entry - inflate all balloons."""
        self.inflate_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - deflate all balloons."""
        self.deflate_all()
        return False


def get_all_available_gpus() -> List[int]:
    """
    Get list of all available GPU IDs.

    Returns:
        List of GPU IDs (e.g., [0, 1, 2, 3])

    Raises:
        RuntimeError: If CUDA is not available
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    return list(range(torch.cuda.device_count()))


def get_cuda_visible_devices() -> List[int]:
    """
    Get list of GPU IDs visible to the current process via CUDA_VISIBLE_DEVICES.

    Returns:
        List of GPU IDs available to this process
    """
    import os
    if not torch.cuda.is_available():
        return []

    # If CUDA_VISIBLE_DEVICES is set, use those
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible is not None:
        try:
            # Parse CUDA_VISIBLE_DEVICES
            visible_ids = [int(x.strip()) for x in cuda_visible.split(',') if x.strip()]
            return visible_ids
        except ValueError:
            # If parsing fails, fall back to all devices
            pass

    # Otherwise return all available devices
    return get_all_available_gpus()


def create_balloon_for_all_gpus(reserve_mb_per_gpu: Optional[Union[int, str, List[Union[int, str]]]] = None,
                                reserve_fraction_per_gpu: Optional[Union[float, str, List[Union[float, str]]]] = None,
                                chunk_size_mb: int = 100,
                                use_cuda_visible_devices: bool = True) -> MultiGPUMemoryBalloon:
    """
    Create a MultiGPUMemoryBalloon for all available GPUs.

    Args:
        reserve_mb_per_gpu: Memory to reserve per GPU in MB
        reserve_fraction_per_gpu: Fraction of memory to reserve per GPU
        chunk_size_mb: Size of each allocation chunk in MB
        use_cuda_visible_devices: If True, only use GPUs visible to this process.
                                  If False, use all GPUs in the system.

    Returns:
        MultiGPUMemoryBalloon instance configured for all available GPUs

    Example:
        # Reserve 2GB on all available GPUs
        balloon = create_balloon_for_all_gpus(reserve_mb_per_gpu=2000)
        balloon.inflate_all()
    """
    if use_cuda_visible_devices:
        gpu_ids = get_cuda_visible_devices()
    else:
        gpu_ids = get_all_available_gpus()

    if not gpu_ids:
        raise RuntimeError("No GPUs available")

    logger.info(f"Creating balloon for all available GPUs: {gpu_ids}")

    return MultiGPUMemoryBalloon(
        gpu_ids=gpu_ids,
        reserve_mb_per_gpu=reserve_mb_per_gpu,
        reserve_fraction_per_gpu=reserve_fraction_per_gpu,
        chunk_size_mb=chunk_size_mb
    )


@contextmanager
def reserve_gpu_memory(gpu_id: int, reserve_mb: Optional[int] = None,
                       reserve_fraction: Optional[float] = None,
                       chunk_size_mb: int = 100):
    """
    Context manager for temporarily reserving GPU memory.

    Args:
        gpu_id: GPU device ID
        reserve_mb: Amount of memory to reserve in MB
        reserve_fraction: Fraction of free memory to reserve
        chunk_size_mb: Size of each allocation chunk in MB

    Example:
        with reserve_gpu_memory(gpu_id=0, reserve_mb=2000):
            # 2GB is reserved on GPU 0
            model = MyModel().cuda()
            # ... training ...
        # Memory automatically released
    """
    balloon = GPUMemoryBalloon(gpu_id=gpu_id, reserve_mb=reserve_mb,
                               reserve_fraction=reserve_fraction,
                               chunk_size_mb=chunk_size_mb)
    try:
        balloon.inflate()
        yield balloon
    finally:
        balloon.deflate()


@contextmanager
def reserve_multi_gpu_memory(gpu_ids: List[int], reserve_mb_per_gpu: Optional[Union[int, str, List[Union[int, str]]]] = None,
                             reserve_fraction_per_gpu: Optional[Union[float, str, List[Union[float, str]]]] = None,
                             chunk_size_mb: int = 100):
    """
    Context manager for temporarily reserving memory on multiple GPUs.

    Args:
        gpu_ids: List of GPU device IDs
        reserve_mb_per_gpu: Memory to reserve per GPU in MB
        reserve_fraction_per_gpu: Fraction of memory to reserve per GPU
        chunk_size_mb: Size of each allocation chunk in MB

    Example:
        with reserve_multi_gpu_memory(gpu_ids=[0, 1, 2], reserve_mb_per_gpu=2000):
            # 2GB is reserved on each GPU
            # ... multi-GPU training ...
        # Memory automatically released on all GPUs
    """
    multi_balloon = MultiGPUMemoryBalloon(
        gpu_ids=gpu_ids,
        reserve_mb_per_gpu=reserve_mb_per_gpu,
        reserve_fraction_per_gpu=reserve_fraction_per_gpu,
        chunk_size_mb=chunk_size_mb
    )
    try:
        multi_balloon.inflate_all()
        yield multi_balloon
    finally:
        multi_balloon.deflate_all()


@contextmanager
def reserve_all_gpus_memory(reserve_mb_per_gpu: Optional[Union[int, str, List[Union[int, str]]]] = None,
                           reserve_fraction_per_gpu: Optional[Union[float, str, List[Union[float, str]]]] = None,
                           chunk_size_mb: int = 100,
                           use_cuda_visible_devices: bool = True):
    """
    Context manager for temporarily reserving memory on all available GPUs.

    Args:
        reserve_mb_per_gpu: Memory to reserve per GPU in MB
        reserve_fraction_per_gpu: Fraction of memory to reserve per GPU
        chunk_size_mb: Size of each allocation chunk in MB
        use_cuda_visible_devices: If True, only use GPUs visible to this process

    Example:
        # Reserve 2GB on all GPUs
        with reserve_all_gpus_memory(reserve_mb_per_gpu=2000):
            # Memory reserved on all GPUs
            # ... training ...
        # Memory automatically released

        # Reserve 50% of free memory on all GPUs
        with reserve_all_gpus_memory(reserve_fraction_per_gpu=0.5):
            # ... training ...
    """
    multi_balloon = create_balloon_for_all_gpus(
        reserve_mb_per_gpu=reserve_mb_per_gpu,
        reserve_fraction_per_gpu=reserve_fraction_per_gpu,
        chunk_size_mb=chunk_size_mb,
        use_cuda_visible_devices=use_cuda_visible_devices
    )
    try:
        multi_balloon.inflate_all()
        yield multi_balloon
    finally:
        multi_balloon.deflate_all()


if __name__ == "__main__":
    # Example usage and testing
    import sys

    # Setup logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not torch.cuda.is_available():
        print("CUDA not available, cannot run examples")
        sys.exit(1)

    print("=== GPU Memory Ballooning Examples ===\n")

    # Example 1: Single GPU with fixed size
    print("Example 1: Reserve 1000 MB on GPU 0")
    balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=1000)
    balloon.inflate()
    print(f"Status: {balloon.get_status()}")
    balloon.deflate()
    print()

    # Example 2: Single GPU with fraction
    print("Example 2: Reserve 50% of free memory on GPU 0")
    balloon = GPUMemoryBalloon(gpu_id=0, reserve_fraction=0.5)
    balloon.inflate()
    print(f"Status: {balloon.get_status()}")
    balloon.deflate()
    print()

    # Example 3: Context manager
    print("Example 3: Using context manager")
    with GPUMemoryBalloon(gpu_id=0, reserve_mb=500) as balloon:
        print(f"Inside context: {balloon.get_status()}")
    print("Memory automatically released\n")

    # Example 4: Conditional inflate/deflate
    print("Example 4: Conditional inflate/deflate")
    balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=500)
    balloon.inflate_if_not_inflated()  # Will inflate
    balloon.inflate_if_not_inflated()  # Will skip (already inflated)
    balloon.deflate_if_inflated()      # Will deflate
    balloon.deflate_if_inflated()      # Will skip (not inflated)
    print()

    # Example 5: Resize balloon
    print("Example 5: Resize balloon")
    balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=500)
    balloon.inflate()
    print(f"Initial: {balloon.get_status()['allocated_mb']} MB")
    balloon.resize_balloon(new_reserve_mb=1000)
    print(f"After resize: {balloon.get_status()['allocated_mb']} MB")
    balloon.deflate()
    print()

    # Example 6: Custom chunk size
    print("Example 6: Custom chunk size (200 MB chunks)")
    balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=1000, chunk_size_mb=200)
    balloon.inflate()
    print(f"Allocated with {balloon.get_status()['num_chunks']} chunks")
    balloon.deflate()
    print()

    # Example 7: All available GPUs
    print(f"Example 7: Reserve on all available GPUs")
    all_gpus = get_all_available_gpus()
    print(f"Available GPUs: {all_gpus}")
    balloon_all = create_balloon_for_all_gpus(reserve_mb_per_gpu=500)
    balloon_all.inflate_all()
    print(f"Status: {balloon_all.get_status_all()}")
    balloon_all.deflate_all()
    print()

    # Example 8: Context manager for all GPUs
    print(f"Example 8: Context manager for all GPUs")
    with reserve_all_gpus_memory(reserve_mb_per_gpu=300) as balloon:
        status = balloon.get_status_all()
        print(f"Reserved on {len(status)} GPUs")
    print("Memory automatically released on all GPUs\n")

    # Example 9: Multi-GPU with specific GPUs
    if torch.cuda.device_count() > 1:
        print(f"Example 9: Reserve memory on specific GPUs")
        multi_balloon = MultiGPUMemoryBalloon(
            gpu_ids=[0, 1],
            reserve_mb_per_gpu=500
        )
        multi_balloon.inflate_all()
        print(f"Status: {multi_balloon.get_status_all()}")
        multi_balloon.deflate_all()

    print("\n✅ All examples completed successfully!")
