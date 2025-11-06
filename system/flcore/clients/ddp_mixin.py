import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp


class DDPMixin:
    """
    Mixin class to add PyTorch DistributedDataParallel support to federated learning clients.
    
    This mixin provides distributed training capabilities while maintaining compatibility 
    with the existing federated learning architecture.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # DDP-specific attributes
        self.is_distributed = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.process_group = None
        self.ddp_backend = 'nccl'
        self.ddp_init_method = 'env://'
        self.visible_gpus = None  # List of GPU IDs to use
        self.gpu_mapping = None   # Mapping from local_rank to actual GPU ID
        
        # Initialize distributed training if environment variables are set
        self._init_distributed_if_available()
    
    def _init_distributed_if_available(self):
        """
        Initialize distributed training if environment variables are present.
        Expected environment variables:
        - RANK: Global rank of the process
        - LOCAL_RANK: Local rank within the node
        - WORLD_SIZE: Total number of processes
        - MASTER_ADDR: Address of the master node
        - MASTER_PORT: Port of the master node
        - CUDA_VISIBLE_DEVICES: Optional GPU selection (e.g., "0,2,3")
        """
        # Check if all required DDP environment variables are set
        required_vars = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
        if all(env_var in os.environ for env_var in required_vars):
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            
            # Parse visible GPUs from environment
            self._parse_visible_gpus()
            
            if self.world_size > 1:
                self.init_distributed()
        else:
            # DDP environment not properly set, stay in single process mode
            missing_vars = [var for var in required_vars if var not in os.environ]
            if len(missing_vars) > 0:
                print(f"Client {getattr(self, 'id', 'unknown')}: DDP not initialized - missing environment variables: {missing_vars}")
                print(f"Client {getattr(self, 'id', 'unknown')}: Running in single process mode")
            
            # Set default single process values
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self._parse_visible_gpus()
    
    def _parse_visible_gpus(self):
        """
        Parse CUDA_VISIBLE_DEVICES to determine available GPUs.
        Creates mapping from local_rank to actual GPU ID.
        """
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        
        if cuda_visible is not None:
            # Parse comma-separated GPU IDs
            self.visible_gpus = [int(gpu.strip()) for gpu in cuda_visible.split(',') if gpu.strip().isdigit()]
            print(f"Client {getattr(self, 'id', 'unknown')}: Visible GPUs: {self.visible_gpus}")
            
            # Create mapping from local_rank to actual GPU ID
            if len(self.visible_gpus) > 0:
                self.gpu_mapping = {}
                for local_rank in range(len(self.visible_gpus)):
                    self.gpu_mapping[local_rank] = self.visible_gpus[local_rank]
                print(f"Client {getattr(self, 'id', 'unknown')}: GPU mapping: {self.gpu_mapping}")
        else:
            # Use all available GPUs
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.visible_gpus = list(range(gpu_count))
                self.gpu_mapping = {i: i for i in range(gpu_count)}
                print(f"Client {getattr(self, 'id', 'unknown')}: Using all {gpu_count} GPUs")
    
    def get_assigned_gpu(self):
        """
        Get the actual GPU ID for this process based on local_rank and visible GPUs.
        
        Returns:
            int: Actual GPU ID to use, or None if no GPU assignment
        """
        if not torch.cuda.is_available():
            return None
        
        if self.gpu_mapping and self.local_rank in self.gpu_mapping:
            return self.gpu_mapping[self.local_rank]
        elif self.visible_gpus and self.local_rank < len(self.visible_gpus):
            return self.visible_gpus[self.local_rank]
        else:
            # Fallback: use local_rank directly
            return self.local_rank
    
    def init_distributed(self, backend='nccl', init_method='env://'):
        """
        Initialize the distributed process group.
        
        Args:
            backend (str): Backend to use ('nccl' for GPU, 'gloo' for CPU)
            init_method (str): Initialization method for the process group
        """
        if self.is_distributed:
            print(f"Client {self.id}: DDP already initialized")
            return
        
        try:
            # Use CUDA if available and backend is nccl
            if backend == 'nccl' and torch.cuda.is_available():
                assigned_gpu = self.get_assigned_gpu()
                if assigned_gpu is not None:
                    torch.cuda.set_device(assigned_gpu)
                    self.device = f'cuda:{assigned_gpu}'
                    print(f"Client {self.id}: Assigned to GPU {assigned_gpu} (local_rank {self.local_rank})")
                else:
                    print(f"Client {self.id}: No GPU assigned, falling back to CPU")
                    self.device = 'cpu'
                    backend = 'gloo'
            elif backend == 'gloo':
                self.device = 'cpu'
            
            # Initialize the process group
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=self.world_size,
                rank=self.rank
            )
            
            self.process_group = dist.group.WORLD
            self.is_distributed = True
            self.ddp_backend = backend
            self.ddp_init_method = init_method
            
            print(f"Client {self.id}: DDP initialized - Rank {self.rank}/{self.world_size}, Device: {self.device}")
            
        except Exception as e:
            print(f"Client {self.id}: Failed to initialize DDP: {e}")
            self.is_distributed = False
            self.world_size = 1
            self.rank = 0
    
    def wrap_model_with_ddp(self):
        """
        Wrap the model with DistributedDataParallel if distributed training is enabled.
        Should be called after model initialization.
        """
        if not self.is_distributed:
            return
        
        try:
            # Move model to the appropriate device
            self.model.to(self.device)
            
            # Get the actual GPU ID for DDP wrapping
            assigned_gpu = self.get_assigned_gpu()
            device_ids = [assigned_gpu] if (self.ddp_backend == 'nccl' and assigned_gpu is not None) else None
            output_device = assigned_gpu if (self.ddp_backend == 'nccl' and assigned_gpu is not None) else None
            
            # Wrap main model with DDP
            if hasattr(self.model, 'inner_model'):
                self.model.inner_model = DDP(
                    self.model.inner_model,
                    device_ids=device_ids,
                    output_device=output_device,
                    process_group=self.process_group
                )
            elif hasattr(self.model, 'backbone'):
                self.model.backbone = DDP(
                    self.model.backbone,
                    device_ids=device_ids,
                    output_device=output_device,
                    process_group=self.process_group
                )
            else:
                self.model = DDP(
                    self.model,
                    device_ids=device_ids,
                    output_device=output_device,
                    process_group=self.process_group
                )
            
            print(f"Client {self.id}: Model wrapped with DDP")
            
        except Exception as e:
            print(f"Client {self.id}: Failed to wrap model with DDP: {e}")
    
    def create_distributed_dataloader(self, dataset, batch_size, shuffle=True, **kwargs):
        """
        Create a DataLoader with DistributedSampler for distributed training.
        
        Args:
            dataset: Dataset to create loader for
            batch_size: Batch size per process
            shuffle: Whether to shuffle data
            **kwargs: Additional DataLoader arguments
            
        Returns:
            DataLoader with DistributedSampler if distributed, regular DataLoader otherwise
        """
        if self.is_distributed and dataset is not None:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle
            )
            
            # Remove shuffle from kwargs since DistributedSampler handles it
            dataloader_kwargs = {k: v for k, v in kwargs.items() if k != 'shuffle'}
            
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                **dataloader_kwargs
            )
        else:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                **kwargs
            )
    
    def load_train_data(self, batch_size=None, dataset_limit=0):
        """Override to use distributed sampler if DDP is enabled."""
        if batch_size is None:
            batch_size = self.batch_size
        
        if self.is_distributed:
            # Get the dataset from node_data
            train_dataset = self.node_data.get_train_dataset(dataset_limit)
            if train_dataset is not None:
                loader = self.create_distributed_dataloader(
                    train_dataset, 
                    batch_size, 
                    shuffle=True
                )
                self.train_samples = len(train_dataset)
                return loader
        
        # Fallback to original implementation
        return super().load_train_data(batch_size, dataset_limit)
    
    def load_test_data(self, batch_size=None, dataset_limit=0):
        """Override to use distributed sampler if DDP is enabled."""
        if batch_size is None:
            batch_size = self.batch_size
        
        if self.is_distributed:
            # Get the dataset from node_data
            test_dataset = self.node_data.get_test_dataset(dataset_limit)
            if test_dataset is not None:
                loader = self.create_distributed_dataloader(
                    test_dataset, 
                    batch_size, 
                    shuffle=False  # No shuffle for test data
                )
                self.test_samples = len(test_dataset)
                return loader
        
        # Fallback to original implementation
        return super().load_test_data(batch_size, dataset_limit)
    
    def sync_model_parameters(self):
        """
        Synchronize model parameters across all DDP processes.
        Useful for ensuring consistency before federated aggregation.
        """
        if not self.is_distributed:
            return
        
        try:
            # Synchronize all model parameters
            if hasattr(self.model, 'module'):  # DDP wrapped model
                for param in self.model.module.parameters():
                    dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                    param.data /= self.world_size
            else:
                for param in self.model.parameters():
                    dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                    param.data /= self.world_size
                    
            # Ensure all processes are synchronized
            dist.barrier()
            
        except Exception as e:
            print(f"Client {self.id}: Failed to sync parameters: {e}")
    
    def _move_to_gpu(self, device):
        """Enhanced GPU movement for DDP."""
        if self.is_distributed:
            # For DDP, use the assigned GPU based on visible devices
            assigned_gpu = self.get_assigned_gpu()
            if assigned_gpu is not None:
                ddp_device = f'cuda:{assigned_gpu}'
                print(f"Node {self.id} moving to DDP GPU: {ddp_device} (local_rank {self.local_rank})")
                self._move_to_device(ddp_device)
            else:
                print(f"Node {self.id} no GPU assigned, moving to CPU")
                self._move_to_device('cpu')
        else:
            # Fallback to original implementation
            super()._move_to_gpu(device)
    
    def set_epoch(self, epoch):
        """
        Set the current epoch for distributed samplers.
        This ensures proper shuffling across epochs in distributed training.
        """
        if self.is_distributed and hasattr(self, 'train_dataloader'):
            if hasattr(self.train_dataloader, 'sampler') and isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)
    
    def cleanup_distributed(self):
        """Clean up distributed training resources."""
        if self.is_distributed:
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
                print(f"Client {self.id}: DDP cleanup completed")
            except Exception as e:
                print(f"Client {self.id}: Error during DDP cleanup: {e}")
            finally:
                self.is_distributed = False
                self.process_group = None
    
    def __del__(self):
        """Ensure cleanup on object destruction."""
        if hasattr(self, 'is_distributed') and self.is_distributed:
            self.cleanup_distributed()