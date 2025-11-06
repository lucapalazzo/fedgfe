from flcore.clients.clientgfe import clientGFE
from flcore.clients.ddp_mixin import DDPMixin
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


class clientGFEDDP(DDPMixin, clientGFE):
    """
    Federated Learning client with DistributedDataParallel support.
    
    This class extends clientGFE with DDP capabilities for multi-GPU distributed training
    while maintaining full compatibility with the federated learning framework.
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize both parent classes
        super().__init__(*args, **kwargs)
        
        # Wrap model with DDP if distributed training is enabled
        if self.is_distributed:
            self.wrap_model_with_ddp()
            print(f"Client {self.id}: Initialized with DDP support (Rank {self.rank}/{self.world_size})")
        else:
            print(f"Client {self.id}: Initialized without DDP (single process)")
    
    def wrap_model_with_ddp(self):
        """
        Override the mixin method to handle VITFC model structure specifically.
        """
        if not self.is_distributed:
            return
        
        try:
            # Move model to appropriate device first
            self.model.to(self.device)
            
            # For VITFC models, wrap the backbone specifically
            if hasattr(self.model, 'backbone') and self.model.backbone is not None:
                # Wrap backbone with DDP
                self.model.backbone = DDP(
                    self.model.backbone,
                    device_ids=[self.local_rank] if self.ddp_backend == 'nccl' else None,
                    output_device=self.local_rank if self.ddp_backend == 'nccl' else None,
                    process_group=self.process_group,
                    find_unused_parameters=True  # Important for federated learning
                )
                print(f"Client {self.id}: Backbone wrapped with DDP")
                
            # Also wrap downstream task if it exists
            if hasattr(self.model, 'downstream_task') and self.model.downstream_task is not None:
                self.model.downstream_task = DDP(
                    self.model.downstream_task,
                    device_ids=[self.local_rank] if self.ddp_backend == 'nccl' else None,
                    output_device=self.local_rank if self.ddp_backend == 'nccl' else None,
                    process_group=self.process_group,
                    find_unused_parameters=True
                )
                print(f"Client {self.id}: Downstream task wrapped with DDP")
            
            # Wrap pretext tasks if they exist
            if hasattr(self.model, 'pretext_tasks') and self.model.pretext_tasks:
                for i, pretext_task in enumerate(self.model.pretext_tasks):
                    self.model.pretext_tasks[i] = DDP(
                        pretext_task,
                        device_ids=[self.local_rank] if self.ddp_backend == 'nccl' else None,
                        output_device=self.local_rank if self.ddp_backend == 'nccl' else None,
                        process_group=self.process_group,
                        find_unused_parameters=True
                    )
                print(f"Client {self.id}: {len(self.model.pretext_tasks)} pretext tasks wrapped with DDP")
                
        except Exception as e:
            print(f"Client {self.id}: Failed to wrap VITFC model with DDP: {e}")
            # Fallback to single process mode
            self.is_distributed = False
    
    def train(self, client_device=None, rewind_train_node=None, training_task="both"):
        """
        Override train method to handle DDP synchronization.
        """
        if self.is_distributed:
            # Set epoch for distributed samplers to ensure proper shuffling
            if hasattr(self, 'ssl_round'):
                self.set_epoch(self.ssl_round)
        
        # Call parent train method
        result = super().train(client_device, rewind_train_node, training_task)
        
        if self.is_distributed:
            # Synchronize model parameters after local training
            # This ensures consistency before federated aggregation
            self.sync_model_parameters()
        
        return result
    
    def _move_to_device(self, device):
        """
        Override device movement to handle DDP wrapped models.
        """
        if self.is_distributed:
            # For DDP, we need to handle wrapped models differently
            if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'module'):
                # DDP wrapped backbone
                self.model.backbone.to(device)
            if hasattr(self.model, 'downstream_task') and hasattr(self.model.downstream_task, 'module'):
                # DDP wrapped downstream task
                self.model.downstream_task.to(device)
            if hasattr(self.model, 'pretext_tasks'):
                for pretext_task in self.model.pretext_tasks:
                    if hasattr(pretext_task, 'module'):
                        pretext_task.to(device)
        
        # Call parent implementation for non-DDP components
        super()._move_to_device(device)
    
    def set_parameters(self, model):
        """
        Override parameter setting to handle DDP wrapped models.
        """
        if self.is_distributed:
            # For DDP models, we need to access the underlying module
            target_model = self.model
            
            # Handle DDP wrapped backbone
            if hasattr(model, 'backbone') and hasattr(self.model, 'backbone'):
                if hasattr(self.model.backbone, 'module'):
                    # Target is DDP wrapped
                    target_backbone = self.model.backbone.module
                else:
                    target_backbone = self.model.backbone
                
                if hasattr(model.backbone, 'module'):
                    # Source is DDP wrapped
                    source_backbone = model.backbone.module
                else:
                    source_backbone = model.backbone
                
                # Copy parameters
                for new_param, old_param in zip(source_backbone.parameters(), target_backbone.parameters()):
                    old_param.data = new_param.data.clone()
        else:
            # Fallback to parent implementation
            super().set_parameters(model)
    
    def get_distributed_info(self):
        """
        Get information about the distributed setup.
        
        Returns:
            dict: Information about DDP configuration
        """
        return {
            'is_distributed': self.is_distributed,
            'world_size': self.world_size,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'backend': self.ddp_backend,
            'device': self.device,
            'process_group': self.process_group is not None
        }