import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR


class OptimizerManager:
    """Manages optimizer setup and learning rate scheduling for federated learning clients."""
    
    def __init__(self, optimizer_type="adamw", learning_rate=0.001, weight_decay=0.01, momentum=0.9):
        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        
        self.train_optimizer = None
        self.finetuning_optimizer = None
        self._scheduler = None
    
    def setup_optimizer(self, model, downstream_task=None, pretext_tasks=None, client_id=None):
        """Setup optimizers for model, downstream task, and pretext tasks."""
        print(f"Creating optimizer for node {client_id} with model optimizer {self.optimizer_type} and learning rate {self.learning_rate}")
        
        # Create base optimizers
        if self.optimizer_type == "adamw":
            self.train_optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
            self.finetuning_optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "sgd":
            self.train_optimizer = torch.optim.SGD(
                model.parameters(), 
                lr=self.learning_rate, 
                momentum=self.momentum, 
                weight_decay=self.weight_decay
            )
            self.finetuning_optimizer = torch.optim.SGD(
                model.parameters(), 
                lr=self.learning_rate, 
                momentum=self.momentum, 
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        
        # Add downstream task parameters
        if downstream_task is not None:
            self._add_downstream_task_params(downstream_task, client_id)
        
        # Add pretext task parameters
        if pretext_tasks is not None:
            self._add_pretext_task_params(model, pretext_tasks, client_id)
        
        return self.train_optimizer
    
    def _add_downstream_task_params(self, downstream_task, client_id):
        """Add downstream task parameters to optimizers."""
        task_name = getattr(downstream_task, '__class__', type(downstream_task)).__name__
        print(f"Node {client_id} adding downstream task {task_name} to optimizer")
        
        param_group = {
            'params': downstream_task.parameters(), 
            'lr': self.learning_rate, 
            'weight_decay': self.weight_decay
        }
        
        if self.optimizer_type == "sgd":
            param_group['momentum'] = self.momentum
        
        self.train_optimizer.add_param_group(param_group)
        self.finetuning_optimizer.add_param_group(param_group)
    
    def _add_pretext_task_params(self, model, pretext_tasks, client_id):
        """Add pretext task parameters to optimizers."""
        model.pretext_train = True
        
        for pretext_task in pretext_tasks:
            model.pretext_task_name = pretext_task
            
            # Use task-specific learning rates if available
            learning_rate = getattr(model, 'task_learning_rate', 0) or self.learning_rate
            weight_decay = getattr(model, 'task_weight_decay', 0) or self.weight_decay
            
            print(f"Node {client_id} adding pretext task {model.pretext_task_name} to optimizer")
            
            # Check if pretext task has custom optimizer adjustments
            params_modified = model.pretext_task.adjust_optimizer() if hasattr(model.pretext_task, 'adjust_optimizer') else None
            
            if params_modified is None:
                # Standard parameter group
                param_group = {
                    'params': model.pretext_task.parameters(), 
                    'lr': learning_rate, 
                    'weight_decay': weight_decay
                }
                
                if self.optimizer_type == "sgd":
                    param_group['momentum'] = self.momentum
                
                if self.optimizer_type == "adamw":
                    self.train_optimizer.add_param_group(param_group)
                elif self.optimizer_type == "sgd":
                    self.train_optimizer.add_param_group(param_group)
            else:
                # Custom parameter groups
                if self.optimizer_type == "adamw":
                    for param_group in params_modified:
                        self.train_optimizer.add_param_group(param_group)
                elif self.optimizer_type == "sgd":
                    # Fallback to standard approach for SGD
                    param_group = {
                        'params': model.pretext_task.parameters(), 
                        'lr': learning_rate, 
                        'weight_decay': weight_decay, 
                        'momentum': self.momentum
                    }
                    self.finetuning_optimizer.add_param_group(param_group)
        
        model.pretext_train = False
    
    def setup_learning_rate_scheduler(self, optimizer, rounds, use_scheduler=True):
        """Setup learning rate scheduler based on optimizer type."""
        if not use_scheduler:
            self.scheduler = ConstantLR(optimizer, factor=1.0, total_iters=rounds)
            return self.scheduler
        
        print("Creating learning rate scheduler for node")
        
        if self.optimizer_type == "adamw":
            self.scheduler = SequentialLR(
                optimizer,
                [
                    ConstantLR(optimizer, factor=1.0, total_iters=500),
                    CosineAnnealingLR(optimizer, T_max=rounds*10, eta_min=0.0),
                ],
                milestones=[500]
            )
        elif self.optimizer_type == "sgd":
            self.scheduler = SequentialLR(
                optimizer,
                [
                    LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=rounds // 4),
                    CosineAnnealingLR(optimizer, T_max=rounds // 2, eta_min=0.0)
                ],
                milestones=[rounds // 4]
            )
        
        return self.scheduler
    
    def print_optimizer_info(self, client_id):
        """Print information about optimizer parameter groups."""
        if self.train_optimizer is None:
            print(f"Node {client_id}: No optimizer initialized")
            return
        
        for param_group_index, param_group in enumerate(self.train_optimizer.param_groups):
            num_params = sum(p.numel() for p in param_group["params"] if p.requires_grad)
            size = sum(p.numel()*p.element_size() for p in param_group["params"] if p.requires_grad)
            print(f"Node {client_id} optimizer param group {param_group_index} "
                  f"tensors {len(param_group['params'])} parameters {num_params} "
                  f"size {size} lr {param_group['lr']}")
    
    def get_current_lr(self):
        """Get current learning rate from scheduler."""
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()
        return [self.learning_rate]
    
    def step_scheduler(self):
        """Step the learning rate scheduler."""
        if self.scheduler is not None:
            self.scheduler.step()
    
    @property 
    def scheduler(self):
        """Get the current scheduler."""
        return self._scheduler
    
    @scheduler.setter
    def scheduler(self, value):
        """Set the scheduler."""
        self._scheduler = value