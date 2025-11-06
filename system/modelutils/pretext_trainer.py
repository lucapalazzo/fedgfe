import os
import torch
from tqdm import tqdm
from utils.node_metric import NodeMetric


class PretextTrainer:
    """Handles pretext task training for federated learning clients."""
    
    def __init__(self, client_id, args, data_log_callback=None):
        self.client_id = client_id
        self.args = args
        self.data_log_callback = data_log_callback
        
    def train_pretext(self, model, pretext_tasks, epochs, dataloader, optimizer, optimizer_manager, 
                     downstream_task=None, downstream_loss_operation="none", device=None, 
                     ssl_round=0, model_freeze_callback=None, get_label_callback=None,
                     copy_model_params_callback=None, count_updated_params_callback=None):
        """
        Train model on pretext tasks.
        
        Args:
            model: The model to train
            pretext_tasks: List of pretext task names
            epochs: Number of training epochs
            dataloader: Training dataloader
            optimizer: The optimizer to use
            optimizer_manager: Optimizer manager instance
            downstream_task: Optional downstream task for joint training
            downstream_loss_operation: How to combine downstream loss ("none", "sum")
            device: Device to train on
            ssl_round: Current SSL round number
            model_freeze_callback: Function to freeze/unfreeze model parts
            get_label_callback: Function to extract labels from data
            copy_model_params_callback: Function to copy model parameters
            count_updated_params_callback: Function to count updated parameters
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        num_batches = len(dataloader.dataset) // dataloader.batch_size

        pretext_task_name = model.pretext_task_name 
        print (f"*** Node {self.client_id} {pretext_task_name} memory before training {torch.cuda.memory_allocated(device)//1024**2} MB")
        
        print(f"\nNode {self.client_id} training on pretext task {pretext_task_name} "
              f"current LR {optimizer_manager.get_current_lr()}")
        
        # Freeze appropriate model parts
        if model_freeze_callback:
            model_freeze_callback(backbone=False, pretext=False, downstream=True)
        
        round_loss = 0
        round_downstream_loss = 0
        model = model.to(device)
        
        for step in range(epochs):
            if dataloader is None:
                print(f"Node {self.client_id} has no data")
                return
                
            losses = 0
            downstream_losses = 0
            
            pbarbatch = tqdm(total=num_batches, desc=f"Batch ", unit='batch', leave=False)
            
            for i, (x, y) in enumerate(dataloader):
                if not self._check_batch(x, y):
                    continue
                
                # Move data to device
                x, y = self._prepare_batch_data(x, y, device, get_label_callback)
                
                # Store parameters for tracking changes
                backbone_pre_params = None
                if copy_model_params_callback:
                    backbone_pre_params = copy_model_params_callback(model.backbone)
                
                # Forward pass
                output = model(x)
                loss = model.loss(output, y)
                summed_loss = loss
                
                # Handle downstream task if present
                downstream_loss = self._handle_downstream_task(
                    model, downstream_task, x, y, downstream_loss_operation, device
                )
                
                if downstream_loss_operation == "sum":
                    summed_loss = loss + downstream_loss
                
                # Backward pass
                optimizer.zero_grad()
                summed_loss.backward()
                losses += loss.item()
                optimizer.step()
                
                # Track parameter updates
                updated_info = ""
                if backbone_pre_params is not None and count_updated_params_callback:
                    updated_backbone, total_backbone = count_updated_params_callback(
                        backbone_pre_params, model=model.backbone
                    )
                    updated_info = f"Backbone: {updated_backbone}/{total_backbone} ({updated_backbone/total_backbone:.2f})"
                    for p in backbone_pre_params:
                        p.to('cpu')
                        del p
                
                # Debug output images
                if os.path.exists("save_debug_images"):
                    model.pretext_task.debug_output_images(x, output, node_id=self.client_id, postfix="train")
                
                # Update progress bar
                if downstream_task is not None:
                    pbarbatch.set_postfix({
                        'Loss': f'{loss.item():.2f}',
                        'DSLoss': f'{downstream_loss.item():.2f}',
                        'Epoch': f'{step+1}/{epochs}',
                        'Backbone': updated_info
                    })
                else:
                    pbarbatch.set_postfix({
                        'Loss': f'{loss.item():.2f}',
                        'Epoch': f'{step+1}/{epochs}',
                        'Backbone': updated_info
                    })
                
                pbarbatch.update(1)
                
                # Check sample limit
                if (self.args.limit_samples_number > 0 and 
                    i * dataloader.batch_size > self.args.limit_samples_number):
                    break
            
            round_loss += losses
            round_downstream_loss += downstream_losses
            optimizer_manager.step_scheduler()
            pbarbatch.close()
            
            # Periodic GPU memory cleanup
            if torch.cuda.is_available() and (step + 1) % 5 == 0:
                torch.cuda.empty_cache()
        
        # Log training metrics
        self._log_training_metrics(pretext_task_name, round_loss, round_downstream_loss, epochs, ssl_round)
        
        # Test pretext task
        print()
        
        test_metrics = self._test_pretext_task(
            model, dataloader, get_label_callback, device
        )
        # print(f"Node {self.client_id} pretext task {pretext_task_name} metrics on train {test_metrics}")
        # print (f"*** Node {self.client_id} {pretext_task_name} memory after training {torch.cuda.memory_allocated(device)//1024**2} MB")

    
    def _check_batch(self, x, y):
        """Check if batch is valid."""
        if x is None or y is None:
            return False
        if isinstance(x, list) and len(x) == 0:
            return False
        if isinstance(x, torch.Tensor) and x.numel() == 0:
            return False
        return True
    
    def _prepare_batch_data(self, x, y, device, get_label_callback=None):
        """Prepare batch data by moving to device and extracting labels."""
        if isinstance(x, list):
            x[0] = x[0].to(device)
        else:
            x = x.to(device)
        
        if get_label_callback:
            y = get_label_callback(y)
        
        return x, y
    
    def _handle_downstream_task(self, model, downstream_task, x, y, downstream_loss_operation, device):
        """Handle downstream task computation during pretext training."""
        downstream_loss = torch.tensor(0.0, device=device)
        
        if downstream_task is not None:
            downstream_task.backbone_enabled = True
            downstream_output = downstream_task(x)
            
            if downstream_loss_operation == "none":
                with torch.no_grad():
                    if downstream_output is None:
                        print(f"Node {self.client_id} downstream task output is None")
                        return downstream_loss
                    downstream_loss = downstream_task.loss(downstream_output, y)
            elif downstream_loss_operation == "sum":
                downstream_loss = downstream_task.loss(downstream_output, y)
                downstream_task.backbone_enabled = False

            del downstream_output
        
        return downstream_loss
    
    def _log_training_metrics(self, pretext_task_name, round_loss, round_downstream_loss, epochs, ssl_round):
        """Log training metrics."""
        if self.data_log_callback:
            self.data_log_callback({
                f"train/node_{self.client_id}/pretext_train_loss_{pretext_task_name}": round_loss/epochs,
                "ssl_round": ssl_round
            })
            self.data_log_callback({
                f"train/node_{self.client_id}/pretext_train_ds_loss_{pretext_task_name}": round_downstream_loss/epochs,
                "ssl_round": ssl_round
            })
    
    def _test_pretext_task(self, model, dataloader, get_label_callback, device):
        """Test pretext task performance."""
        batch_metrics = NodeMetric(phase=NodeMetric.Phase.TRAIN)
        batch_metrics.define_metrics(model.defined_test_metrics)
        
        with torch.no_grad():
            for x, y in dataloader:
                if isinstance(x, list):
                    x[0] = x[0].to(device)
                else:
                    x = x.to(device)
                
                if get_label_callback:
                    y = get_label_callback(y)
                y = y.to(device)
                
                output = model(x)
                batch_metrics.steps = 1
                model.test_metrics(output, y, samples=x, metrics=batch_metrics)
                
                if os.path.exists("save_debug_images"):
                    model.pretext_task.debug_output_images(x, output, node_id=self.client_id, postfix="train_test")
        
        return batch_metrics
    
    def test_pretext(self, model, testloader, get_label_callback, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        metrics = NodeMetric(phase=NodeMetric.Phase.TEST)
        metrics.define_metrics(model.defined_test_metrics)
        metrics.task_name = model.pretext_task_name
        
        batch_metrics = NodeMetric(phase=NodeMetric.Phase.TEST)
        batch_metrics.define_metrics(model.defined_test_metrics)
        
        with torch.no_grad():
            for x, y in testloader:
                if isinstance(x, list):
                    x[0] = x[0].to(device)
                else:
                    x = x.to(device)
                
                if get_label_callback:
                    y = get_label_callback(y)
                y = y.to(device)
                
                output = model(x)
                batch_metrics.steps = 1
                model.test_metrics(output, y, samples=x, metrics=batch_metrics)
                metrics += batch_metrics
                
                if os.path.exists("save_debug_images"):
                    model.pretext_task.debug_output_images(x, output, node_id=self.client_id, postfix="test")
        
        return metrics