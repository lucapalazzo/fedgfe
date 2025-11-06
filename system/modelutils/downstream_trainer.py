import torch
from tqdm import tqdm


class DownstreamTrainer:
    """Handles downstream task training for federated learning clients."""
    
    def __init__(self, client_id, args):
        self.client_id = client_id
        self.args = args
    
    def train_downstream(self, model, downstream_task, epochs, dataloader, optimizer, 
                        pretext_tasks=None, device=None, training_task="both",
                        model_freeze_callback=None, get_label_callback=None,
                        check_batch_callback=None):
        """
        Train downstream task.
        
        Args:
            model: The model with backbone
            downstream_task: Downstream task to train
            epochs: Number of training epochs
            dataloader: Training dataloader
            optimizer: The optimizer to use
            pretext_tasks: List of pretext tasks (to determine backbone freezing)
            device: Device to train on
            training_task: Type of training ("both", "downstream")
            model_freeze_callback: Function to freeze/unfreeze model parts
            get_label_callback: Function to extract labels from data
            check_batch_callback: Function to validate batch data
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        if downstream_task is None:
            return
            
        if not (training_task == "both" or training_task == "downstream"):
            return
        
        print(f"Node {self.client_id} training downstream task "
              f"{type(downstream_task).__name__} for {epochs} epochs")
        
        num_batches = len(dataloader.dataset) // dataloader.batch_size
        
        # Determine backbone freezing strategy
        if pretext_tasks is None or len(pretext_tasks) == 0:
            print(f"No pretext task, not freezing backbone parameters")
            freeze_backbone = False
        else:
            freeze_backbone = True
        
        # Freeze appropriate model parts
        if model_freeze_callback:
            model_freeze_callback(backbone=freeze_backbone, pretext=True, downstream=False)
        
        # Ensure model is in correct mode
        model.pretext_train = False
        
        # Training loop
        for step in range(epochs):
            pbarbatch = tqdm(total=num_batches, desc=f"Batch ", unit='batch', leave=False)
            
            for i, (x, y) in enumerate(dataloader):
                # Validate batch
                if check_batch_callback and not check_batch_callback(x, y):
                    continue
                
                # Prepare batch data
                x, y = self._prepare_batch_data(x, y, device, get_label_callback)
                
                # Forward pass through downstream task
                output = downstream_task(x)
                if output is None or isinstance(output, torch._C._TensorMeta):
                    print(f"Node {self.client_id} downstream output is invalid")
                    continue
                
                # Compute loss
                optimizer.zero_grad()
                loss = downstream_task.loss(output, y, samples=x).to(device)
                
                # Backward pass
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(downstream_task.parameters(), max_norm=1.0)

                optimizer.step()
                
                # Update progress bar
                pbarbatch.set_postfix({
                    'Loss': f'{loss.item():.4f}', 
                    'Epoch': f'{step+1}/{epochs}'
                })
                pbarbatch.update(1)
                
                # Check sample limit
                if (self.args.limit_samples_number > 0 and 
                    i * dataloader.batch_size > self.args.limit_samples_number):
                    break
            
            pbarbatch.close()
            
            # Periodic GPU memory cleanup
            if torch.cuda.is_available() and (step + 1) % 10 == 0:
                torch.cuda.empty_cache()
    
    def _prepare_batch_data(self, x, y, device, get_label_callback=None):
        """Prepare batch data by moving to device and extracting labels."""
        # Move input to device
        if isinstance(x, list):
            x[0] = x[0].to(device)
        else:
            x = x.to(device)
        
        # Extract labels if callback provided
        if get_label_callback:
            y = get_label_callback(y)
        
        # Move labels to device
        y = y.to(device)
        
        return x, y
    
    def evaluate_downstream(self, model, downstream_task, dataloader, device=None, 
                          get_label_callback=None, check_batch_callback=None):
        """
        Evaluate downstream task performance.
        
        Args:
            model: The model with backbone
            downstream_task: Downstream task to evaluate
            dataloader: Evaluation dataloader
            device: Device to run on
            get_label_callback: Function to extract labels from data
            check_batch_callback: Function to validate batch data
            
        Returns:
            Dictionary with evaluation metrics
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if downstream_task is None:
            return {}
        
        model.eval()
        model.pretext_train = False
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                # Validate batch
                if check_batch_callback and not check_batch_callback(x, y):
                    continue
                
                # Prepare batch data
                x, y = self._prepare_batch_data(x, y, device, get_label_callback)
                
                # Forward pass
                output = downstream_task(x)
                if output is None:
                    continue
                
                # Compute loss
                loss = downstream_task.loss(output, y, samples=x)
                total_loss += loss.item()
                total_samples += y.shape[0]
        
        model.train()
        
        return {
            'loss': total_loss / max(total_samples, 1),
            'samples': total_samples
        }
    
    def get_downstream_predictions(self, model, downstream_task, dataloader, device=None,
                                 get_label_callback=None, check_batch_callback=None):
        """
        Get downstream task predictions.
        
        Args:
            model: The model with backbone
            downstream_task: Downstream task
            dataloader: Dataloader for prediction
            device: Device to run on
            get_label_callback: Function to extract labels from data
            check_batch_callback: Function to validate batch data
            
        Returns:
            Tuple of (predictions, labels)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if downstream_task is None:
            return [], []
        
        model.eval()
        model.pretext_train = False
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in dataloader:
                # Validate batch
                if check_batch_callback and not check_batch_callback(x, y):
                    continue
                
                # Prepare batch data
                x, y = self._prepare_batch_data(x, y, device, get_label_callback)
                
                # Forward pass
                output = downstream_task(x)
                if output is None:
                    continue
                
                # Store predictions and labels
                all_predictions.append(output.detach().cpu())
                all_labels.append(y.detach().cpu())
        
        model.train()
        
        # Concatenate all batches
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
        
        return all_predictions, all_labels