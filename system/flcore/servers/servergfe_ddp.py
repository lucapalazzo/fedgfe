import os
import time
import torch
import torch.distributed as dist
from flcore.servers.servergfe import FedGFE
from flcore.clients.clientgfe_ddp import clientGFEDDP


class FedGFEDDP(FedGFE):
    """
    FedGFE server with DistributedDataParallel support.
    
    This class extends FedGFE to work with DDP-enabled clients and handle
    distributed training coordination in federated learning scenarios.
    """
    
    def __init__(self, args, times, pretext_tasks=None, **kwargs):
        
        # DDP-specific server attributes
        self.ddp_enabled = getattr(args, 'ddp_enabled', False)
        self.ddp_backend = getattr(args, 'ddp_backend', 'nccl')
        self.ddp_world_size = getattr(args, 'ddp_world_size', 1)
        self.ddp_visible_gpus = getattr(args, 'ddp_visible_gpus', None)
        
        super().__init__(args, times, pretext_tasks, **kwargs)
        
        print(f"FedGFE Server initialized with DDP support: {self.ddp_enabled}")
        
        if self.ddp_enabled:
            print(f"DDP Configuration - World Size: {self.ddp_world_size}, Backend: {self.ddp_backend}")
            if self.ddp_visible_gpus:
                print(f"DDP Visible GPUs: {self.ddp_visible_gpus}")
            else:
                print("DDP using all available GPUs")
    
    def set_clients(self, clientObj):
        """
        Override to create DDP-enabled clients if DDP is enabled.
        """
        if self.ddp_enabled:
            # Use DDP client class
            super().set_clients(clientGFEDDP)
            print(f"Created {len(self.clients)} DDP-enabled clients")
        else:
            # Use regular client class
            super().set_clients(clientObj)
            print(f"Created {len(self.clients)} regular clients")
        
        # Log DDP status for each client
        for client in self.clients:
            if hasattr(client, 'get_distributed_info'):
                ddp_info = client.get_distributed_info()
                if ddp_info['is_distributed']:
                    print(f"Client {client.id}: DDP enabled - Rank {ddp_info['rank']}/{ddp_info['world_size']}")
                else:
                    print(f"Client {client.id}: Running in single-process mode")
    
    def aggregate_parameters(self):
        """
        Override aggregation to handle DDP wrapped models.
        """
        if not self.ddp_enabled:
            return super().aggregate_parameters()
        
        # Enhanced aggregation for DDP models
        assert len(self.uploaded_models) > 0
        
        # Extract parameters from DDP wrapped models
        model_params = []
        for client_model in self.uploaded_models:
            params = []
            
            # Handle DDP wrapped models
            if hasattr(client_model, 'backbone'):
                if hasattr(client_model.backbone, 'module'):
                    # DDP wrapped backbone
                    backbone_params = list(client_model.backbone.module.parameters())
                else:
                    backbone_params = list(client_model.backbone.parameters())
                params.extend(backbone_params)
            
            # Handle downstream task parameters if needed
            if hasattr(client_model, 'downstream_task') and client_model.downstream_task is not None:
                if hasattr(client_model.downstream_task, 'module'):
                    # DDP wrapped downstream task
                    downstream_params = list(client_model.downstream_task.module.parameters())
                else:
                    downstream_params = list(client_model.downstream_task.parameters())
                params.extend(downstream_params)
            
            model_params.append(params)
        
        # Perform federated averaging
        if len(model_params) > 0 and len(model_params[0]) > 0:
            # Average parameters across clients
            global_params = []
            for param_idx in range(len(model_params[0])):
                param_sum = None
                for client_idx, client_params in enumerate(model_params):
                    if param_sum is None:
                        param_sum = client_params[param_idx].data.clone()
                    else:
                        param_sum += client_params[param_idx].data
                
                global_params.append(param_sum / len(model_params))
            
            # Update global model with averaged parameters
            self.update_global_model_parameters(global_params)
    
    def update_global_model_parameters(self, global_params):
        """
        Update global model with aggregated parameters.
        """
        param_idx = 0
        
        # Update backbone parameters
        if hasattr(self.global_model, 'backbone'):
            for param in self.global_model.backbone.parameters():
                param.data = global_params[param_idx].clone()
                param_idx += 1
        
        # Update downstream task parameters if they exist
        if hasattr(self.global_model, 'downstream_task') and self.global_model.downstream_task is not None:
            for param in self.global_model.downstream_task.parameters():
                param.data = global_params[param_idx].clone()
                param_idx += 1
    
    def send_models_to_clients(self):
        """
        Override to handle DDP model parameter distribution.
        """
        if not self.ddp_enabled:
            return super().send_models_to_clients()
        
        # Enhanced model distribution for DDP clients
        for client in self.selected_clients:
            # Move global model to appropriate device
            client._move_to_gpu(client.device if hasattr(client, 'device') else self.device)
            
            # Set parameters (this handles DDP wrapped models in clientGFEDDP)
            client.set_parameters(self.global_model)
            
            # Move back to CPU for memory efficiency (if not distributed)
            if not client.is_distributed:
                client._move_to_cpu()
            
            # Synchronize parameters across DDP processes for this client
            if hasattr(client, 'sync_model_parameters'):
                client.sync_model_parameters()
    
    def train_clients(self, round, training_task="both"):
        """
        Override train_clients to handle DDP epoch setting.
        """
        # Set epoch for all DDP clients to ensure proper data shuffling
        for client in self.clients:
            if hasattr(client, 'set_epoch') and hasattr(client, 'is_distributed') and client.is_distributed:
                client.set_epoch(round)
        
        # Call parent train_clients
        super().train_clients(round, training_task)

    def client_round_ending_hook(self, client):
        """
        Override to handle DDP-specific cleanup and synchronization.
        """
        # Ensure DDP synchronization before aggregation
        if hasattr(client, 'sync_model_parameters') and client.is_distributed:
            client.sync_model_parameters()
        
        # Call parent hook
        super().client_round_ending_hook(client)
        
        # Additional DDP cleanup if needed
        if hasattr(client, 'is_distributed') and client.is_distributed:
            # Force GPU memory cleanup for DDP processes
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    
    def train(self):
        """
        Override train method to replicate FedGFE behavior with DDP enhancements.
        """
        training_task = "both" 
        if self.nodes_training_sequence == "sslfirst":
            training_task = "pretext"

        self.global_model.to(self.device)
        if self.model_aggregation == "fedavg":
            self.send_models()

        for i in range(1, self.global_rounds + 1):
            self.round = i
            if self.round == int(self.global_rounds * 0.8):
                for client in self.clients:
                    client.optimizer = client.finetuning_optimizer 

            if self.round > self.ssl_rounds and self.nodes_training_sequence == "sslfirst":
                training_task = "downstream"

            if training_task == "both":
                if self.pretext_tasks != None and len(self.pretext_tasks) > 0:
                    self.ssl_round += 1
                self.data_log({"ssl_round": self.ssl_round})
                self.downstream_round += 1
                self.data_log({"downstream_round": self.downstream_round})
            elif training_task == "downstream":
                self.downstream_round += 1
                self.data_log({"downstream_round": self.downstream_round})
            elif training_task == "pretext":
                if self.pretext_tasks != None and len(self.pretext_tasks) > 0:
                    self.ssl_round += 1
                self.data_log({"ssl_round": self.ssl_round})

            if self.no_wandb == False:
                self.data_log({"round": self.round})

            s_t = time.time()
            self.selected_clients = self.clients

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            if self.model_aggregation == "fedavg":
                self.send_models() 
            
            self.train_clients(i, training_task=training_task)
         
            print(self.uploaded_ids)

            self.Budget.append(time.time() - s_t)
            print('-'*50 + "Round time: ", self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            if self.model_aggregation == "fedavg":
                self.receive_models()
                self.aggregate_parameters()

            if self.model_backbone_save_checkpoint:
                self.save_checkpoint()

        self.save_results()
        self.evaluate()
        
        # DDP cleanup before finishing
        if self.ddp_enabled:
            self.cleanup_ddp()
            
        import wandb
        wandb.finish()

    def cleanup_ddp(self):
        """
        Clean up DDP resources for all clients.
        """
        if not self.ddp_enabled:
            return
        
        print("Cleaning up DDP resources...")
        for client in self.clients:
            if hasattr(client, 'cleanup_distributed'):
                client.cleanup_distributed()
        
        print("DDP cleanup completed")
    
    def __del__(self):
        """Ensure cleanup on server destruction."""
        if hasattr(self, 'ddp_enabled') and self.ddp_enabled:
            self.cleanup_ddp()