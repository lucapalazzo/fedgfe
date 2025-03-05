import os
from flcore.trainmodel.downstream import Downstream
import torch.nn as nn
import torch
from flcore.trainmodel.unet import UNet
from timm.models.vision_transformer import VisionTransformer
from transformers import pipeline


from transformers import ViTModel

from flcore.trainmodel.TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from flcore.trainmodel.TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from torch.nn.modules.loss import CrossEntropyLoss
from flcore.trainmodel.TransUNet.utils import DiceLoss



from utils.image_utils import save_image_from_tensor,save_grid_images

from monai.losses import DiceLoss as DiceLossMonai
import numpy as np
import torch.nn.functional as F
import wandb

class DownstreamSegmentation (Downstream):
    def __init__(self, backbone, num_classes=10, cls_token_only=False, img_size=224, patch_count=-1, patch_size=-1, wandb_log=False):
        super(DownstreamSegmentation, self).__init__(backbone, cls_token_only=cls_token_only, img_size=img_size, patch_count=patch_count, patch_size=patch_size, wandb_log=wandb_log)

        input_dim = 0

        self.num_masks = num_classes
        self.mask_num_layers = num_classes
        if num_classes == 1:
            self.mask_out_layers = 2
        else:
            self.mask_out_layers = num_classes

        self.mask_threshold = None

        self.test_metrics = { "precision": [], "recall": [], "accuracy": [], "dice": [] }
        self.test_metrics_aggregated = { "precision": [], "recall": [], "accuracy": [], "dice": [] }
        self.test_running_round = -1
        self.epoch_running_batch = -1

        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        # config_vit = CONFIGS_ViT_seg['ViT-B_16']
        config_vit.n_classes = self.mask_out_layers
        config_vit.n_skip = 3
        config_vit.pretrained_path = None
        config_vit.patches.grid = [ int(self.patch_count**0.5), int(self.patch_count**0.5) ]
        # config_vit.resnet.width_factor = 2
        # config_vit.patches.grid = None
        config_vit.patches.size = [ self.patch_size, self.patch_size ]
        config_vit.patch_size = backbone.config.patch_size
        config_vit.hidden_size = backbone.config.hidden_size
        config_vit.transformer.num_heads = backbone.config.num_attention_heads
        config_vit.transformer.num_layers = backbone.config.num_hidden_layers

        config_vit['n_classes'] = self.mask_out_layers
        # config_vit['n_skip'] = 2
        # config_vit['decoder_channels'] = [256, 64, 16]

        self.vitseg = ViT_seg(config_vit, img_size=224, num_classes=config_vit.n_classes)
        if isinstance(backbone, VisionTransformer):
            input_dim = backbone.embed_dim
        elif isinstance(backbone, ViTModel):
            self.vitseg.transformer.encoder = backbone

        self.defined_test_metrics = [ "segmentation_accuracy", "segmentation_precision",
                                     "segmentation_recall", "segmentation_dice" ] 
        self.defined_train_metrics = [ "segmentation_loss" ] 

        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(self.mask_out_layers)
        self.dice_loss_monai = DiceLossMonai(sigmoid=True, squared_pred=True, to_onehot_y=True)
        self.dice_loss_monai_nobg = DiceLossMonai(sigmoid=True, squared_pred=True, to_onehot_y=True, include_background=False)
        # self.dice_loss_monai = DiceLossMonai(reduction='none')


    def parameters_count(self):
        encoder_params = sum(p.numel() for p in self.vitseg.transformer.parameters())
        decoder_params = sum(p.numel() for p in self.vitseg.decoder.parameters())
        head_params = sum(p.numel() for p in self.vitseg.segmentation_head.parameters())
        print ( f"Encoder Params: {encoder_params}, Decoder Params: {decoder_params}, Head Params: {head_params}")
    
    def define_metrics(self, metrics_path = None):
        if self.wandb_log == False:
            return None
        
        super().define_metrics( metrics_path=metrics_path)
        
        path = "/" if metrics_path is None else "/"+metrics_path+"/"
        metrics = []

        for metric in self.defined_test_metrics:
            metrics.append(f"test{path}{metric}")
            
        for metric in self.defined_train_metrics:
            metrics.append(f"train{path}{metric}")

        # metrics.append(f"train{path}segmentation_loss" )
        # metrics.append(f"test{path}segmentation_accuracy")
        # metrics.append(f"test{path}segmentation_precision")
        # metrics.append(f"test{path}segmentation_recall")
        # metrics.append(f"test{path}segmentation_dice")
        
        for metric in metrics:
           a = wandb.define_metric(metric, step_metric="round")

    def train_metrics ( self, dataloader ):
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in dataloader:
                if type(y) == dict:
                    if 'semantic_masks' in y:
                        y = y['semantic_masks']
                    else:
                        y = y['masks']

                y = y.to(x.device)
                output = self(x)

                if output == None:
                    continue

                loss = self.downstream_loss(output, y)
                train_num += 1
                losses += loss.item()

            losses /= train_num

            # print ( f"Downstream task loss {losses/train_num}")
            self.train_metrics_log( loss=losses, round = self.round)

    def test_metrics_log(self, round = None, prefix="test"):
        if round is None:
            round = self.round

        if self.wandb_log == True:
            wandb.log({f"test/{self.metrics_path}/segmentation_accuracy": self.test_metrics_aggregated['accuracy'], "round": round})
            wandb.log({f"test/{self.metrics_path}/segmentation_precision": self.test_metrics_aggregated['precision'], "round": round})
            wandb.log({f"test/{self.metrics_path}/segmentation_recall": self.test_metrics_aggregated['recall'], "round": round})
            wandb.log({f"test/{self.metrics_path}/segmentation_dice": self.test_metrics_aggregated['dice'], "round": round})
    
    def train_metrics_log(self, loss, round = 0, prefix="train"):
        if self.wandb_log == True:
            wandb.log({f"train/{self.metrics_path}/segmentation_loss": loss, "round": round})

    def test_metrics_calculate(self, logits, labels, round = None):

        if round is None:
            round = self.round

        groundtruth_mask = labels if labels.shape[1] == 1 else labels.permute(0, 3, 1, 2)
        pred_mask = logits
        b, c, h, w = pred_mask.shape

        if groundtruth_mask.shape[2] != h or groundtruth_mask.shape[3] != w:
            groundtruth_mask = F.interpolate(groundtruth_mask, size=(h, w), mode='nearest')

        if self.test_running_round != self.round:
            self.test_metrics['precision'] = []
            self.test_metrics['recall'] = []
            self.test_metrics['accuracy'] = []
            self.test_metrics['dice'] = []
            self.test_running_round = self.round

        for i in range(b):
            pred_mask_binary = ( pred_mask[i][1]> 0.5).int()
            groundtruth_mask_binary = groundtruth_mask[i][0].int()

            self.test_metrics['precision'].append(self.test_metrics_precision(groundtruth_mask_binary, pred_mask_binary))
            self.test_metrics['recall'].append(self.test_metrics_recall(groundtruth_mask_binary, pred_mask_binary))
            self.test_metrics['accuracy'].append(self.test_metrics_accuracy(groundtruth_mask_binary, pred_mask_binary))
            self.test_metrics['dice'].append(self.test_metrics_dice(groundtruth_mask_binary, pred_mask_binary))
        
        self.test_metrics_aggregated['precision'] = torch.mean(torch.stack(self.test_metrics['precision']))
        self.test_metrics_aggregated['recall'] = torch.mean(torch.stack(self.test_metrics['recall']))
        self.test_metrics_aggregated['accuracy'] = torch.mean(torch.stack(self.test_metrics['accuracy']))
        self.test_metrics_aggregated['dice'] = torch.mean(torch.stack(self.test_metrics['dice']))
        
        print ( f"Precision: {self.test_metrics_aggregated['precision']}, Recall: {self.test_metrics_aggregated['recall']}, Accuracy: {self.test_metrics_aggregated['accuracy']} Dice: {self.test_metrics_aggregated['dice']}")

        return self.test_metrics_aggregated['precision'], self.test_metrics_aggregated['recall'], self.test_metrics_aggregated['accuracy'], self.test_metrics_aggregated['dice']
    
    def test_metrics_recall(self, groundtruth_mask, pred_mask):
        intersect = torch.sum(pred_mask*groundtruth_mask)
        total_pixel_truth = torch.sum(groundtruth_mask)
        recall = torch.mean(intersect/total_pixel_truth)
        decimals = 2
        return torch.round(recall * 10**decimals)/10**decimals

    def test_metrics_precision(self, groundtruth_mask, pred_mask):
        intersect = torch.sum(pred_mask*groundtruth_mask)
        total_pixel_pred = torch.sum(pred_mask)
        precision = torch.mean(intersect/total_pixel_pred)
        decimals = 2
        return torch.round(precision * 10**decimals)/10**decimals

    def test_metrics_accuracy(self,groundtruth_mask, pred_mask):
        intersect = torch.sum(pred_mask*groundtruth_mask)
        union = torch.sum(pred_mask) + torch.sum(groundtruth_mask) - intersect
        xor = torch.sum(groundtruth_mask==pred_mask)
        acc = torch.mean(xor/(union + xor - intersect))
        decimals = 2
        return torch.round(acc * 10**decimals)/10**decimals
    
    def test_metrics_dice(self, groundtruth_mask, pred_mask):
        intersect = torch.sum(pred_mask*groundtruth_mask)
        total_sum = torch.sum(pred_mask) +  torch.sum(groundtruth_mask)
        dice = torch.mean(2*intersect/total_sum)
        decimals = 2
        return torch.round(dice * 10**decimals)/10**decimals
    
    def downstream_loss ( self, logits, labels, samples = None):
        # return torch.Tensor([1])

        if torch.isnan(logits).any():
            print ( "Logits contain NaN")
            return torch.Tensor([1])
        target = labels
        if type(labels) == dict:
            target = labels['masks']

        if target.shape[3] == 1:
            target = target.permute(0,3,1,2)

        if target.shape[2] != self.img_size or target.shape[3] != self.img_size:
            target = nn.functional.interpolate(target, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        images = []
        images_count = target.shape[1] + self.mask_out_layers
        if self.segmentation_mask_threshold != None:
            images_count += 1
        
        if samples != None:
            images_count += 1
        
        for i in range(target.shape[0]):
            predicted_masks = list(logits[i].split(1,dim=0))
            foreground = predicted_masks[1]
            target_images = [target[i][j].unsqueeze(0) for j in range(self.num_masks)]
            # for m in range(self.mask_num_layers):
            #     images.append(predicted_masks[m])
            #     images.append(target_images[m])
            images = predicted_masks + target_images
            if self.segmentation_mask_threshold != None:
                binary_tensor = (foreground >= self.segmentation_mask_threshold).float()
                images.append(binary_tensor)
                if samples != None:
                    
                    if samples.shape[1] == 3:
                        sample = samples[i][2::]
                    else:
                        sample = samples[i]
                    masked_image = sample * binary_tensor
                    images.append(masked_image)
        
        softmax = False
        # if self.mask_num_layers > 1:
        #     softmax = True


        binary_tensor = None
        if os.path.exists("save_debug_images"):
            if torch.is_grad_enabled():
                filename = "masked.png"
            else:
                filename = "masked_pretext.png"
            save_grid_images( images, nrow=images_count, output_path=(filename))


        if target.shape[1] == 1:
            target = target.squeeze(1)

        if self.mask_num_layers == 1:
            loss_ce = self.ce_loss(logits, target.long())
        else:
            loss_ce = self.ce_loss(logits, target)
        
        loss_dice = self.dice_loss(logits, target, softmax=softmax)
        # loss_dice_monai = self.dice_loss_monai(logits, target.unsqueeze(dim=1))
        # loss_dice_monai_nobg = self.dice_loss_monai_nobg(logits, target.unsqueeze(dim=1))
        # print ( f"CE Loss: {loss_ce}, Dice Loss: {loss_dice}, Dice Loss Monai: {loss_dice_monai} Dice Loss Monai No BG: {loss_dice_monai_nobg}")
        return loss_ce + loss_dice
    
    def parameters(self, recurse=True):
        moduleList = nn.ModuleList()
        moduleList.add_module("decoder", self.vitseg.decoder)
        moduleList.add_module("head", self.vitseg.segmentation_head)
        return moduleList.parameters(recurse)

    def forward(self, x):

        # if self.backbone is not None:
        #     backbone_x = self.backbone(x)
        if self.patch_size != 16:
            print ( "Segmentation only works with patch size 16")
            return None

        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x, attn_weights, features = self.vitseg.transformer(x)  # (B, n_patch, hidden)
        
        if x.shape[1] == self.patch_count + 1:
            # we need to remove the cls token
            x = x[:, 1:, :]
        x = self.vitseg.decoder(x, features)
        if torch.isnan(x).any():
            print ( "X contains NaN")
            return torch.Tensor
        logits = self.vitseg.segmentation_head(x)
        return logits