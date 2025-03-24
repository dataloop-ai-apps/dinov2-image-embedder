import dtlpy as dl
import torchvision.transforms as transforms
import traceback
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
from typing import List
import os
import glob
import numpy as np
import math

logger = logging.getLogger("[DINOV2-ADAPTER]")
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SimpleDataset(Dataset):
    """
    A simple dataset loader for self-supervised learning.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, '**', '*.jpeg'), recursive=True) +
                                  glob.glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True) +
                                  glob.glob(os.path.join(data_dir, '**', '*.png'), recursive=True))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    Maps representations to the space where contrastive loss is applied.
    """
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x


class DINOv2Adapter(dl.BaseModelAdapter):
    def __init__(self, model_entity=None):
        super().__init__(model_entity)
        self.device = None
        self.backbone = None
        self.projection_head = None
        self.optimizer = None
        
    def load(self, local_path, **kwargs):
        """
        Loads model checkpoint from a local directory.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load a pre-trained DINOv2 backbone
        model_name = self.configuration.get("model_name", "dinov2_vits14")
        logger.info(f"Loading DINOv2 backbone: {model_name}")
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.backbone.to(self.device)
        
        # Create projection head for contrastive learning
        input_dim = 384 if "vits" in model_name else 768 if "vitb" in model_name else 1024
        self.projection_head = ProjectionHead(input_dim=input_dim)
        self.projection_head.to(self.device)
        
        # Load checkpoint if exists
        checkpoint_path = os.path.join(local_path, 'dino_backbone_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.backbone.load_state_dict(checkpoint['backbone_state_dict'], strict=False)
            self.projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
            logger.info("Checkpoint loaded successfully")
        else:
            logger.info("No checkpoint found; using pre-trained backbone with new projection head")
    
    def save(self, local_path, **kwargs):
        os.makedirs(local_path, exist_ok=True)
        checkpoint = {
            'backbone_state_dict': self.backbone.state_dict(),
            'projection_head_state_dict': self.projection_head.state_dict()
        }
        checkpoint_path = os.path.join(local_path, 'dino_backbone_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def augment(self, images):
        """
        Apply random augmentations to create different views of the same images.
        """
        # Define augmentations for contrastive learning
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return augmentation(images)
    
    def nt_xent_loss(self, similarity_matrix, temperature=0.5):
        """
        Compute NT-Xent loss (normalized temperature-scaled cross entropy).
        This is the contrastive loss function used in SimCLR.
        """
        batch_size = similarity_matrix.size(0)
        
        # Create labels for positive pairs (diagonal elements)
        labels = torch.arange(batch_size).to(self.device)
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Divide by temperature
        logits = logits / temperature
        
        # Compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive samples
        mask = torch.zeros_like(log_prob)
        mask.scatter_(1, labels.unsqueeze(1), 1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        
        return loss
    
    def _process_epoch_dataloader(self, dataloader, optimizer, scaler, mode="train"):
        """
        Process one epoch using contrastive learning.
        """
        epoch_loss = 0.0
        num_batches = 0
        
        if mode == "train":
            self.backbone.train()
            self.projection_head.train()
        else:
            self.backbone.eval()
            self.projection_head.eval()
        
        for batch in dataloader:
            images = batch[0].to(self.device)
            batch_size = images.shape[0]
            
            # Skip small batches
            if batch_size < 2:
                continue
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # Create two randomly augmented views of the same images
                view1 = self.augment(images)
                view2 = self.augment(images)
                
                # Get embeddings from the backbone
                emb1 = self.backbone(view1)
                emb2 = self.backbone(view2)
                
                # Project embeddings to contrastive space
                z1 = self.projection_head(emb1)
                z2 = self.projection_head(emb2)
                
                # Normalize projections
                z1 = F.normalize(z1, dim=1)
                z2 = F.normalize(z2, dim=1)
                
                # Compute similarity matrix
                similarity_matrix = torch.matmul(z1, z2.T)
                
                # Compute contrastive loss
                loss = self.nt_xent_loss(similarity_matrix, self.temperature)
            
            if mode == "train":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        return epoch_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, data_path, output_path, **kwargs):
        """
        Trains the backbone using contrastive learning.
        """
        # Hyperparameters
        num_epochs = self.configuration.get('num_epochs', 20)
        learning_rate = self.configuration.get('learning_rate', 1e-5)
        batch_size = self.configuration.get('batch_size', 32)
        weight_decay = self.configuration.get('weight_decay', 1e-4)
        save_interval = self.configuration.get('save_interval', 5)
        self.temperature = self.configuration.get('temperature', 0.5)
        
        train_dir = os.path.join(data_path, 'train')
        val_dir = os.path.join(data_path, 'validation')
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # Create datasets and dataloaders
        train_dataset = SimpleDataset(train_dir, transform=transform)
        val_dataset = SimpleDataset(val_dir, transform=transform)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True
        )
        
        # Set up optimizer with layer-wise learning rate decay
        param_groups = []
        
        # Backbone parameters with layer-wise learning rate decay
        for name, param in self.backbone.named_parameters():
            # Apply different learning rates to different layers
            if 'patch_embed' in name or 'cls_token' in name or 'pos_embed' in name:
                lr_scale = 0.1  # Slower learning for early layers
            elif 'blocks.0.' in name or 'blocks.1.' in name:
                lr_scale = 0.2
            elif 'blocks.2.' in name or 'blocks.3.' in name:
                lr_scale = 0.5
            else:
                lr_scale = 1.0  # Full learning rate for later layers
                
            param_groups.append({
                "params": [param],
                "lr": learning_rate * lr_scale,
                "weight_decay": weight_decay
            })
        
        # Projection head parameters (always use full learning rate)
        param_groups.append({
            "params": self.projection_head.parameters(),
            "lr": learning_rate,
            "weight_decay": weight_decay
        })
        
        # Create optimizer and scheduler
        self.optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
        )
        
        # Create gradient scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler()
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            train_loss = self._process_epoch_dataloader(train_loader, self.optimizer, scaler, mode="train")
            
            # Validation phase
            with torch.no_grad():
                val_loss = self._process_epoch_dataloader(val_loader, self.optimizer, scaler, mode="validate")
            
            # Log progress
            logger.info(f"Epoch {epoch+1}: Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            print(f"Epoch {epoch+1}: Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            
            # Update learning rate
            scheduler.step()
            
            # Save checkpoint at intervals
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(output_path, f"dino_backbone_epoch_{epoch+1}.pth")
                torch.save({
                    'backbone_state_dict': self.backbone.state_dict(),
                    'projection_head_state_dict': self.projection_head.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, checkpoint_path)
                logger.info(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = os.path.join(output_path, "best_dino_backbone.pth")
                torch.save({
                    'backbone_state_dict': self.backbone.state_dict(),
                    'projection_head_state_dict': self.projection_head.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, best_checkpoint_path)
                logger.info(f"Best model updated at epoch {epoch+1} with validation loss {val_loss:.4f}")
    
    def embed_images(self, images: List[Image.Image]):
        """
        Generate embeddings for a list of images.
        """
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        batch_tensors = []
        for image in images:
            try:
                if not image.mode == "RGB":
                    image = image.convert("RGB")
                image_tensor = transform(image).unsqueeze(0)
                batch_tensors.append(image_tensor)
            except Exception as e:
                raise Exception(f"Failed to preprocess image: {traceback.format_exc()}")
        
        batch = torch.cat(batch_tensors).to(self.device)
        
        with torch.no_grad():
            self.backbone.eval()
            batch_embeddings = self.backbone(batch)
        
        return batch_embeddings.cpu().numpy().tolist()
    
    def embed(self, batch: List[dl.Item], **kwargs):
        """
        Generate embeddings for a list of Dataloop items.
        """
        features = []
        for item in batch:
            if "image" in item.mimetype:
                orig_image = Image.fromarray(
                    item.download(save_locally=False, to_array=True)
                )
                image_features = self.embed_images([orig_image])
                features.extend(image_features)
            else:
                features.append(None)
        return features
    
    def prepare_item_func(self, item: dl.Item):
        """
        Prepare an item for processing.
        """
        if 'image' not in item.mimetype:
            raise ValueError(f"Item {item.id} is not an image, but {item.mimetype}.")
        return item
    
    def convert_from_dtlpy(self, data_path, **kwargs):
        """ Convert Dataloop structure data to model structured

            Virtual method - need to implement

            e.g. take dlp dir structure and construct annotation file

        :param data_path: str local File System directory path where
                           we already downloaded the data from dataloop platform
        :return:
        """