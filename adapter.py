import dtlpy as dl
import torchvision.transforms as transforms
import traceback
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
from typing import List
import os
import glob
import numpy as np
from tqdm import tqdm

logger = logging.getLogger("[DINOV2-ADAPTER]")
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Suppress PIL debug logging
logging.getLogger('PIL').setLevel(logging.WARNING)

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
        return image


class DINOv2Adapter(dl.BaseModelAdapter):
    def __init__(self, model_entity=None):
        self.device = None
        self.backbone = None
        self.projection_head = None
        self.momentum_backbone = None
        self.momentum_projection_head = None
        self.optimizer = None
        self.momentum = 0.999
        super().__init__(model_entity)
        
    def load(self, local_path, **kwargs):
        """
        Loads model checkpoint from a local directory.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load a pre-trained DINOv2 backbone
        model_name = self.configuration.get("model_name", "dinov2_vits14")
        logger.info(f"Loading DINOv2 backbone: {model_name}")
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.backbone.to(self.device)
        
        # Create projection head with larger dimensions and more layers
        input_dim = 384 if "vits" in model_name else 768 if "vitb" in model_name else 1024
        
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )
        self.projection_head.to(self.device)
        
        # Create momentum encoder to prevent collapse
        self.momentum_backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.momentum_backbone.to(self.device)
        
        self.momentum_projection_head = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )
        self.momentum_projection_head.to(self.device)
        
        # Initialize momentum encoder with same weights
        self._copy_params(self.backbone, self.momentum_backbone)
        self._copy_params(self.projection_head, self.momentum_projection_head)
        
        # Freeze momentum encoder
        for param in self.momentum_backbone.parameters():
            param.requires_grad = False
        for param in self.momentum_projection_head.parameters():
            param.requires_grad = False
        
        # Update the feature set size in the configuration
        self.model_entity.configuration["embeddings_size"] = input_dim
        self.model_entity.update(True)

        # Load checkpoint if exists
        checkpoint_path = os.path.join(local_path, 'dino_backbone_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            try:
                missing_keys, unexpected_keys = self.backbone.load_state_dict(
                    checkpoint['backbone_state_dict'], strict=False
                )
                if missing_keys:
                    logger.warning(f"Missing keys in backbone: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys in backbone: {unexpected_keys}")
                    
                self.projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
                
                # Update momentum encoder
                self._copy_params(self.backbone, self.momentum_backbone)
                self._copy_params(self.projection_head, self.momentum_projection_head)
                
                logger.info("Checkpoint loaded successfully")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                logger.info("Proceeding with pre-trained backbone and new projection head")
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
    
    def _copy_params(self, source, target):
        """Copy parameters from source to target network."""
        for param_source, param_target in zip(source.parameters(), target.parameters()):
            param_target.data.copy_(param_source.data)

    def _update_momentum_encoder(self):
        """Update momentum encoder with exponential moving average."""
        for param_online, param_target in zip(self.backbone.parameters(), self.momentum_backbone.parameters()):
            param_target.data = param_target.data * self.momentum + param_online.data * (1. - self.momentum)
        
        for param_online, param_target in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
            param_target.data = param_target.data * self.momentum + param_online.data * (1. - self.momentum)
    
    def augment(self, images):
        """
        Apply random augmentations to tensor images for contrastive learning.
        """
        # Random resized crop
        h, w = images.shape[-2], images.shape[-1]
        
        # Calculate crop parameters similar to RandomResizedCrop
        scale_min, scale_max = 0.2, 1.0
        ratio_min, ratio_max = 3./4., 4./3.
        
        for _ in range(10):  # Try up to 10 times to get a valid crop
            target_area = h * w * torch.empty(1).uniform_(scale_min, scale_max).item()
            log_ratio = torch.empty(1).uniform_(torch.log(torch.tensor(ratio_min)), torch.log(torch.tensor(ratio_max))).item()
            aspect_ratio = torch.exp(torch.tensor(log_ratio)).item()
            
            crop_w = int(round((target_area * aspect_ratio) ** 0.5))
            crop_h = int(round((target_area / aspect_ratio) ** 0.5))
            
            if 0 < crop_h <= h and 0 < crop_w <= w:
                top = torch.randint(0, h - crop_h + 1, (1,)).item()
                left = torch.randint(0, w - crop_w + 1, (1,)).item()
                break
        else:
            # Fallback to center crop if no valid crop found
            crop_h, crop_w = min(h, w), min(h, w)
            top = (h - crop_h) // 2
            left = (w - crop_w) // 2
        
        # Perform the crop and resize
        images = transforms.functional.crop(images, top, left, crop_h, crop_w)
        images = transforms.functional.resize(images, (224, 224))
        
        # Color jitter
        if torch.rand(1) < 0.8:
            brightness_factor = 1 + 0.8 * (torch.rand(1) - 0.5)
            contrast_factor = 1 + 0.8 * (torch.rand(1) - 0.5)
            saturation_factor = 1 + 0.8 * (torch.rand(1) - 0.5)
            hue_factor = 0.2 * (torch.rand(1) - 0.5)
            
            images = transforms.functional.adjust_brightness(images, brightness_factor.item())
            images = transforms.functional.adjust_contrast(images, contrast_factor.item())
            images = transforms.functional.adjust_saturation(images, saturation_factor.item())
            images = transforms.functional.adjust_hue(images, hue_factor.item())
        
        # Random grayscale
        if torch.rand(1) < 0.2:
            images = transforms.functional.rgb_to_grayscale(images, num_output_channels=3)
        
        # Random horizontal flip
        if torch.rand(1) < 0.5:
            images = transforms.functional.hflip(images)
        
        # Gaussian blur
        if torch.rand(1) < 0.5:
            kernel_size = 23
            sigma = 0.1 + (2.0 - 0.1) * torch.rand(1)
            images = transforms.functional.gaussian_blur(images, kernel_size, sigma.item())
        
        # Normalize
        images = transforms.functional.normalize(
            images, 
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        return images
    
    def _process_epoch_dataloader(self, dataloader, optimizer, scaler, mode="train"):
        """
        Process one epoch using contrastive learning with momentum encoder.
        """
        epoch_loss = 0.0
        num_batches = 0
        
        if mode == "train":
            self.backbone.train()
            self.projection_head.train()
            self.momentum_backbone.eval()
            self.momentum_projection_head.eval()
        else:
            self.backbone.eval()
            self.projection_head.eval()
            self.momentum_backbone.eval()
            self.momentum_projection_head.eval()
        
        for batch in tqdm(dataloader, unit="batch"):
            images = batch.to(self.device)
            batch_size = images.shape[0]
            
            # Skip small batches
            if batch_size < 2:
                continue
            
            if mode == "train":
                optimizer.zero_grad()
            
            with torch.autocast(device_type=self.device, enabled=(self.device == "cuda")):
                # Create two randomly augmented views of the same images
                view1 = self.augment(images)
                view2 = self.augment(images)
                
                # Online network (trainable)
                emb1 = self.backbone(view1)
                z1 = self.projection_head(emb1)
                z1 = F.normalize(z1, dim=1)
                
                # Momentum network (not trainable)
                with torch.no_grad():
                    emb2 = self.momentum_backbone(view2)
                    z2 = self.momentum_projection_head(emb2)
                    z2 = F.normalize(z2, dim=1)
                
                # Compute loss using asymmetric approach
                # Positive pairs: z1[i] should be similar to z2[i]
                pos_sim = torch.sum(z1 * z2, dim=1)  # [batch_size]
                
                # Negative pairs: z1[i] should be dissimilar to z2[j] where j != i
                all_sim = torch.matmul(z1, z2.T)  # [batch_size, batch_size]
                
                # Apply temperature
                pos_sim = pos_sim / self.temperature
                all_sim = all_sim / self.temperature
                
                # Compute InfoNCE loss
                loss = 0
                for i in range(batch_size):
                    numerator = torch.exp(pos_sim[i])
                    denominator = torch.sum(torch.exp(all_sim[i]))
                    loss += -torch.log(numerator / denominator)
                
                loss = loss / batch_size
            
            if mode == "train":
                scaler.scale(loss).backward()
                
                # Gradient clipping to prevent explosion
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.projection_head.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                # Update momentum encoder
                self._update_momentum_encoder()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        return epoch_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, data_path, output_path, **kwargs):
        """
        Trains the backbone using contrastive learning.
        """
        # Hyperparameters
        num_epochs = self.configuration.get('num_epochs', 20)
        learning_rate = self.configuration.get('learning_rate', 1e-4)
        batch_size = self.configuration.get('batch_size', 8)
        weight_decay = self.configuration.get('weight_decay', 1e-2)
        save_interval = self.configuration.get('save_interval', 5)
        self.temperature = self.configuration.get('temperature', 0.1)
        patience = self.configuration.get('patience', 10)
        on_epoch_end_callback = kwargs.get('on_epoch_end_callback', None)

        train_dir = os.path.join(data_path, 'train')
        val_dir = os.path.join(data_path, 'validation')
        
        # Base transform for loading images as tensors
        base_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # Deterministic validation transform (fully preprocessed)
        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets and dataloaders
        train_dataset = SimpleDataset(train_dir, transform=base_transform)
        val_dataset = SimpleDataset(val_dir, transform=val_transform)
        
        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
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
        
        # Projection head parameters
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
        
        scaler = torch.GradScaler(device=self.device)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            train_loss = self._process_epoch_dataloader(train_loader, self.optimizer, scaler, mode="train")
            
            # Validation phase
            with torch.no_grad():
                val_loss = self._process_epoch_dataloader(val_loader, self.optimizer, scaler, mode="validate")
            
            # Log progress
            logger.info(f"Epoch {epoch+1}: Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            
            # Update learning rate
            scheduler.step()

            self._update_model_metrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss)

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
                logger.info(f"Checkpoint saved at epoch {epoch+1}")
            
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
                patience_counter = 0
            else:
                patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

            if on_epoch_end_callback is not None:
                on_epoch_end_callback(i_epoch=epoch, n_epoch=num_epochs)
    
    def _update_model_metrics(self, epoch, train_loss, val_loss):
        """
        Updates model metrics in Dataloop platform.
        """
        samples = []
        metrics_dict = {
            'loss/train': train_loss,
            'loss/validation': val_loss
        }
        
        for metric_name, value in metrics_dict.items():
            legend, figure = metric_name.split('/')
            
            if not np.isfinite(value):
                filters = dl.Filters(resource=dl.FiltersResource.METRICS)
                filters.add(field='modelId', values=self.model_entity.id)
                filters.add(field='figure', values=figure)
                filters.add(field='data.x', values=epoch)
                items = self.model_entity.metrics.list(filters=filters)
                
                if items.items_count > 0:
                    value = items.items[0].y
                else:
                    value = 0
                logger.warning(f'Value is not finite. Using value {value} for figure {figure}')
            
            samples.append(dl.PlotSample(
                figure=figure,
                legend=legend,
                x=epoch,
                y=value
            ))
        
        self.model_entity.metrics.create(
            samples=samples,
            dataset_id=self.model_entity.dataset_id
        )

    def embed_images(self, images: List[Image.Image]):
        """
        Generate embeddings for a list of images.
        """
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
            orig_image = Image.fromarray(item.download(save_locally=False, to_array=True))
            image_features = self.embed_images([orig_image])
            features.extend(image_features)
        return features
    
    def prepare_item_func(self, item: dl.Item):
        """
        Prepare an item for processing.
        """
        if 'image' not in item.mimetype:
            raise ValueError(f"Item {item.id} is not an image, but {item.mimetype}.")
        return item
    
    def convert_from_dtlpy(self, data_path, **kwargs):
        # Subsets validation
        subsets = self.model_entity.metadata.get("system", {}).get("subsets", None)
        if 'train' not in subsets:
            raise ValueError('Could not find train set. DINOv2 requires train and validation set for training.')
        if 'validation' not in subsets:
            raise ValueError('Could not find validation set. DINOv2 requires train and validation set for training.')

        for subset, filters_dict in subsets.items():
            filters = dl.Filters(custom_filter=filters_dict)
            pages = self.model_entity.dataset.items.list(filters=filters)
            if pages.items_count == 0:
                raise ValueError(f'Could not find items in subset {subset}.')