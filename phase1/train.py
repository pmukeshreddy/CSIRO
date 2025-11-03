"""
Training and validation functions for metadata predictor.
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm


class Trainer:
    """Trainer class for metadata predictor."""
    
    def __init__(self, model, optimizer, device):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            optimizer: Optimizer instance
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        # Loss functions
        self.regression_loss_fn = nn.MSELoss()
        self.classification_loss_fn = nn.CrossEntropyLoss()
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch.
        
        Args:
            dataloader: Training DataLoader
            
        Returns:
            Dictionary containing epoch metrics
        """
        self.model.train()
        total_loss = 0
        
        # Metrics
        ndvi_losses = []
        height_losses = []
        state_losses = []
        species_losses = []
        
        pbar = tqdm(dataloader, desc='Training')
        for batch in pbar:
            # Move to device
            images = batch['image'].to(self.device)
            ndvi_true = batch['ndvi'].to(self.device)
            height_true = batch['height'].to(self.device)
            state_true = batch['state'].to(self.device)
            species_true = batch['species'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate losses
            ndvi_loss = self.regression_loss_fn(outputs['ndvi'], ndvi_true)
            height_loss = self.regression_loss_fn(outputs['height'], height_true)
            state_loss = self.classification_loss_fn(outputs['state_logits'], state_true)
            species_loss = self.classification_loss_fn(outputs['species_logits'], species_true)
            
            # Combined loss
            loss = ndvi_loss + height_loss + state_loss + species_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            ndvi_losses.append(ndvi_loss.item())
            height_losses.append(height_loss.item())
            state_losses.append(state_loss.item())
            species_losses.append(species_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'ndvi': ndvi_loss.item(),
                'height': height_loss.item(),
                'state': state_loss.item(),
                'species': species_loss.item()
            })
        
        return {
            'total_loss': total_loss / len(dataloader),
            'ndvi_loss': np.mean(ndvi_losses),
            'height_loss': np.mean(height_losses),
            'state_loss': np.mean(state_losses),
            'species_loss': np.mean(species_losses)
        }
    
    def validate(self, dataloader):
        """
        Validate the model.
        
        Args:
            dataloader: Validation DataLoader
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0
        
        # Metrics
        ndvi_losses = []
        height_losses = []
        state_losses = []
        species_losses = []
        
        # Accuracy tracking
        state_correct = 0
        species_correct = 0
        total_samples = 0
        
        pbar = tqdm(dataloader, desc='Validation')
        with torch.no_grad():
            for batch in pbar:
                # Move to device
                images = batch['image'].to(self.device)
                ndvi_true = batch['ndvi'].to(self.device)
                height_true = batch['height'].to(self.device)
                state_true = batch['state'].to(self.device)
                species_true = batch['species'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate losses
                ndvi_loss = self.regression_loss_fn(outputs['ndvi'], ndvi_true)
                height_loss = self.regression_loss_fn(outputs['height'], height_true)
                state_loss = self.classification_loss_fn(outputs['state_logits'], state_true)
                species_loss = self.classification_loss_fn(outputs['species_logits'], species_true)
                
                loss = ndvi_loss + height_loss + state_loss + species_loss
                
                # Track metrics
                total_loss += loss.item()
                ndvi_losses.append(ndvi_loss.item())
                height_losses.append(height_loss.item())
                state_losses.append(state_loss.item())
                species_losses.append(species_loss.item())
                
                # Calculate accuracies
                state_pred = torch.argmax(outputs['state_logits'], dim=1)
                species_pred = torch.argmax(outputs['species_logits'], dim=1)
                
                state_correct += (state_pred == state_true).sum().item()
                species_correct += (species_pred == species_true).sum().item()
                total_samples += images.size(0)
                
                pbar.set_postfix({'loss': loss.item()})
        
        return {
            'total_loss': total_loss / len(dataloader),
            'ndvi_loss': np.mean(ndvi_losses),
            'height_loss': np.mean(height_losses),
            'state_loss': np.mean(state_losses),
            'species_loss': np.mean(species_losses),
            'state_accuracy': state_correct / total_samples,
            'species_accuracy': species_correct / total_samples
        }
