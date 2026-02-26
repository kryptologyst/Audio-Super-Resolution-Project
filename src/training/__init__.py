"""
Training utilities for audio super-resolution models.

This module contains training loops, optimizers, schedulers,
and other utilities for training audio super-resolution models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
import logging


class Trainer:
    """
    Main trainer class for audio super-resolution models.
    
    Handles training loops, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: nn.Module = nn.L1Loss(),
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device('cpu'),
        save_dir: Optional[Path] = None,
        log_interval: int = 100
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            save_dir: Directory to save checkpoints
            log_interval: Interval for logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir or Path('checkpoints')
        self.log_interval = log_interval
        
        # Initialize optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer
        
        # Initialize scheduler
        self.scheduler = scheduler
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Setup logging
        self.setup_logging()
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            input_audio = batch['input'].to(self.device)
            target_audio = batch['target'].to(self.device)
            
            # Add channel dimension if needed
            if input_audio.dim() == 2:
                input_audio = input_audio.unsqueeze(1)
            if target_audio.dim() == 2:
                target_audio = target_audio.unsqueeze(1)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_audio = self.model(input_audio)
            
            # Compute loss
            loss = self.criterion(pred_audio, target_audio)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            # Log intermediate results
            if batch_idx % self.log_interval == 0:
                self.logger.info(
                    f'Epoch {self.epoch}, Batch {batch_idx}/{num_batches}, '
                    f'Loss: {loss.item():.6f}'
                )
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self) -> float:
        """
        Validate the model.
        
        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                input_audio = batch['input'].to(self.device)
                target_audio = batch['target'].to(self.device)
                
                # Add channel dimension if needed
                if input_audio.dim() == 2:
                    input_audio = input_audio.unsqueeze(1)
                if target_audio.dim() == 2:
                    target_audio = target_audio.unsqueeze(1)
                
                # Forward pass
                pred_audio = self.model(input_audio)
                
                # Compute loss
                loss = self.criterion(pred_audio, target_audio)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pth')
            self.logger.info(f'New best model saved with validation loss: {self.best_val_loss:.6f}')
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f'Loaded checkpoint from epoch {self.epoch}')
    
    def train(self, num_epochs: int, resume_from: Optional[Path] = None):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        # Resume from checkpoint if specified
        if resume_from and resume_from.exists():
            self.load_checkpoint(resume_from)
        
        self.logger.info(f'Starting training for {num_epochs} epochs')
        self.logger.info(f'Model has {sum(p.numel() for p in self.model.parameters())} parameters')
        
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log epoch results
            self.logger.info(
                f'Epoch {epoch}: Train Loss: {train_loss:.6f}, '
                f'Val Loss: {val_loss:.6f}, '
                f'LR: {self.optimizer.param_groups[0]["lr"]:.2e}'
            )
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(is_best)
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f'Training completed in {total_time:.2f} seconds')
        self.logger.info(f'Best validation loss: {self.best_val_loss:.6f}')


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adam',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    **kwargs
) -> optim.Optimizer:
    """
    Create optimizer for model.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = 'step',
    **kwargs
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_type: Type of scheduler
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance or None
    """
    if scheduler_type.lower() == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1),
            **kwargs
        )
    elif scheduler_type.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 0),
            **kwargs
        )
    elif scheduler_type.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            **kwargs
        )
    else:
        return None
