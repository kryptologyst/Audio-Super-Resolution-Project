"""
Loss Functions for Audio Super-Resolution

This module contains various loss functions suitable for training
audio super-resolution models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class SpectralLoss(nn.Module):
    """
    Spectral loss computed in the frequency domain.
    
    Measures the difference between magnitude spectrograms
    of predicted and target audio.
    """
    
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        window: str = 'hann'
    ):
        """
        Initialize spectral loss.
        
        Args:
            n_fft: FFT window size
            hop_length: Hop length for STFT
            win_length: Window length (defaults to n_fft)
            window: Window function type
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        
        # Create window function
        if window == 'hann':
            self.register_buffer('window', torch.hann_window(self.win_length))
        else:
            self.register_buffer('window', torch.ones(self.win_length))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral loss.
        
        Args:
            pred: Predicted audio tensor
            target: Target audio tensor
            
        Returns:
            Spectral loss value
        """
        # Compute STFT
        pred_stft = torch.stft(
            pred, self.n_fft, self.hop_length, self.win_length, self.window
        )
        target_stft = torch.stft(
            target, self.n_fft, self.hop_length, self.win_length, self.window
        )
        
        # Compute magnitude spectrograms
        pred_mag = torch.sqrt(pred_stft[..., 0]**2 + pred_stft[..., 1]**2)
        target_mag = torch.sqrt(target_stft[..., 0]**2 + target_stft[..., 1]**2)
        
        # Compute L1 loss on magnitude spectrograms
        return F.l1_loss(pred_mag, target_mag)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained audio features.
    
    Computes loss in a learned feature space that better
    captures perceptual differences.
    """
    
    def __init__(self, feature_dim: int = 128):
        """
        Initialize perceptual loss.
        
        Args:
            feature_dim: Dimension of feature space
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # Simple feature extractor (in practice, could use pre-trained model)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, 15, padding=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 15, padding=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 15, padding=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, feature_dim, 15, padding=7),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted audio tensor
            target: Target audio tensor
            
        Returns:
            Perceptual loss value
        """
        # Ensure input has channel dimension
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        
        # Extract features
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        # Compute L2 loss in feature space
        return F.mse_loss(pred_features, target_features)


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss combining multiple loss functions.
    
    Combines time-domain, frequency-domain, and perceptual losses
    at different scales for robust training.
    """
    
    def __init__(
        self,
        time_weight: float = 1.0,
        spectral_weight: float = 0.1,
        perceptual_weight: float = 0.1,
        scales: Tuple[int, ...] = (1, 2, 4)
    ):
        """
        Initialize multi-scale loss.
        
        Args:
            time_weight: Weight for time-domain loss
            spectral_weight: Weight for spectral loss
            perceptual_weight: Weight for perceptual loss
            scales: Downsampling scales for multi-scale processing
        """
        super().__init__()
        self.time_weight = time_weight
        self.spectral_weight = spectral_weight
        self.perceptual_weight = perceptual_weight
        self.scales = scales
        
        # Initialize loss components
        self.spectral_loss = SpectralLoss()
        self.perceptual_loss = PerceptualLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale loss.
        
        Args:
            pred: Predicted audio tensor
            target: Target audio tensor
            
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        
        # Time-domain loss
        if self.time_weight > 0:
            time_loss = F.l1_loss(pred, target)
            total_loss += self.time_weight * time_loss
        
        # Spectral loss
        if self.spectral_weight > 0:
            spectral_loss = self.spectral_loss(pred, target)
            total_loss += self.spectral_weight * spectral_loss
        
        # Perceptual loss
        if self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_loss(pred, target)
            total_loss += self.perceptual_weight * perceptual_loss
        
        # Multi-scale losses
        for scale in self.scales:
            if scale > 1:
                # Downsample
                pred_down = F.avg_pool1d(pred.unsqueeze(0), scale).squeeze(0)
                target_down = F.avg_pool1d(target.unsqueeze(0), scale).squeeze(0)
                
                # Compute loss at this scale
                scale_loss = F.l1_loss(pred_down, target_down)
                total_loss += scale_loss / scale
        
        return total_loss


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for GAN-style training.
    
    Can be used to train a discriminator alongside the generator
    for more realistic audio generation.
    """
    
    def __init__(self, discriminator: nn.Module):
        """
        Initialize adversarial loss.
        
        Args:
            discriminator: Discriminator network
        """
        super().__init__()
        self.discriminator = discriminator
    
    def generator_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Compute generator loss.
        
        Args:
            pred: Predicted audio tensor
            
        Returns:
            Generator loss value
        """
        # Generator wants discriminator to classify predictions as real
        fake_pred = self.discriminator(pred)
        return F.binary_cross_entropy_with_logits(
            fake_pred, torch.ones_like(fake_pred)
        )
    
    def discriminator_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute discriminator loss.
        
        Args:
            pred: Predicted audio tensor
            target: Target audio tensor
            
        Returns:
            Discriminator loss value
        """
        # Real samples
        real_pred = self.discriminator(target)
        real_loss = F.binary_cross_entropy_with_logits(
            real_pred, torch.ones_like(real_pred)
        )
        
        # Fake samples
        fake_pred = self.discriminator(pred.detach())
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_pred, torch.zeros_like(fake_pred)
        )
        
        return real_loss + fake_loss


def create_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss function
        **kwargs: Loss function arguments
        
    Returns:
        Initialized loss function
    """
    if loss_type.lower() == 'l1':
        return nn.L1Loss()
    elif loss_type.lower() == 'l2' or loss_type.lower() == 'mse':
        return nn.MSELoss()
    elif loss_type.lower() == 'spectral':
        return SpectralLoss(**kwargs)
    elif loss_type.lower() == 'perceptual':
        return PerceptualLoss(**kwargs)
    elif loss_type.lower() == 'multiscale':
        return MultiScaleLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
