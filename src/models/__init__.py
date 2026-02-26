"""
Audio Super-Resolution Models

This module contains various neural network architectures for audio super-resolution,
including SRCNN, UNet, and WaveNet-style models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math


class SRCNN(nn.Module):
    """
    Super-Resolution CNN for audio enhancement.
    
    A lightweight CNN architecture that learns to map low-resolution audio
    features to high-resolution features.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        num_layers: int = 3,
        upsampling_factor: int = 2
    ):
        """
        Initialize SRCNN model.
        
        Args:
            input_channels: Number of input channels
            hidden_channels: Number of hidden channels
            kernel_size: Convolution kernel size
            num_layers: Number of convolutional layers
            upsampling_factor: Upsampling factor for super-resolution
        """
        super().__init__()
        
        self.upsampling_factor = upsampling_factor
        
        # Build convolutional layers
        layers = []
        in_channels = input_channels
        
        for i in range(num_layers):
            out_channels = hidden_channels if i < num_layers - 1 else input_channels
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(inplace=True) if i < num_layers - 1 else nn.Identity()
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SRCNN.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
            
        Returns:
            Enhanced audio tensor
        """
        # Upsample input if needed
        if self.upsampling_factor > 1:
            x = F.interpolate(x, scale_factor=self.upsampling_factor, mode='linear', align_corners=False)
        
        return self.conv_layers(x)


class UNet(nn.Module):
    """
    U-Net architecture for audio super-resolution.
    
    Features skip connections to preserve fine details during upsampling.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 64,
        upsampling_factor: int = 2,
        num_levels: int = 4
    ):
        """
        Initialize UNet model.
        
        Args:
            input_channels: Number of input channels
            base_channels: Base number of channels
            upsampling_factor: Upsampling factor
            num_levels: Number of encoder/decoder levels
        """
        super().__init__()
        
        self.upsampling_factor = upsampling_factor
        self.num_levels = num_levels
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        # Build encoder
        in_channels = input_channels
        for i in range(num_levels):
            out_channels = base_channels * (2 ** i)
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            self.skip_connections.append(
                nn.Conv1d(out_channels, out_channels, 1)
            )
            in_channels = out_channels
        
        # Bottleneck
        bottleneck_channels = base_channels * (2 ** num_levels)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels, bottleneck_channels, 3, padding=1),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck_channels, bottleneck_channels, 3, padding=1),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )
        
        # Build decoder
        in_channels = bottleneck_channels
        for i in range(num_levels - 1, -1, -1):
            out_channels = base_channels * (2 ** i)
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels, out_channels, 2, stride=2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels
        
        # Final output layer
        self.final_conv = nn.Conv1d(base_channels, input_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through UNet.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
            
        Returns:
            Enhanced audio tensor
        """
        # Upsample input if needed
        if self.upsampling_factor > 1:
            x = F.interpolate(x, scale_factor=self.upsampling_factor, mode='linear', align_corners=False)
        
        # Encoder
        skip_outputs = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            skip_outputs.append(x)
            x = F.max_pool1d(x, 2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer(x)
            if i < len(skip_outputs):
                skip = self.skip_connections[-(i+1)](skip_outputs[-(i+1)])
                x = x + skip
        
        return self.final_conv(x)


class WaveNetBlock(nn.Module):
    """WaveNet-style dilated convolution block."""
    
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.gate = nn.Conv1d(channels, channels, 1)
        self.residual = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = self.residual(x)
        gated = torch.tanh(self.conv1(x)) * torch.sigmoid(self.conv2(x))
        gated = self.gate(gated)
        return gated + residual, gated


class WaveNetSuperResolution(nn.Module):
    """
    WaveNet-style model for audio super-resolution.
    
    Uses dilated convolutions to capture long-range dependencies
    in audio signals.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 128,
        num_blocks: int = 8,
        num_layers_per_block: int = 4,
        upsampling_factor: int = 2
    ):
        """
        Initialize WaveNet super-resolution model.
        
        Args:
            input_channels: Number of input channels
            hidden_channels: Number of hidden channels
            num_blocks: Number of WaveNet blocks
            num_layers_per_block: Number of layers per block
            upsampling_factor: Upsampling factor
        """
        super().__init__()
        
        self.upsampling_factor = upsampling_factor
        
        # Input projection
        self.input_conv = nn.Conv1d(input_channels, hidden_channels, 1)
        
        # WaveNet blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            for j in range(num_layers_per_block):
                dilation = 2 ** j
                self.blocks.append(WaveNetBlock(hidden_channels, dilation))
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, input_channels, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through WaveNet model.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
            
        Returns:
            Enhanced audio tensor
        """
        # Upsample input if needed
        if self.upsampling_factor > 1:
            x = F.interpolate(x, scale_factor=self.upsampling_factor, mode='linear', align_corners=False)
        
        x = self.input_conv(x)
        
        # Process through WaveNet blocks
        skip_connections = []
        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        
        # Sum skip connections
        x = sum(skip_connections)
        
        return self.output_conv(x)


def create_model(
    model_type: str,
    config: Dict[str, Any]
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('srcnn', 'unet', 'wavenet')
        config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    if model_type.lower() == 'srcnn':
        return SRCNN(**config)
    elif model_type.lower() == 'unet':
        return UNet(**config)
    elif model_type.lower() == 'wavenet':
        return WaveNetSuperResolution(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
