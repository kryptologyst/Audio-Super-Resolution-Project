"""
Audio Data Loading and Preprocessing

This module handles loading, preprocessing, and augmentation of audio data
for super-resolution training and evaluation.
"""

import os
import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional, Union
from pathlib import Path
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import random


class AudioDataset(Dataset):
    """
    Dataset class for audio super-resolution.
    
    Handles loading of audio files and applies preprocessing,
    augmentation, and target generation.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        metadata_file: Optional[str] = None,
        sample_rate: int = 16000,
        target_sample_rate: int = 32000,
        segment_length: Optional[int] = None,
        augment: bool = True,
        noise_factor: float = 0.1,
        pitch_shift_range: Tuple[float, float] = (-2.0, 2.0),
        time_stretch_range: Tuple[float, float] = (0.9, 1.1)
    ):
        """
        Initialize AudioDataset.
        
        Args:
            data_dir: Directory containing audio files
            metadata_file: Optional CSV file with metadata
            sample_rate: Input sample rate
            target_sample_rate: Target sample rate for super-resolution
            segment_length: Length of audio segments (None for full files)
            augment: Whether to apply data augmentation
            noise_factor: Factor for noise augmentation
            pitch_shift_range: Range for pitch shifting (semitones)
            time_stretch_range: Range for time stretching
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        self.segment_length = segment_length
        self.augment = augment
        self.noise_factor = noise_factor
        self.pitch_shift_range = pitch_shift_range
        self.time_stretch_range = time_stretch_range
        
        # Load metadata or scan directory
        if metadata_file and os.path.exists(metadata_file):
            self.metadata = pd.read_csv(metadata_file)
            self.audio_files = self.metadata['path'].tolist()
        else:
            self.audio_files = list(self.data_dir.glob('*.wav')) + list(self.data_dir.glob('*.mp3'))
            self.metadata = None
        
        # Initialize transforms
        self._init_transforms()
        
    def _init_transforms(self):
        """Initialize audio transforms."""
        self.resample_transform = torchaudio.transforms.Resample(
            orig_freq=self.sample_rate,
            new_freq=self.target_sample_rate
        )
        
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single audio sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing 'input', 'target', and metadata
        """
        # Load audio file
        if self.metadata is not None:
            audio_path = self.data_dir / self.audio_files[idx]
        else:
            audio_path = self.audio_files[idx]
        
        # Load audio
        audio, sr = torchaudio.load(str(audio_path))
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Resample to target sample rate
        if sr != self.target_sample_rate:
            audio = self.resample_transform(audio)
        
        # Create low-resolution version
        low_res_audio = self._create_low_resolution(audio)
        
        # Apply segmenting if specified
        if self.segment_length is not None:
            low_res_audio, audio = self._segment_audio(low_res_audio, audio)
        
        # Apply augmentation
        if self.augment:
            low_res_audio, audio = self._augment_audio(low_res_audio, audio)
        
        return {
            'input': low_res_audio.squeeze(0),
            'target': audio.squeeze(0),
            'path': str(audio_path),
            'sample_rate': self.target_sample_rate
        }
    
    def _create_low_resolution(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Create low-resolution version of audio.
        
        Args:
            audio: High-resolution audio tensor
            
        Returns:
            Low-resolution audio tensor
        """
        # Downsample
        downsampled = F.interpolate(
            audio.unsqueeze(0),
            scale_factor=self.sample_rate / self.target_sample_rate,
            mode='linear',
            align_corners=False
        ).squeeze(0)
        
        # Upsample back to target resolution (simulating low-quality upsampling)
        upsampled = F.interpolate(
            downsampled.unsqueeze(0),
            scale_factor=self.target_sample_rate / self.sample_rate,
            mode='linear',
            align_corners=False
        ).squeeze(0)
        
        return upsampled
    
    def _segment_audio(
        self, 
        low_res: torch.Tensor, 
        high_res: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Segment audio into fixed-length chunks.
        
        Args:
            low_res: Low-resolution audio
            high_res: High-resolution audio
            
        Returns:
            Segmented audio tensors
        """
        min_length = min(low_res.shape[-1], high_res.shape[-1])
        
        if min_length <= self.segment_length:
            # Pad if too short
            pad_length = self.segment_length - min_length
            low_res = F.pad(low_res, (0, pad_length))
            high_res = F.pad(high_res, (0, pad_length))
        else:
            # Random crop if too long
            start_idx = random.randint(0, min_length - self.segment_length)
            low_res = low_res[..., start_idx:start_idx + self.segment_length]
            high_res = high_res[..., start_idx:start_idx + self.segment_length]
        
        return low_res, high_res
    
    def _augment_audio(
        self, 
        low_res: torch.Tensor, 
        high_res: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply data augmentation to audio.
        
        Args:
            low_res: Low-resolution audio
            high_res: High-resolution audio
            
        Returns:
            Augmented audio tensors
        """
        # Add noise
        if self.noise_factor > 0:
            noise = torch.randn_like(low_res) * self.noise_factor
            low_res = low_res + noise
            high_res = high_res + noise
        
        # Pitch shifting (using librosa)
        if random.random() < 0.5:
            pitch_shift = random.uniform(*self.pitch_shift_range)
            low_res_np = low_res.numpy()
            high_res_np = high_res.numpy()
            
            low_res_shifted = librosa.effects.pitch_shift(
                low_res_np, sr=self.target_sample_rate, n_steps=pitch_shift
            )
            high_res_shifted = librosa.effects.pitch_shift(
                high_res_np, sr=self.target_sample_rate, n_steps=pitch_shift
            )
            
            low_res = torch.from_numpy(low_res_shifted).float()
            high_res = torch.from_numpy(high_res_shifted).float()
        
        # Time stretching
        if random.random() < 0.5:
            stretch_factor = random.uniform(*self.time_stretch_range)
            low_res_np = low_res.numpy()
            high_res_np = high_res.numpy()
            
            low_res_stretched = librosa.effects.time_stretch(
                low_res_np, rate=stretch_factor
            )
            high_res_stretched = librosa.effects.time_stretch(
                high_res_np, rate=stretch_factor
            )
            
            # Resize to original length
            low_res_stretched = librosa.util.fix_length(
                low_res_stretched, size=len(low_res_np)
            )
            high_res_stretched = librosa.util.fix_length(
                high_res_stretched, size=len(high_res_np)
            )
            
            low_res = torch.from_numpy(low_res_stretched).float()
            high_res = torch.from_numpy(high_res_stretched).float()
        
        return low_res, high_res


def create_data_loaders(
    train_dir: Union[str, Path],
    val_dir: Optional[Union[str, Path]] = None,
    test_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    **dataset_kwargs
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        test_dir: Test data directory
        batch_size: Batch size
        num_workers: Number of worker processes
        **dataset_kwargs: Additional dataset arguments
        
    Returns:
        Dictionary containing data loaders
    """
    loaders = {}
    
    # Training loader
    train_dataset = AudioDataset(train_dir, augment=True, **dataset_kwargs)
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation loader
    if val_dir:
        val_dataset = AudioDataset(val_dir, augment=False, **dataset_kwargs)
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # Test loader
    if test_dir:
        test_dataset = AudioDataset(test_dir, augment=False, **dataset_kwargs)
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loaders


def generate_synthetic_data(
    output_dir: Union[str, Path],
    num_samples: int = 1000,
    sample_rate: int = 32000,
    duration: float = 2.0,
    noise_level: float = 0.1
) -> None:
    """
    Generate synthetic audio data for training.
    
    Args:
        output_dir: Output directory for generated data
        num_samples: Number of samples to generate
        sample_rate: Sample rate for generated audio
        duration: Duration of each sample in seconds
        noise_level: Level of noise to add
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    
    for i in range(num_samples):
        # Generate synthetic audio (mixture of sine waves)
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Multiple frequency components
        frequencies = np.random.uniform(100, 2000, size=np.random.randint(2, 6))
        amplitudes = np.random.uniform(0.1, 0.8, size=len(frequencies))
        
        audio = np.zeros_like(t)
        for freq, amp in zip(frequencies, amplitudes):
            audio += amp * np.sin(2 * np.pi * freq * t)
        
        # Add noise
        noise = np.random.normal(0, noise_level, len(audio))
        audio += noise
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        # Save audio file
        filename = f"synthetic_{i:04d}.wav"
        filepath = output_dir / filename
        sf.write(str(filepath), audio, sample_rate)
        
        metadata.append({
            'id': i,
            'path': filename,
            'sample_rate': sample_rate,
            'duration': duration,
            'frequencies': frequencies.tolist(),
            'amplitudes': amplitudes.tolist()
        })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_dir / 'metadata.csv', index=False)
    
    print(f"Generated {num_samples} synthetic audio samples in {output_dir}")


# Import torch.nn.functional for interpolation
import torch.nn.functional as F
