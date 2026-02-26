"""
Tests for audio super-resolution models and utilities.

This module contains unit tests for the various components
of the audio super-resolution system.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import SRCNN, UNet, WaveNetSuperResolution, create_model, count_parameters
from data import AudioDataset, generate_synthetic_data
from losses import SpectralLoss, PerceptualLoss, MultiScaleLoss, create_loss_function
from metrics import AudioMetrics, MetricsTracker
from utils import get_device, set_seed, count_parameters as utils_count_parameters


class TestModels:
    """Test model architectures."""
    
    def test_srcnn_creation(self):
        """Test SRCNN model creation."""
        model = SRCNN(input_channels=1, hidden_channels=64, upsampling_factor=2)
        assert isinstance(model, SRCNN)
        assert count_parameters(model) > 0
    
    def test_unet_creation(self):
        """Test UNet model creation."""
        model = UNet(input_channels=1, base_channels=32, upsampling_factor=2)
        assert isinstance(model, UNet)
        assert count_parameters(model) > 0
    
    def test_wavenet_creation(self):
        """Test WaveNet model creation."""
        model = WaveNetSuperResolution(
            input_channels=1, 
            hidden_channels=64, 
            upsampling_factor=2
        )
        assert isinstance(model, WaveNetSuperResolution)
        assert count_parameters(model) > 0
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        models = [
            SRCNN(input_channels=1, upsampling_factor=2),
            UNet(input_channels=1, base_channels=32, upsampling_factor=2),
            WaveNetSuperResolution(input_channels=1, hidden_channels=64, upsampling_factor=2)
        ]
        
        batch_size = 2
        input_length = 1000
        
        for model in models:
            model.eval()
            x = torch.randn(batch_size, 1, input_length)
            
            with torch.no_grad():
                output = model(x)
            
            # Check output shape
            expected_length = input_length * model.upsampling_factor
            assert output.shape == (batch_size, 1, expected_length)
    
    def test_create_model_factory(self):
        """Test model factory function."""
        configs = [
            {'type': 'srcnn', 'config': {'input_channels': 1, 'upsampling_factor': 2}},
            {'type': 'unet', 'config': {'input_channels': 1, 'upsampling_factor': 2}},
            {'type': 'wavenet', 'config': {'input_channels': 1, 'upsampling_factor': 2}}
        ]
        
        for config in configs:
            model = create_model(config['type'], config['config'])
            assert model is not None
            assert count_parameters(model) > 0


class TestData:
    """Test data loading and processing."""
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generate_synthetic_data(
                output_dir=temp_dir,
                num_samples=10,
                sample_rate=16000,
                duration=1.0
            )
            
            # Check that files were created
            temp_path = Path(temp_dir)
            wav_files = list(temp_path.glob('*.wav'))
            assert len(wav_files) == 10
            
            # Check metadata file
            metadata_file = temp_path / 'metadata.csv'
            assert metadata_file.exists()
    
    def test_audio_dataset(self):
        """Test AudioDataset class."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate test data
            generate_synthetic_data(
                output_dir=temp_dir,
                num_samples=5,
                sample_rate=16000,
                duration=1.0
            )
            
            # Create dataset
            dataset = AudioDataset(
                data_dir=temp_dir,
                sample_rate=8000,
                target_sample_rate=16000,
                segment_length=8000,
                augment=False
            )
            
            assert len(dataset) == 5
            
            # Test getting a sample
            sample = dataset[0]
            assert 'input' in sample
            assert 'target' in sample
            assert 'sample_rate' in sample
            
            # Check tensor shapes
            assert sample['input'].shape[0] == 8000  # segment_length
            assert sample['target'].shape[0] == 8000


class TestLosses:
    """Test loss functions."""
    
    def test_spectral_loss(self):
        """Test spectral loss computation."""
        loss_fn = SpectralLoss(n_fft=512, hop_length=128)
        
        batch_size = 2
        length = 1000
        
        pred = torch.randn(batch_size, length)
        target = torch.randn(batch_size, length)
        
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_perceptual_loss(self):
        """Test perceptual loss computation."""
        loss_fn = PerceptualLoss(feature_dim=64)
        
        batch_size = 2
        length = 1000
        
        pred = torch.randn(batch_size, length)
        target = torch.randn(batch_size, length)
        
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_multiscale_loss(self):
        """Test multi-scale loss computation."""
        loss_fn = MultiScaleLoss(
            time_weight=1.0,
            spectral_weight=0.1,
            perceptual_weight=0.1,
            scales=[1, 2]
        )
        
        batch_size = 2
        length = 1000
        
        pred = torch.randn(batch_size, length)
        target = torch.randn(batch_size, length)
        
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_loss_factory(self):
        """Test loss function factory."""
        loss_types = ['l1', 'l2', 'spectral', 'perceptual', 'multiscale']
        
        for loss_type in loss_types:
            loss_fn = create_loss_function(loss_type)
            assert loss_fn is not None


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_audio_metrics(self):
        """Test audio metrics computation."""
        metrics_calc = AudioMetrics(sample_rate=16000)
        
        # Create test signals
        length = 16000  # 1 second at 16kHz
        target = np.random.randn(length)
        pred = target + 0.1 * np.random.randn(length)  # Add some noise
        
        metrics = metrics_calc.compute_all_metrics(pred, target)
        
        # Check that metrics are computed
        expected_metrics = ['pesq', 'stoi', 'si_sdr', 'snr', 'spectral_mse', 'spectral_l1']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
    
    def test_metrics_tracker(self):
        """Test metrics tracking."""
        tracker = MetricsTracker()
        
        # Add some metrics
        metrics1 = {'pesq': 2.5, 'stoi': 0.8, 'si_sdr': 15.0}
        metrics2 = {'pesq': 2.7, 'stoi': 0.85, 'si_sdr': 16.0}
        
        tracker.update(metrics1)
        tracker.update(metrics2)
        
        # Check statistics
        stats = tracker.compute_statistics()
        assert 'pesq' in stats
        assert stats['pesq']['count'] == 2
        assert stats['pesq']['mean'] == 2.6


class TestUtils:
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand = torch.rand(5)
        np_rand = np.random.rand(5)
        
        # Set seed again and generate again
        set_seed(42)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        # Should be the same
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = SRCNN(input_channels=1, hidden_channels=32)
        param_count = utils_count_parameters(model)
        assert param_count > 0
        assert isinstance(param_count, int)


if __name__ == '__main__':
    pytest.main([__file__])
