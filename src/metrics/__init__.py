"""
Evaluation Metrics for Audio Super-Resolution

This module contains various metrics for evaluating the quality
of audio super-resolution models.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import librosa
from pesq import pesq
from pystoi import stoi
import warnings
warnings.filterwarnings('ignore')


class AudioMetrics:
    """
    Comprehensive audio quality metrics calculator.
    
    Computes various metrics including PESQ, STOI, SI-SDR,
    and perceptual metrics for audio super-resolution evaluation.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio metrics calculator.
        
        Args:
            sample_rate: Sample rate for audio processing
        """
        self.sample_rate = sample_rate
    
    def compute_all_metrics(
        self, 
        pred: np.ndarray, 
        target: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            pred: Predicted audio signal
            target: Target audio signal
            
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        
        # Ensure same length
        min_length = min(len(pred), len(target))
        pred = pred[:min_length]
        target = target[:min_length]
        
        # PESQ
        try:
            metrics['pesq'] = self.compute_pesq(pred, target)
        except Exception as e:
            print(f"PESQ computation failed: {e}")
            metrics['pesq'] = 0.0
        
        # STOI
        try:
            metrics['stoi'] = self.compute_stoi(pred, target)
        except Exception as e:
            print(f"STOI computation failed: {e}")
            metrics['stoi'] = 0.0
        
        # SI-SDR
        metrics['si_sdr'] = self.compute_si_sdr(pred, target)
        
        # SNR
        metrics['snr'] = self.compute_snr(pred, target)
        
        # Spectral metrics
        metrics['spectral_mse'] = self.compute_spectral_mse(pred, target)
        metrics['spectral_l1'] = self.compute_spectral_l1(pred, target)
        
        return metrics
    
    def compute_pesq(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Compute PESQ (Perceptual Evaluation of Speech Quality).
        
        Args:
            pred: Predicted audio signal
            target: Target audio signal
            
        Returns:
            PESQ score
        """
        # PESQ expects 16kHz audio
        if self.sample_rate != 16000:
            pred_16k = librosa.resample(pred, orig_sr=self.sample_rate, target_sr=16000)
            target_16k = librosa.resample(target, orig_sr=self.sample_rate, target_sr=16000)
        else:
            pred_16k = pred
            target_16k = target
        
        return pesq(16000, target_16k, pred_16k, 'wb')
    
    def compute_stoi(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Compute STOI (Short-Time Objective Intelligibility).
        
        Args:
            pred: Predicted audio signal
            target: Target audio signal
            
        Returns:
            STOI score
        """
        return stoi(target, pred, self.sample_rate, extended=False)
    
    def compute_si_sdr(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Compute SI-SDR (Scale-Invariant Signal-to-Distortion Ratio).
        
        Args:
            pred: Predicted audio signal
            target: Target audio signal
            
        Returns:
            SI-SDR score in dB
        """
        # Compute optimal scaling
        alpha = np.dot(target, pred) / np.dot(pred, pred)
        
        # Scale-invariant signal
        target_scaled = alpha * pred
        
        # Compute SI-SDR
        signal_power = np.sum(target_scaled ** 2)
        noise_power = np.sum((target - target_scaled) ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        si_sdr = 10 * np.log10(signal_power / noise_power)
        return si_sdr
    
    def compute_snr(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Compute SNR (Signal-to-Noise Ratio).
        
        Args:
            pred: Predicted audio signal
            target: Target audio signal
            
        Returns:
            SNR score in dB
        """
        signal_power = np.sum(target ** 2)
        noise_power = np.sum((target - pred) ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def compute_spectral_mse(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Compute MSE in spectral domain.
        
        Args:
            pred: Predicted audio signal
            target: Target audio signal
            
        Returns:
            Spectral MSE
        """
        # Compute spectrograms
        pred_stft = librosa.stft(pred, n_fft=1024, hop_length=256)
        target_stft = librosa.stft(target, n_fft=1024, hop_length=256)
        
        # Compute magnitude spectrograms
        pred_mag = np.abs(pred_stft)
        target_mag = np.abs(target_stft)
        
        # Compute MSE
        mse = np.mean((pred_mag - target_mag) ** 2)
        return mse
    
    def compute_spectral_l1(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Compute L1 loss in spectral domain.
        
        Args:
            pred: Predicted audio signal
            target: Target audio signal
            
        Returns:
            Spectral L1 loss
        """
        # Compute spectrograms
        pred_stft = librosa.stft(pred, n_fft=1024, hop_length=256)
        target_stft = librosa.stft(target, n_fft=1024, hop_length=256)
        
        # Compute magnitude spectrograms
        pred_mag = np.abs(pred_stft)
        target_mag = np.abs(target_stft)
        
        # Compute L1 loss
        l1 = np.mean(np.abs(pred_mag - target_mag))
        return l1


class MetricsTracker:
    """
    Tracks metrics across multiple samples.
    
    Computes mean, std, and other statistics for batches of metrics.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics_history: Dict[str, List[float]] = {}
    
    def update(self, metrics: Dict[str, float]) -> None:
        """
        Update metrics history.
        
        Args:
            metrics: Dictionary of metric values
        """
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
    
    def compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for all tracked metrics.
        
        Returns:
            Dictionary containing mean, std, min, max for each metric
        """
        statistics = {}
        
        for metric_name, values in self.metrics_history.items():
            if values:
                statistics[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return statistics
    
    def reset(self) -> None:
        """Reset metrics history."""
        self.metrics_history = {}
    
    def get_summary(self) -> str:
        """
        Get a formatted summary of metrics.
        
        Returns:
            Formatted string summary
        """
        stats = self.compute_statistics()
        
        summary = "Audio Quality Metrics Summary:\n"
        summary += "=" * 40 + "\n"
        
        for metric_name, metric_stats in stats.items():
            summary += f"{metric_name.upper()}:\n"
            summary += f"  Mean: {metric_stats['mean']:.4f} ± {metric_stats['std']:.4f}\n"
            summary += f"  Range: [{metric_stats['min']:.4f}, {metric_stats['max']:.4f}]\n"
            summary += f"  Samples: {metric_stats['count']}\n\n"
        
        return summary


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    metrics_calculator: AudioMetrics
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        metrics_calculator: AudioMetrics instance
        
    Returns:
        Dictionary of average metrics
    """
    model.eval()
    tracker = MetricsTracker()
    
    with torch.no_grad():
        for batch in dataloader:
            input_audio = batch['input'].to(device)
            target_audio = batch['target'].to(device)
            
            # Forward pass
            pred_audio = model(input_audio.unsqueeze(1)).squeeze(1)
            
            # Convert to numpy for metrics computation
            pred_np = pred_audio.cpu().numpy()
            target_np = target_audio.cpu().numpy()
            
            # Compute metrics for each sample in batch
            for i in range(pred_np.shape[0]):
                metrics = metrics_calculator.compute_all_metrics(
                    pred_np[i], target_np[i]
                )
                tracker.update(metrics)
    
    # Compute average metrics
    stats = tracker.compute_statistics()
    avg_metrics = {name: stats[name]['mean'] for name in stats.keys()}
    
    return avg_metrics
