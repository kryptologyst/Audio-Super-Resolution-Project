#!/usr/bin/env python3
"""
Evaluation script for audio super-resolution models.

This script evaluates trained models on test data and computes
comprehensive metrics including PESQ, STOI, SI-SDR, etc.
"""

import argparse
import sys
from pathlib import Path
import torch
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from models import create_model
from data import create_data_loaders
from metrics import AudioMetrics, evaluate_model
from utils import get_device, set_seed, load_config, setup_logging


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate audio super-resolution model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--split', type=str, default='test', help='Data split to evaluate on')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Setup device
    device_config = config.get('device', 'auto')
    if device_config == 'auto':
        device = get_device()
    else:
        device = torch.device(device_config)
    
    print(f"Using device: {device}")
    
    # Setup logging
    setup_logging(config.get('logging', {}).get('level', 'INFO'))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loader
    data_config = config['data']
    test_dir = Path(args.data_dir) / args.split
    
    if not test_dir.exists():
        print(f"Test directory {test_dir} does not exist!")
        return
    
    test_loader = create_data_loaders(
        train_dir=test_dir,  # Using train_dir parameter for test data
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        sample_rate=data_config['sample_rate'],
        target_sample_rate=data_config['target_sample_rate'],
        segment_length=data_config.get('segment_length'),
        augment=False  # No augmentation for evaluation
    )['train']  # Get the loader from the returned dict
    
    # Create model
    model_config = config['model']
    model = create_model(model_config['type'], model_config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {args.checkpoint}")
    print(f"Model epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Create metrics calculator
    eval_config = config['evaluation']
    metrics_calculator = AudioMetrics(sample_rate=eval_config['sample_rate'])
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device, metrics_calculator)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper()}: {value:.4f}")
    
    # Save results
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(output_dir / 'evaluation_results.csv', index=False)
    
    # Create visualization
    create_evaluation_plots(model, test_loader, device, output_dir)
    
    print(f"\nResults saved to {output_dir}")


def create_evaluation_plots(model, dataloader, device, output_dir):
    """Create evaluation plots and audio samples."""
    model.eval()
    
    # Get a few samples for visualization
    samples = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Take first 3 batches
                break
            
            input_audio = batch['input'].to(device)
            target_audio = batch['target'].to(device)
            
            # Add channel dimension if needed
            if input_audio.dim() == 2:
                input_audio = input_audio.unsqueeze(1)
            if target_audio.dim() == 2:
                target_audio = target_audio.unsqueeze(1)
            
            # Forward pass
            pred_audio = model(input_audio).squeeze(1)
            
            # Convert to numpy
            input_np = input_audio.squeeze(1).cpu().numpy()
            target_np = target_audio.squeeze(1).cpu().numpy()
            pred_np = pred_audio.cpu().numpy()
            
            samples.append({
                'input': input_np,
                'target': target_np,
                'pred': pred_np
            })
    
    # Create waveform plots
    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 4*len(samples)))
    if len(samples) == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        # Plot waveforms
        time_axis = np.linspace(0, len(sample['input'][0])/16000, len(sample['input'][0]))
        
        axes[i, 0].plot(time_axis, sample['input'][0])
        axes[i, 0].set_title(f'Sample {i+1} - Input (Low-res)')
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('Amplitude')
        
        axes[i, 1].plot(time_axis, sample['target'][0])
        axes[i, 1].set_title(f'Sample {i+1} - Target (High-res)')
        axes[i, 1].set_xlabel('Time (s)')
        axes[i, 1].set_ylabel('Amplitude')
        
        axes[i, 2].plot(time_axis, sample['pred'][0])
        axes[i, 2].set_title(f'Sample {i+1} - Prediction')
        axes[i, 2].set_xlabel('Time (s)')
        axes[i, 2].set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'waveform_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create spectrogram plots
    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 4*len(samples)))
    if len(samples) == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        # Compute spectrograms
        input_spec = np.abs(np.fft.fft(sample['input'][0]))
        target_spec = np.abs(np.fft.fft(sample['target'][0]))
        pred_spec = np.abs(np.fft.fft(sample['pred'][0]))
        
        # Plot spectrograms
        axes[i, 0].plot(input_spec[:len(input_spec)//2])
        axes[i, 0].set_title(f'Sample {i+1} - Input Spectrum')
        axes[i, 0].set_xlabel('Frequency Bin')
        axes[i, 0].set_ylabel('Magnitude')
        
        axes[i, 1].plot(target_spec[:len(target_spec)//2])
        axes[i, 1].set_title(f'Sample {i+1} - Target Spectrum')
        axes[i, 1].set_xlabel('Frequency Bin')
        axes[i, 1].set_ylabel('Magnitude')
        
        axes[i, 2].plot(pred_spec[:len(pred_spec)//2])
        axes[i, 2].set_title(f'Sample {i+1} - Prediction Spectrum')
        axes[i, 2].set_xlabel('Frequency Bin')
        axes[i, 2].set_ylabel('Magnitude')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spectrum_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Evaluation plots saved!")


if __name__ == '__main__':
    main()
