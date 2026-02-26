#!/usr/bin/env python3
"""
Training script for audio super-resolution models.

This script trains various models (SRCNN, UNet, WaveNet) for audio super-resolution
using the configuration files in the configs/ directory.
"""

import argparse
import sys
from pathlib import Path
import torch
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from models import create_model
from data import create_data_loaders, generate_synthetic_data
from losses import create_loss_function
from training import Trainer, create_optimizer, create_scheduler
from utils import get_device, set_seed, load_config, setup_logging, create_experiment_name, setup_experiment_dir


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train audio super-resolution model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='experiments', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--generate_data', action='store_true', help='Generate synthetic data')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of synthetic samples')
    
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
    
    # Generate synthetic data if requested
    if args.generate_data:
        print("Generating synthetic data...")
        generate_synthetic_data(
            output_dir=Path(args.data_dir) / 'synthetic',
            num_samples=args.num_samples,
            sample_rate=config['data']['target_sample_rate'],
            duration=2.0
        )
    
    # Create experiment directory
    experiment_name = create_experiment_name(config)
    exp_dir = setup_experiment_dir(args.output_dir, experiment_name)
    
    # Save configuration
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    # Create data loaders
    data_config = config['data']
    train_dir = Path(args.data_dir) / 'train'
    val_dir = Path(args.data_dir) / 'val'
    
    # Create directories if they don't exist
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # If no data exists, generate some
    if not any(train_dir.iterdir()):
        print("No training data found, generating synthetic data...")
        generate_synthetic_data(
            output_dir=train_dir,
            num_samples=800,
            sample_rate=data_config['target_sample_rate'],
            duration=2.0
        )
    
    if not any(val_dir.iterdir()):
        print("No validation data found, generating synthetic data...")
        generate_synthetic_data(
            output_dir=val_dir,
            num_samples=200,
            sample_rate=data_config['target_sample_rate'],
            duration=2.0
        )
    
    # Create data loaders
    loaders = create_data_loaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        sample_rate=data_config['sample_rate'],
        target_sample_rate=data_config['target_sample_rate'],
        segment_length=data_config.get('segment_length'),
        augment=data_config.get('augment', True),
        noise_factor=data_config.get('noise_factor', 0.1),
        pitch_shift_range=data_config.get('pitch_shift_range', [-2.0, 2.0]),
        time_stretch_range=data_config.get('time_stretch_range', [0.9, 1.1])
    )
    
    # Create model
    model_config = config['model']
    model = create_model(model_config['type'], model_config)
    
    print(f"Created {model_config['type']} model")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    loss_config = config['loss']
    criterion = create_loss_function(loss_config['type'], **loss_config)
    
    # Create optimizer
    training_config = config['training']
    optimizer = create_optimizer(
        model,
        optimizer_type=training_config['optimizer'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 1e-4)
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=training_config.get('scheduler'),
        **training_config.get('scheduler_params', {})
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders.get('val'),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=exp_dir / 'checkpoints',
        log_interval=config.get('logging', {}).get('log_interval', 100)
    )
    
    # Resume from checkpoint if specified
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
    elif (exp_dir / 'checkpoints' / 'latest.pth').exists():
        resume_path = exp_dir / 'checkpoints' / 'latest.pth'
    
    # Train model
    trainer.train(
        num_epochs=training_config['num_epochs'],
        resume_from=resume_path
    )
    
    print(f"Training completed! Results saved to {exp_dir}")


if __name__ == '__main__':
    main()
