#!/usr/bin/env python3
"""
Quick start script for Audio Super-Resolution project.

This script provides a simple way to get started with the project
by generating synthetic data and training a basic model.
"""

import argparse
import sys
from pathlib import Path
import subprocess
import yaml

def main():
    """Main quick start function."""
    parser = argparse.ArgumentParser(description='Quick start for Audio Super-Resolution')
    parser.add_argument('--model', type=str, default='srcnn', 
                       choices=['srcnn', 'unet', 'wavenet'],
                       help='Model to train')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of synthetic samples to generate')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    
    args = parser.parse_args()
    
    print("🎵 Audio Super-Resolution Quick Start")
    print("=" * 50)
    
    # Create directories
    data_dir = Path(args.data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    
    print(f"📁 Creating directories...")
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    print(f"🎼 Generating {args.num_samples} synthetic samples...")
    
    # Generate training data
    subprocess.run([
        sys.executable, 'scripts/generate_synthetic_data.py',
        '--output_dir', str(train_dir),
        '--num_samples', str(int(args.num_samples * 0.8)),
        '--duration', '2.0',
        '--sample_rate', '32000'
    ], check=True)
    
    # Generate validation data
    subprocess.run([
        sys.executable, 'scripts/generate_synthetic_data.py',
        '--output_dir', str(val_dir),
        '--num_samples', str(int(args.num_samples * 0.2)),
        '--duration', '2.0',
        '--sample_rate', '32000'
    ], check=True)
    
    print("✅ Data generation completed!")
    
    # Train model
    print(f"🚀 Training {args.model.upper()} model for {args.epochs} epochs...")
    
    # Modify config for quick training
    config_path = Path(f'configs/{args.model}.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update training parameters
    config['training']['num_epochs'] = args.epochs
    config['training']['learning_rate'] = 0.001
    
    # Save modified config
    quick_config_path = Path(f'configs/{args.model}_quick.yaml')
    with open(quick_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    # Train model
    subprocess.run([
        sys.executable, 'scripts/train.py',
        '--config', str(quick_config_path),
        '--data_dir', str(data_dir),
        '--output_dir', 'experiments',
        '--generate_data'  # This will be ignored since data already exists
    ], check=True)
    
    print("✅ Training completed!")
    
    # Run evaluation
    print("📊 Running evaluation...")
    
    # Find the latest checkpoint
    exp_dir = Path('experiments')
    checkpoint_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
    if checkpoint_dirs:
        latest_exp = max(checkpoint_dirs, key=lambda x: x.stat().st_mtime)
        checkpoint_path = latest_exp / 'checkpoints' / 'best.pth'
        
        if checkpoint_path.exists():
            subprocess.run([
                sys.executable, 'scripts/evaluate.py',
                '--config', str(quick_config_path),
                '--checkpoint', str(checkpoint_path),
                '--data_dir', str(data_dir),
                '--output_dir', 'results'
            ], check=True)
            
            print("✅ Evaluation completed!")
        else:
            print("⚠️  No checkpoint found for evaluation")
    else:
        print("⚠️  No experiment directory found")
    
    # Clean up
    quick_config_path.unlink()
    
    print("\n🎉 Quick start completed!")
    print("\nNext steps:")
    print("1. Check the 'experiments' directory for training results")
    print("2. Check the 'results' directory for evaluation results")
    print("3. Run the demo: streamlit run demo/app.py")
    print("4. Explore the model cards: MODEL_CARDS.md")
    print("5. Read the privacy disclaimer: PRIVACY_DISCLAIMER.md")
    
    print(f"\n📚 Documentation:")
    print("- README.md: Project overview and setup")
    print("- MODEL_CARDS.md: Detailed model information")
    print("- PRIVACY_DISCLAIMER.md: Important ethical guidelines")


if __name__ == '__main__':
    main()
