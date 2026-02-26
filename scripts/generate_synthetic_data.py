#!/usr/bin/env python3
"""
Script to generate synthetic audio data for training and testing.

This script creates synthetic audio data with various characteristics
to train and evaluate audio super-resolution models.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from typing import List, Tuple


def generate_sine_wave_mixture(
    duration: float,
    sample_rate: int,
    num_components: int = 3,
    freq_range: Tuple[float, float] = (100, 2000),
    amp_range: Tuple[float, float] = (0.1, 0.8)
) -> Tuple[np.ndarray, List[float], List[float]]:
    """
    Generate a mixture of sine waves.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate
        num_components: Number of sine wave components
        freq_range: Frequency range (min, max) Hz
        amp_range: Amplitude range (min, max)
        
    Returns:
        Audio signal, frequencies, amplitudes
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    frequencies = np.random.uniform(freq_range[0], freq_range[1], num_components)
    amplitudes = np.random.uniform(amp_range[0], amp_range[1], num_components)
    
    audio = np.zeros_like(t)
    for freq, amp in zip(frequencies, amplitudes):
        audio += amp * np.sin(2 * np.pi * freq * t)
    
    return audio, frequencies.tolist(), amplitudes.tolist()


def generate_chirp_signal(
    duration: float,
    sample_rate: int,
    start_freq: float = 100,
    end_freq: float = 2000,
    amplitude: float = 0.5
) -> np.ndarray:
    """
    Generate a chirp signal (frequency sweep).
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate
        start_freq: Starting frequency
        end_freq: Ending frequency
        amplitude: Signal amplitude
        
    Returns:
        Chirp signal
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Linear frequency sweep
    freq_sweep = start_freq + (end_freq - start_freq) * t / duration
    
    # Generate chirp
    audio = amplitude * np.sin(2 * np.pi * freq_sweep * t)
    
    return audio


def generate_noise_signal(
    duration: float,
    sample_rate: int,
    noise_type: str = 'white',
    amplitude: float = 0.1
) -> np.ndarray:
    """
    Generate noise signal.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate
        noise_type: Type of noise ('white', 'pink', 'brown')
        amplitude: Noise amplitude
        
    Returns:
        Noise signal
    """
    length = int(sample_rate * duration)
    
    if noise_type == 'white':
        audio = np.random.normal(0, amplitude, length)
    elif noise_type == 'pink':
        # Simplified pink noise (1/f)
        freqs = np.fft.fftfreq(length, 1/sample_rate)
        freqs[0] = 1  # Avoid division by zero
        pink_filter = 1 / np.sqrt(np.abs(freqs))
        pink_filter[0] = 0
        
        white_noise = np.random.normal(0, 1, length)
        audio = np.real(np.fft.ifft(np.fft.fft(white_noise) * pink_filter))
        audio = amplitude * audio / np.max(np.abs(audio))
    elif noise_type == 'brown':
        # Brown noise (1/f^2)
        freqs = np.fft.fftfreq(length, 1/sample_rate)
        freqs[0] = 1
        brown_filter = 1 / np.abs(freqs)
        brown_filter[0] = 0
        
        white_noise = np.random.normal(0, 1, length)
        audio = np.real(np.fft.ifft(np.fft.fft(white_noise) * brown_filter))
        audio = amplitude * audio / np.max(np.abs(audio))
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return audio


def generate_musical_notes(
    duration: float,
    sample_rate: int,
    notes: List[str] = None,
    amplitudes: List[float] = None
) -> np.ndarray:
    """
    Generate musical notes.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate
        notes: List of note names (e.g., ['C4', 'E4', 'G4'])
        amplitudes: List of amplitudes for each note
        
    Returns:
        Musical chord signal
    """
    if notes is None:
        notes = ['C4', 'E4', 'G4']  # C major chord
    if amplitudes is None:
        amplitudes = [0.3] * len(notes)
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t)
    
    for note, amp in zip(notes, amplitudes):
        freq = librosa.note_to_hz(note)
        audio += amp * np.sin(2 * np.pi * freq * t)
    
    return audio


def generate_synthetic_sample(
    sample_type: str,
    duration: float,
    sample_rate: int,
    **kwargs
) -> Tuple[np.ndarray, dict]:
    """
    Generate a synthetic audio sample.
    
    Args:
        sample_type: Type of sample to generate
        duration: Duration in seconds
        sample_rate: Sample rate
        **kwargs: Additional parameters
        
    Returns:
        Audio signal and metadata
    """
    metadata = {
        'type': sample_type,
        'duration': duration,
        'sample_rate': sample_rate
    }
    
    if sample_type == 'sine_mixture':
        audio, freqs, amps = generate_sine_wave_mixture(
            duration, sample_rate, **kwargs
        )
        metadata.update({
            'frequencies': freqs,
            'amplitudes': amps,
            'num_components': len(freqs)
        })
    
    elif sample_type == 'chirp':
        audio = generate_chirp_signal(duration, sample_rate, **kwargs)
        metadata.update({
            'start_freq': kwargs.get('start_freq', 100),
            'end_freq': kwargs.get('end_freq', 2000)
        })
    
    elif sample_type == 'noise':
        audio = generate_noise_signal(duration, sample_rate, **kwargs)
        metadata.update({
            'noise_type': kwargs.get('noise_type', 'white'),
            'amplitude': kwargs.get('amplitude', 0.1)
        })
    
    elif sample_type == 'musical':
        audio = generate_musical_notes(duration, sample_rate, **kwargs)
        metadata.update({
            'notes': kwargs.get('notes', ['C4', 'E4', 'G4']),
            'amplitudes': kwargs.get('amplitudes', [0.3, 0.3, 0.3])
        })
    
    else:
        raise ValueError(f"Unknown sample type: {sample_type}")
    
    # Normalize audio
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # Add slight noise
    noise_level = kwargs.get('noise_level', 0.01)
    noise = np.random.normal(0, noise_level, len(audio))
    audio += noise
    
    return audio, metadata


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate synthetic audio data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--duration', type=float, default=2.0, help='Duration per sample')
    parser.add_argument('--sample_rate', type=int, default=32000, help='Sample rate')
    parser.add_argument('--types', nargs='+', default=['sine_mixture', 'chirp', 'noise', 'musical'],
                       help='Types of samples to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    metadata_list = []
    
    for i in range(args.num_samples):
        # Randomly select sample type
        sample_type = np.random.choice(args.types)
        
        # Generate sample
        audio, metadata = generate_synthetic_sample(
            sample_type=sample_type,
            duration=args.duration,
            sample_rate=args.sample_rate,
            num_components=np.random.randint(2, 6),
            freq_range=(50, 3000),
            amp_range=(0.1, 0.8),
            noise_level=0.01
        )
        
        # Save audio file
        filename = f"synthetic_{i:04d}.wav"
        filepath = output_dir / filename
        sf.write(str(filepath), audio, args.sample_rate)
        
        # Add metadata
        metadata['id'] = i
        metadata['filename'] = filename
        metadata_list.append(metadata)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{args.num_samples} samples")
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(output_dir / 'metadata.csv', index=False)
    
    # Print summary
    print(f"\nGenerated {args.num_samples} synthetic audio samples")
    print(f"Output directory: {output_dir}")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Duration: {args.duration} seconds per sample")
    
    # Print type distribution
    type_counts = metadata_df['type'].value_counts()
    print("\nSample type distribution:")
    for sample_type, count in type_counts.items():
        print(f"  {sample_type}: {count} samples")


if __name__ == '__main__':
    main()
