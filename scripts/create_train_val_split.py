#!/usr/bin/env python3
"""
Create train/validation split for floor plan fine-tuning dataset.

This script splits the 30 floor plan folders into train (24) and validation (6) sets.
"""

import os
import sys
import random
from pathlib import Path
import argparse


def create_split(base_dir, train_ratio=0.8, seed=42, output_dir=None):
    """
    Create train/val split from floor plan dataset.
    
    Args:
        base_dir: Path to tuning_dataset directory
        train_ratio: Ratio of data to use for training (default 0.8 = 24/30)
        seed: Random seed for reproducibility
        output_dir: Directory to save split files (default: base_dir)
    """
    base_path = Path(base_dir)
    images_dir = base_path / "images"
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)
    
    # Get all folder names (e.g., floorplan1, floorplan2, etc.)
    folders = sorted([f.name for f in images_dir.iterdir() if f.is_dir()])
    
    if len(folders) == 0:
        print(f"Error: No folders found in images directory")
        sys.exit(1)
    
    print(f"Found {len(folders)} folder(s) to split")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle and split
    shuffled_folders = folders.copy()
    random.shuffle(shuffled_folders)
    
    num_train = int(len(shuffled_folders) * train_ratio)
    train_folders = sorted(shuffled_folders[:num_train])
    val_folders = sorted(shuffled_folders[num_train:])
    
    # Determine output directory
    if output_dir is None:
        output_dir = base_path
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write train list
    train_file = output_dir / "train_list.txt"
    with open(train_file, 'w') as f:
        for folder in train_folders:
            f.write(f"{folder}\n")
    
    # Write val list
    val_file = output_dir / "val_list.txt"
    with open(val_file, 'w') as f:
        for folder in val_folders:
            f.write(f"{folder}\n")
    
    print(f"✅ Created train/val split:")
    print(f"   - Train: {len(train_folders)} folders -> {train_file}")
    print(f"   - Validation: {len(val_folders)} folders -> {val_file}")
    print(f"\n   Train folders: {train_folders}")
    print(f"   Val folders: {val_folders}")
    
    return train_file, val_file


def main():
    parser = argparse.ArgumentParser(
        description="Create train/validation split for floor plan dataset"
    )
    parser.add_argument(
        "base_dir",
        type=str,
        help="Path to tuning_dataset directory"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training (default: 0.8 = 24/30)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save split files (default: base_dir)"
    )
    
    args = parser.parse_args()
    
    create_split(
        args.base_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

