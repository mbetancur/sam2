#!/usr/bin/env python3
"""
Convert mask images to palettised (P) mode for SAM2 training.

This script converts all mask images in the annotations directory to palettised PNG format,
which is required for SAM2 fine-tuning with multi-instance masks.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import argparse
import shutil
from datetime import datetime


def convert_mask_to_palettised(mask_path, backup=True, backup_dir=None):
    """
    Convert a mask image to palettised (P) mode.
    
    Args:
        mask_path: Path to the mask image
        backup: Whether to create a backup before converting
        backup_dir: Directory to save backups (default: same directory with _backup suffix)
    
    Returns:
        tuple: (success: bool, message: str)
    """
    mask_path = Path(mask_path)
    
    if not mask_path.exists():
        return False, f"File not found: {mask_path}"
    
    try:
        # Open the mask
        mask = Image.open(mask_path)
        
        # Check if already palettised
        if mask.mode == 'P':
            return True, f"Already palettised (mode: P)"
        
        # Create backup if requested
        if backup:
            if backup_dir is None:
                backup_dir = mask_path.parent / f"{mask_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{mask_path.suffix}"
            else:
                backup_dir = Path(backup_dir) / mask_path.name
                backup_dir.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(mask_path, backup_dir)
            backup_msg = f" (backup: {backup_dir.name})"
        else:
            backup_msg = ""
        
        # Convert to palettised mode
        # For RGB/RGBA, use quantize to create a palette
        # For grayscale, convert directly
        if mask.mode in ['RGB', 'RGBA']:
            # Convert RGBA to RGB first if needed
            if mask.mode == 'RGBA':
                # Create a white background
                rgb_mask = Image.new('RGB', mask.size, (255, 255, 255))
                rgb_mask.paste(mask, mask=mask.split()[3])  # Use alpha channel as mask
                mask = rgb_mask
            
            # Quantize to create a palette (max 256 colors)
            # This preserves distinct colors/regions
            try:
                # Try new PIL API first (Pillow 10.0+)
                mask = mask.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
            except (AttributeError, TypeError):
                # Fall back to older API
                mask = mask.quantize(colors=256)
        elif mask.mode == 'L':
            # Grayscale - convert directly to P mode
            mask = mask.convert('P')
        else:
            # Try to convert through RGB first
            if mask.mode not in ['1', 'P']:
                try:
                    mask = mask.convert('RGB').quantize(colors=256, method=Image.Quantize.MEDIANCUT)
                except (AttributeError, TypeError):
                    mask = mask.convert('RGB').quantize(colors=256)
            else:
                return False, f"Cannot convert mode {mask.mode} to palettised"
        
        # Save the converted mask
        mask.save(mask_path, format='PNG')
        
        # Verify the conversion
        verify_mask = Image.open(mask_path)
        if verify_mask.mode == 'P':
            return True, f"Converted from {mask.mode} to P mode{backup_msg}"
        else:
            return False, f"Conversion failed - still mode {verify_mask.mode}"
    
    except Exception as e:
        return False, f"Error converting: {str(e)}"


def convert_all_masks(base_dir, backup=True, backup_dir=None, dry_run=False):
    """
    Convert all mask images in the annotations directory to palettised mode.
    
    Args:
        base_dir: Path to tuning_dataset directory
        backup: Whether to create backups
        backup_dir: Directory for backups (default: base_dir/annotations_backup)
        dry_run: If True, only report what would be converted without actually converting
    """
    base_path = Path(base_dir)
    annotations_dir = base_path / "annotations"
    
    if not annotations_dir.exists():
        print(f"❌ Annotations directory not found: {annotations_dir}")
        sys.exit(1)
    
    # Set up backup directory
    if backup and backup_dir is None:
        backup_dir = base_path / "annotations_backup"
    
    if backup and not dry_run:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"📦 Backups will be saved to: {backup_dir}")
    
    # Find all mask files
    mask_files = []
    for folder in sorted(annotations_dir.iterdir()):
        if folder.is_dir():
            mask_file = folder / "00000.png"
            if mask_file.exists():
                mask_files.append(mask_file)
    
    if len(mask_files) == 0:
        print(f"❌ No mask files found in {annotations_dir}")
        sys.exit(1)
    
    print(f"Found {len(mask_files)} mask file(s) to process")
    print("=" * 60)
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be modified\n")
    
    converted_count = 0
    already_palettised = 0
    failed_count = 0
    skipped_count = 0
    
    for mask_file in mask_files:
        folder_name = mask_file.parent.name
        print(f"\nProcessing: {folder_name}/00000.png")
        
        if dry_run:
            # Just check the mode
            try:
                mask = Image.open(mask_file)
                if mask.mode == 'P':
                    print(f"   ✅ Already palettised (would skip)")
                    already_palettised += 1
                else:
                    print(f"   ⚠️  Would convert from {mask.mode} to P mode")
                    converted_count += 1
            except Exception as e:
                print(f"   ❌ Error reading: {str(e)}")
                failed_count += 1
        else:
            # Actually convert
            success, message = convert_mask_to_palettised(
                mask_file,
                backup=backup,
                backup_dir=backup_dir / folder_name if backup else None
            )
            
            if success:
                if "Already" in message:
                    print(f"   ✅ {message}")
                    already_palettised += 1
                else:
                    print(f"   ✅ {message}")
                    converted_count += 1
            else:
                print(f"   ❌ {message}")
                failed_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Conversion Summary:")
    print(f"   ✅ Converted: {converted_count}")
    print(f"   ✓ Already palettised: {already_palettised}")
    if failed_count > 0:
        print(f"   ❌ Failed: {failed_count}")
    print(f"   📁 Total processed: {len(mask_files)}")
    
    if dry_run:
        print("\n💡 Run without --dry-run to actually convert the files")
    elif converted_count > 0:
        print(f"\n✅ Successfully converted {converted_count} mask(s) to palettised mode!")
        if backup:
            print(f"📦 Original files backed up to: {backup_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert mask images to palettised (P) mode for SAM2 training"
    )
    parser.add_argument(
        "base_dir",
        type=str,
        help="Path to tuning_dataset directory"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files before converting"
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=None,
        help="Directory to save backups (default: base_dir/annotations_backup)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without actually converting"
    )
    
    args = parser.parse_args()
    
    convert_all_masks(
        args.base_dir,
        backup=not args.no_backup,
        backup_dir=args.backup_dir,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

