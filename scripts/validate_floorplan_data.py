#!/usr/bin/env python3
"""
Data validation script for floor plan fine-tuning dataset.

This script validates that the floor plan dataset is correctly organized
and ready for SAM2 fine-tuning.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np


def validate_folder_structure(base_dir):
    """Validate that the folder structure matches expected format."""
    base_path = Path(base_dir)
    images_dir = base_path / "images"
    annotations_dir = base_path / "annotations"
    
    errors = []
    warnings = []
    
    # Check if directories exist
    if not images_dir.exists():
        errors.append(f"Images directory not found: {images_dir}")
        return errors, warnings
    
    if not annotations_dir.exists():
        errors.append(f"Annotations directory not found: {annotations_dir}")
        return errors, warnings
    
    # Get all folder names from images and annotations directories
    image_folders = sorted([f.name for f in images_dir.iterdir() if f.is_dir()])
    annotation_folders = sorted([f.name for f in annotations_dir.iterdir() if f.is_dir()])
    
    if len(image_folders) == 0:
        errors.append("No folders found in images directory")
        return errors, warnings
    
    if len(annotation_folders) == 0:
        errors.append("No folders found in annotations directory")
        return errors, warnings
    
    # Check that folders match between images and annotations
    missing_in_annotations = set(image_folders) - set(annotation_folders)
    missing_in_images = set(annotation_folders) - set(image_folders)
    
    if missing_in_annotations:
        errors.append(f"Image folders without matching annotation folders: {sorted(missing_in_annotations)}")
    if missing_in_images:
        errors.append(f"Annotation folders without matching image folders: {sorted(missing_in_images)}")
    
    # Warn if no common folders
    common_folders = set(image_folders) & set(annotation_folders)
    if len(common_folders) == 0:
        errors.append("No matching folders found between images and annotations directories")
        return errors, warnings
    
    print(f"   Found {len(common_folders)} matching folder(s): {sorted(common_folders)[:5]}{'...' if len(common_folders) > 5 else ''}")
    
    # Check each common folder for required files
    for folder_name in sorted(common_folders):
        img_folder = images_dir / folder_name
        ann_folder = annotations_dir / folder_name
        
        # Check for 00000.png or 00000.jpg file in images (PNG preferred for floor plans)
        img_file = img_folder / "00000.png"
        img_file_jpg = img_folder / "00000.jpg"
        if not img_file.exists() and not img_file_jpg.exists():
            errors.append(f"Folder {folder_name}: Missing 00000.png or 00000.jpg in images")
        elif img_file_jpg.exists() and not img_file.exists():
            warnings.append(f"Folder {folder_name}: Image is .jpg (PNG is preferred for floor plans - lossless)")
        
        # Check for 00000.png file in annotations
        ann_file = ann_folder / "00000.png"
        if not ann_file.exists():
            errors.append(f"Folder {folder_name}: Missing 00000.png in annotations")
    
    return errors, warnings


def validate_mask_format(annotations_dir, folder_name):
    """Validate that mask is palettised PNG with multiple instances."""
    ann_file = Path(annotations_dir) / folder_name / "00000.png"
    
    if not ann_file.exists():
        return None, f"Mask file not found: {ann_file}"
    
    try:
        mask = Image.open(ann_file)
        
        # Check if it's a palettised image
        if mask.mode != 'P':
            return False, f"Mask is not palettised (mode: {mask.mode}), expected 'P'"
        
        # Convert to numpy array to check unique values
        mask_array = np.array(mask)
        unique_values = np.unique(mask_array)
        
        # Remove background (0)
        object_values = unique_values[unique_values != 0]
        
        if len(object_values) == 0:
            return False, "No objects found in mask (only background)"
        
        return True, f"Found {len(object_values)} object instances (pixel values: {sorted(object_values)})"
    
    except Exception as e:
        return None, f"Error reading mask: {str(e)}"


def validate_image_mask_alignment(images_dir, annotations_dir, folder_name):
    """Validate that image and mask have same dimensions."""
    img_file = Path(images_dir) / folder_name / "00000.png"
    if not img_file.exists():
        img_file = Path(images_dir) / folder_name / "00000.jpg"
    
    ann_file = Path(annotations_dir) / folder_name / "00000.png"
    
    if not img_file.exists() or not ann_file.exists():
        return None, "Image or mask file not found"
    
    try:
        img = Image.open(img_file)
        mask = Image.open(ann_file)
        
        if img.size != mask.size:
            return False, f"Size mismatch: image {img.size} vs mask {mask.size}"
        
        return True, f"Dimensions match: {img.size}"
    
    except Exception as e:
        return None, f"Error reading files: {str(e)}"


def main():
    """Main validation function."""
    if len(sys.argv) < 2:
        print("Usage: python validate_floorplan_data.py <tuning_dataset_path>")
        print("Example: python validate_floorplan_data.py ./tuning_dataset")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    
    print(f"Validating floor plan dataset at: {base_dir}")
    print("=" * 60)
    
    # Validate folder structure
    print("\n1. Validating folder structure...")
    errors, warnings = validate_folder_structure(base_dir)
    
    if errors:
        print("❌ ERRORS found:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ Folder structure is correct")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   - {warning}")
    
    if errors:
        print("\n❌ Validation failed. Please fix the errors above.")
        sys.exit(1)
    
    # Validate mask formats and alignment
    print("\n2. Validating mask formats and alignment...")
    images_dir = Path(base_dir) / "images"
    annotations_dir = Path(base_dir) / "annotations"
    
    all_valid = True
    instance_counts = []
    
    for folder_name in sorted([f.name for f in images_dir.iterdir() if f.is_dir()]):
        # Validate mask format
        is_valid, msg = validate_mask_format(annotations_dir, folder_name)
        if is_valid is None:
            print(f"   ❌ Folder {folder_name}: {msg}")
            all_valid = False
        elif not is_valid:
            print(f"   ❌ Folder {folder_name}: {msg}")
            all_valid = False
        else:
            # Extract instance count from message
            if "Found" in msg:
                count = int(msg.split("Found")[1].split("object")[0].strip())
                instance_counts.append(count)
            print(f"   ✅ Folder {folder_name}: {msg}")
        
        # Validate alignment
        is_aligned, align_msg = validate_image_mask_alignment(images_dir, annotations_dir, folder_name)
        if is_aligned is None:
            print(f"   ⚠️  Folder {folder_name}: {align_msg}")
        elif not is_aligned:
            print(f"   ❌ Folder {folder_name}: {align_msg}")
            all_valid = False
        elif is_aligned:
            print(f"   ✅ Folder {folder_name}: {align_msg}")
    
    # Summary
    print("\n" + "=" * 60)
    if all_valid and not errors:
        print("✅ All validations passed!")
        if instance_counts:
            avg_instances = np.mean(instance_counts)
            min_instances = np.min(instance_counts)
            max_instances = np.max(instance_counts)
            print(f"\n📊 Dataset statistics:")
            print(f"   - Average instances per floor plan: {avg_instances:.1f}")
            print(f"   - Min instances: {min_instances}")
            print(f"   - Max instances: {max_instances}")
            print(f"\n💡 Suggested max_num_objects for training: {int(max_instances * 1.2)}")
    else:
        print("❌ Validation failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

