#!/usr/bin/env python3
"""
Setup script for Face Generation GAN
"""

import os
import sys

def main():
    print("="*80)
    print("FACE GENERATION GAN - SETUP")
    print("="*80)
    
    # Create directories
    directories = [
        "data",
        "generated_celeba",
        "generated_transfer",
        "notebooks",
        "src/models",
        "src/data",
        "src/training",
        "src/visualization"
    ]
    
    print("\nCreating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}/")
    
    # Create empty __init__.py files
    init_files = [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/data/__init__.py",
        "src/training/__init__.py",
        "src/visualization/__init__.py"
    ]
    
    for file in init_files:
        with open(file, 'w') as f:
            f.write('')
    
    print("\n" + "="*80)
    print("SETUP COMPLETE!")
    print("="*80)
    print("\n⚠️  IMPORTANT NEXT STEPS:")
    print("\n1. DOWNLOAD DATASETS:")
    print("   - CelebA: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")
    print("   - Face Mask: https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset")
    print("\n2. Extract datasets to:")
    print("   - data/celeba/img_align_celeba/")
    print("   - data/mask/without_mask/")
    print("\n3. Update paths in scripts:")
    print("   - scripts/train_celeba.py: Update CELEBA_DATA_DIR")
    print("   - scripts/train_transfer.py: Update MASK_DATA_DIR")
    print("\n4. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n5. Run training:")
    print("   python scripts/train_celeba.py")
    print("   python scripts/train_transfer.py")
    print("="*80)

if __name__ == "__main__":
    main()