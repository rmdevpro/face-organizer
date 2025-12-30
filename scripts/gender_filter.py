#!/usr/bin/env python3
"""
Gender-based face filtering using DeepFace
Filters faces by gender and saves to output directory
"""

import os
import sys
import argparse
from pathlib import Path
from deepface import DeepFace
from tqdm import tqdm
import shutil

def filter_faces_by_gender(input_dir, output_dir, target_gender='Woman', save_rejected=True):
    """
    Filter faces by gender classification

    Args:
        input_dir: Directory containing face images
        output_dir: Directory to save filtered faces
        target_gender: 'Woman' or 'Man' (default: Woman)
        save_rejected: If True, save rejected images to separate folder
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create rejected folder if needed
    if save_rejected:
        rejected_path = output_path.parent / f"{output_path.name}_rejected"
        rejected_path.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_files = list(input_path.glob('*.png')) + \
                  list(input_path.glob('*.jpg')) + \
                  list(input_path.glob('*.jpeg'))

    print(f"Found {len(image_files)} images to process")
    print(f"Filtering for: {target_gender}")
    print(f"Save rejected: {save_rejected}")

    matched = 0
    rejected = 0
    errors = 0

    for img_file in tqdm(image_files, desc="Classifying"):
        try:
            # Analyze gender
            result = DeepFace.analyze(
                str(img_file),
                actions=['gender'],
                detector_backend='ssd',  # SSD: 3x faster + better accuracy than opencv
                enforce_detection=True,  # MUST detect a face to classify
                silent=True
            )

            # Handle both single result and list of results
            if isinstance(result, list):
                result = result[0]

            detected_gender = result['dominant_gender']

            if detected_gender == target_gender:
                # Copy to matched output directory
                shutil.copy2(img_file, output_path / img_file.name)
                matched += 1
            else:
                # Copy to rejected directory
                if save_rejected:
                    shutil.copy2(img_file, rejected_path / img_file.name)
                    rejected += 1

        except Exception as e:
            # Face not detected or other error - save to rejected
            if save_rejected:
                shutil.copy2(img_file, rejected_path / img_file.name)
            errors += 1
            if errors <= 5:  # Only print first 5 errors
                print(f"\nError processing {img_file.name}: {e}")

    print(f"\n✓ Matched {matched} {target_gender} faces")
    print(f"✗ Rejected {rejected} faces")
    print(f"✗ Errors (no face detected): {errors}")
    print(f"→ Matched saved to: {output_path}")
    if save_rejected:
        print(f"→ Rejected saved to: {rejected_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter faces by gender')
    parser.add_argument('--input', '-i', required=True, help='Input directory with face images')
    parser.add_argument('--output', '-o', required=True, help='Output directory for filtered faces')
    parser.add_argument('--gender', '-g', default='Woman', choices=['Woman', 'Man'],
                       help='Target gender to filter (default: Woman)')
    parser.add_argument('--save-rejected', action='store_true', default=True,
                       help='Save rejected images to separate folder (default: True)')

    args = parser.parse_args()

    filter_faces_by_gender(args.input, args.output, args.gender, args.save_rejected)
