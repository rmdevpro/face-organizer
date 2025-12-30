#!/usr/bin/env python3
"""
Diagnostic script to understand DeepFace gender classification failures
Tests multiple detection backends and shows detailed analysis results
"""

import os
import sys
import argparse
from pathlib import Path
from deepface import DeepFace
import json

# Available detection backends in DeepFace
BACKENDS = ['opencv', 'ssd', 'mtcnn', 'retinaface', 'mediapipe']

def test_single_image(img_path, backend='opencv', enforce_detection=True):
    """
    Test a single image with specified backend

    Args:
        img_path: Path to image file
        backend: Detection backend to use
        enforce_detection: Whether to require face detection

    Returns:
        dict with analysis results or error info
    """
    try:
        result = DeepFace.analyze(
            str(img_path),
            actions=['gender'],
            detector_backend=backend,
            enforce_detection=enforce_detection,
            silent=True
        )

        # Handle both single result and list
        if isinstance(result, list):
            result = result[0]

        return {
            'success': True,
            'backend': backend,
            'enforce_detection': enforce_detection,
            'dominant_gender': result.get('dominant_gender'),
            'gender_confidence': result.get('gender', {}),
            'region': result.get('region'),
            'face_confidence': result.get('face_confidence', 'N/A')
        }

    except Exception as e:
        return {
            'success': False,
            'backend': backend,
            'enforce_detection': enforce_detection,
            'error': str(e)
        }

def diagnose_images(input_dir, num_samples=10, backends=None):
    """
    Test a sample of images with different backends

    Args:
        input_dir: Directory containing face images
        num_samples: Number of images to test
        backends: List of backends to test (default: all)
    """
    if backends is None:
        backends = BACKENDS

    input_path = Path(input_dir)

    # Get sample images
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(list(input_path.glob(ext))[:num_samples])

    image_files = image_files[:num_samples]

    print(f"\n{'='*80}")
    print(f"DEEPFACE GENDER CLASSIFICATION DIAGNOSTIC")
    print(f"{'='*80}\n")
    print(f"Testing {len(image_files)} images from: {input_dir}")
    print(f"Detection backends: {', '.join(backends)}\n")

    # Test each image with each backend
    for img_file in image_files:
        print(f"\n{'-'*80}")
        print(f"IMAGE: {img_file.name}")
        print(f"{'-'*80}\n")

        for backend in backends:
            print(f"  Backend: {backend:12s} | ", end='')

            # Test with enforce_detection=True
            result_true = test_single_image(img_file, backend, enforce_detection=True)

            if result_true['success']:
                gender = result_true['dominant_gender']
                confidence = result_true['gender_confidence'].get(gender, 0)
                print(f"✓ {gender:5s} ({confidence:.1f}%)", end='')

                # Show face detection confidence if available
                if result_true['face_confidence'] != 'N/A':
                    print(f" [face: {result_true['face_confidence']:.2f}]", end='')
                print()
            else:
                error_msg = result_true['error']
                if 'Face could not be detected' in error_msg:
                    print(f"✗ NO FACE DETECTED")
                else:
                    print(f"✗ ERROR: {error_msg[:50]}")

        print()

    # Summary statistics
    print(f"\n{'='*80}")
    print(f"BACKEND COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    backend_stats = {b: {'total': 0, 'detected': 0, 'woman': 0, 'man': 0} for b in backends}

    for img_file in image_files:
        for backend in backends:
            backend_stats[backend]['total'] += 1
            result = test_single_image(img_file, backend, enforce_detection=True)

            if result['success']:
                backend_stats[backend]['detected'] += 1
                gender = result['dominant_gender']
                if gender == 'Woman':
                    backend_stats[backend]['woman'] += 1
                elif gender == 'Man':
                    backend_stats[backend]['man'] += 1

    for backend in backends:
        stats = backend_stats[backend]
        detect_rate = (stats['detected'] / stats['total'] * 100) if stats['total'] > 0 else 0
        woman_pct = (stats['woman'] / stats['detected'] * 100) if stats['detected'] > 0 else 0
        man_pct = (stats['man'] / stats['detected'] * 100) if stats['detected'] > 0 else 0

        print(f"{backend:12s}: {stats['detected']:2d}/{stats['total']:2d} detected ({detect_rate:5.1f}%) | "
              f"Woman: {stats['woman']:2d} ({woman_pct:5.1f}%) | "
              f"Man: {stats['man']:2d} ({man_pct:5.1f}%)")

    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnose DeepFace gender classification issues')
    parser.add_argument('--input', '-i', required=True,
                       help='Input directory with face images to test')
    parser.add_argument('--samples', '-n', type=int, default=10,
                       help='Number of sample images to test (default: 10)')
    parser.add_argument('--backends', '-b', nargs='+', choices=BACKENDS,
                       help='Backends to test (default: all)')

    args = parser.parse_args()

    diagnose_images(args.input, args.samples, args.backends)
