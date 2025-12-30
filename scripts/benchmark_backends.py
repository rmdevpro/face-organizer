#!/usr/bin/env python3
"""
Benchmark DeepFace detection backends for speed comparison
Tests both detection rate and processing speed
"""

import os
import time
from pathlib import Path
from deepface import DeepFace
import argparse

BACKENDS = ['opencv', 'ssd', 'mtcnn', 'retinaface']

def benchmark_backend(image_files, backend, target_gender='Woman'):
    """
    Benchmark a specific backend

    Returns:
        dict with timing and accuracy stats
    """
    start_time = time.time()

    detected = 0
    matched = 0
    failed = 0

    for img_path in image_files:
        try:
            result = DeepFace.analyze(
                str(img_path),
                actions=['gender'],
                detector_backend=backend,
                enforce_detection=True,
                silent=True
            )

            if isinstance(result, list):
                result = result[0]

            detected += 1
            if result['dominant_gender'] == target_gender:
                matched += 1

        except Exception:
            failed += 1

    elapsed = time.time() - start_time
    total = len(image_files)

    return {
        'backend': backend,
        'total': total,
        'detected': detected,
        'matched': matched,
        'failed': failed,
        'elapsed': elapsed,
        'images_per_sec': total / elapsed if elapsed > 0 else 0,
        'detection_rate': (detected / total * 100) if total > 0 else 0
    }

def run_benchmarks(input_dir, num_samples=50, target_gender='Woman'):
    """
    Run benchmarks on all backends
    """
    input_path = Path(input_dir)

    # Get sample images
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(list(input_path.glob(ext)))

    image_files = image_files[:num_samples]

    print(f"\n{'='*80}")
    print(f"DEEPFACE BACKEND SPEED BENCHMARK")
    print(f"{'='*80}\n")
    print(f"Testing {len(image_files)} images from: {input_dir}")
    print(f"Target gender: {target_gender}\n")
    print(f"{'='*80}\n")

    results = []

    for backend in BACKENDS:
        print(f"Testing {backend}...", flush=True)

        # Warmup run (1 image)
        try:
            DeepFace.analyze(
                str(image_files[0]),
                actions=['gender'],
                detector_backend=backend,
                enforce_detection=True,
                silent=True
            )
        except:
            pass

        # Actual benchmark
        result = benchmark_backend(image_files, backend, target_gender)
        results.append(result)

        print(f"  ✓ {result['elapsed']:.1f}s ({result['images_per_sec']:.2f} img/sec)")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}\n")
    print(f"{'Backend':<12} {'Speed':<15} {'Detection':<12} {'Matched':<10} {'Failed':<10}")
    print(f"{'-'*12} {'-'*15} {'-'*12} {'-'*10} {'-'*10}")

    for r in results:
        speed_str = f"{r['images_per_sec']:.2f} img/sec"
        detection_str = f"{r['detected']}/{r['total']} ({r['detection_rate']:.1f}%)"
        matched_str = f"{r['matched']}"
        failed_str = f"{r['failed']}"

        print(f"{r['backend']:<12} {speed_str:<15} {detection_str:<12} {matched_str:<10} {failed_str:<10}")

    # Find best by speed
    fastest = max(results, key=lambda x: x['images_per_sec'])
    best_detection = max(results, key=lambda x: x['detection_rate'])

    print(f"\n{'='*80}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*80}\n")
    print(f"Fastest: {fastest['backend']} ({fastest['images_per_sec']:.2f} img/sec)")
    print(f"Best detection rate: {best_detection['backend']} ({best_detection['detection_rate']:.1f}%)")

    # Speed comparison relative to fastest
    print(f"\n{'='*80}")
    print(f"SPEED COMPARISON (relative to fastest)")
    print(f"{'='*80}\n")

    for r in sorted(results, key=lambda x: x['images_per_sec'], reverse=True):
        relative_speed = r['images_per_sec'] / fastest['images_per_sec'] * 100
        slowdown = fastest['images_per_sec'] / r['images_per_sec'] if r['images_per_sec'] > 0 else 0
        print(f"{r['backend']:<12}: {relative_speed:6.1f}% speed ({slowdown:.2f}x slower)")

    # Estimate total processing time for full dataset
    print(f"\n{'='*80}")
    print(f"ESTIMATED TIME FOR 579,931 IMAGES")
    print(f"{'='*80}\n")

    total_images = 579931
    for r in sorted(results, key=lambda x: x['images_per_sec'], reverse=True):
        if r['images_per_sec'] > 0:
            total_seconds = total_images / r['images_per_sec']
            hours = total_seconds / 3600
            print(f"{r['backend']:<12}: {hours:6.1f} hours ({hours/24:.1f} days)")

    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark DeepFace detection backends')
    parser.add_argument('--input', '-i', required=True,
                       help='Input directory with face images')
    parser.add_argument('--samples', '-n', type=int, default=50,
                       help='Number of sample images to benchmark (default: 50)')
    parser.add_argument('--gender', '-g', default='Woman',
                       choices=['Woman', 'Man'],
                       help='Target gender (default: Woman)')

    args = parser.parse_args()

    run_benchmarks(args.input, args.samples, args.gender)
