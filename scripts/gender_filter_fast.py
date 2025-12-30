#!/usr/bin/env python3
"""
Fast gender filtering - uses DeepFace.analyze() with optimized prefetching
Since true batching hits circular imports, we optimize I/O and use analyze() API
"""

import os
import argparse
from pathlib import Path
import shutil
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set GPU before imports
GPU_ID = int(os.environ.get('GPU_ID', '0'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

from deepface import DeepFace

print(f"GPU {GPU_ID}: DeepFace loaded")

def analyze_single(img_path, target_gender):
    """Analyze one image"""
    try:
        result = DeepFace.analyze(
            str(img_path),
            actions=['gender'],
            detector_backend='ssd',  # SSD: 3x faster + better accuracy than opencv
            enforce_detection=True,
            silent=True
        )

        if isinstance(result, list):
            result = result[0]

        if result['dominant_gender'] == target_gender:
            return (img_path, True)
        return (img_path, False)

    except Exception:
        return (img_path, False)

def filter_faces(input_dir, output_dir, target_gender='Woman', workers=16):
    """
    Filter faces using parallel processing

    Args:
        input_dir: Input directory
        output_dir: Output directory
        target_gender: 'Woman' or 'Man'
        workers: Number of parallel workers (16 = good for dual GPU)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(input_path.glob(ext))

    total = len(image_files)
    print(f"\nGPU {GPU_ID}: Processing {total} images with {workers} workers\n")

    matched = 0
    processed = 0

    # Process with thread pool
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        futures = {executor.submit(analyze_single, img_path, target_gender): img_path
                   for img_path in image_files}

        # Process results as they complete
        for future in as_completed(futures):
            img_path, is_match = future.result()

            if is_match:
                try:
                    dest = output_path / img_path.name
                    shutil.copy2(img_path, dest)
                    matched += 1
                except Exception:
                    pass

            processed += 1

            # Progress every 100 images
            if processed % 100 == 0:
                pct = (processed / total) * 100
                print(f"\rGPU {GPU_ID}: {processed}/{total} ({pct:.1f}%) | Matched: {matched}",
                      end='', flush=True)

    print(f"\n\nGPU {GPU_ID}: ✓ Matched {matched} {target_gender} faces")
    print(f"GPU {GPU_ID}: → Output: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast parallel gender filtering')
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--gender', '-g', default='Woman', choices=['Woman', 'Man'])
    parser.add_argument('--workers', '-w', type=int, default=16,
                       help='Parallel workers (default: 16)')

    args = parser.parse_args()

    filter_faces(args.input, args.output, args.gender, args.workers)
