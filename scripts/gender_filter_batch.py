#!/usr/bin/env python3
"""
Optimized batch gender filtering using multi-GPU and parallel processing
Processes multiple images simultaneously across both GPUs for maximum throughput
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from queue import Queue
import numpy as np
from PIL import Image
import shutil

def setup_gpu(gpu_id):
    """Configure TensorFlow to use specific GPU"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    return gpu_id

def process_batch_worker(gpu_id, input_queue, output_queue, target_gender='Woman'):
    """Worker process that handles batches on a specific GPU using concurrent threading"""
    setup_gpu(gpu_id)

    from deepface import DeepFace
    from concurrent.futures import ThreadPoolExecutor

    print(f"[GPU {gpu_id}] Worker started", flush=True)

    def process_single_image(img_path):
        """Process a single image"""
        try:
            result = DeepFace.analyze(
                str(img_path),
                actions=['gender'],
                enforce_detection=False,
                silent=True
            )

            if isinstance(result, list):
                result = result[0]

            gender = result['dominant_gender']
            return (img_path, gender == target_gender)

        except Exception:
            return (img_path, False)

    while True:
        batch = input_queue.get()
        if batch is None:  # Poison pill
            break

        batch_id, image_paths = batch

        # Process batch using thread pool (4 threads per GPU worker)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_single_image, image_paths))

        output_queue.put((gpu_id, batch_id, results))

    print(f"[GPU {gpu_id}] Worker stopped", flush=True)

def load_images_batch(image_paths):
    """Load a batch of images (currently just validates paths exist)"""
    return [p for p in image_paths if p.exists()]

def filter_faces_batch(input_dir, output_dir, target_gender='Woman', batch_size=32, num_workers=2):
    """
    Filter faces by gender using batched multi-GPU processing

    Args:
        input_dir: Directory containing face images
        output_dir: Directory to save filtered faces
        target_gender: 'Woman' or 'Man'
        batch_size: Number of images to process per batch
        num_workers: Number of GPU workers (typically 2 for dual GPU)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(input_path.glob(ext))

    total_images = len(image_files)
    print(f"Found {total_images} images to process")
    print(f"Target gender: {target_gender}")
    print(f"Batch size: {batch_size}, Workers: {num_workers}")

    if total_images == 0:
        print("No images found!")
        return

    # Create batches
    batches = []
    for i in range(0, total_images, batch_size):
        batch_files = image_files[i:i+batch_size]
        batches.append((i // batch_size, batch_files))

    print(f"Created {len(batches)} batches")

    # Setup multiprocessing queues
    input_queue = mp.Queue(maxsize=num_workers * 2)
    output_queue = mp.Queue()

    # Start GPU worker processes
    workers = []
    for gpu_id in range(num_workers):
        p = mp.Process(
            target=process_batch_worker,
            args=(gpu_id, input_queue, output_queue, target_gender)
        )
        p.start()
        workers.append(p)

    # Feed batches to workers
    print("\nProcessing batches...")

    # Start a thread to feed batches
    def feed_batches():
        for batch in batches:
            input_queue.put(batch)
        # Send poison pills
        for _ in range(num_workers):
            input_queue.put(None)

    import threading
    feeder = threading.Thread(target=feed_batches)
    feeder.start()

    # Collect results and copy matched files
    processed_batches = 0
    total_matched = 0
    total_errors = 0

    while processed_batches < len(batches):
        gpu_id, batch_id, results = output_queue.get()

        # Copy matched files
        for img_path, matched in results:
            if matched:
                try:
                    dest = output_path / img_path.name
                    shutil.copy2(img_path, dest)
                    total_matched += 1
                except Exception as e:
                    total_errors += 1

        processed_batches += 1

        # Progress update
        processed_images = min(processed_batches * batch_size, total_images)
        pct = (processed_images / total_images) * 100
        print(f"\rProgress: {processed_images}/{total_images} ({pct:.1f}%) - "
              f"Matched: {total_matched} - GPU {gpu_id}", end='', flush=True)

    print()  # New line after progress

    # Wait for workers to finish
    feeder.join()
    for w in workers:
        w.join()

    print(f"\n✓ Matched {total_matched} {target_gender} faces")
    print(f"✗ Errors: {total_errors}")
    print(f"→ Saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch filter faces by gender (multi-GPU)')
    parser.add_argument('--input', '-i', required=True, help='Input directory with face images')
    parser.add_argument('--output', '-o', required=True, help='Output directory for filtered faces')
    parser.add_argument('--gender', '-g', default='Woman', choices=['Woman', 'Man'],
                       help='Target gender to filter (default: Woman)')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size for GPU processing (default: 32)')
    parser.add_argument('--workers', '-w', type=int, default=2,
                       help='Number of GPU workers (default: 2)')

    args = parser.parse_args()

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    filter_faces_batch(
        args.input,
        args.output,
        args.gender,
        args.batch_size,
        args.workers
    )
