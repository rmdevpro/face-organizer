#!/usr/bin/env python3
"""
Optimized batched gender filtering with prefetching pipeline
- Loads model once at startup
- True batch inference (64 images at once)
- Prefetches next batch while GPU processes current batch
- Parallel I/O using thread pool
"""

import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import shutil
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

# Set GPU before TensorFlow imports
GPU_ID = int(os.environ.get('GPU_ID', '0'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

import tensorflow as tf
# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

from deepface.models.demography import Gender

print(f"GPU {GPU_ID}: Loading gender model...")
gender_model = Gender.load_model()
print(f"GPU {GPU_ID}: Model loaded!")

def preprocess_image(img_path):
    """Load and preprocess a single image"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None

        # Resize to 224x224
        img = cv2.resize(img, (224, 224))

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to float and normalize (VGGFace style)
        img = img.astype(np.float32)
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863

        return img

    except Exception:
        return None

def load_batch(image_paths, executor):
    """Load and preprocess a batch using parallel threads"""
    # Use thread pool to load images in parallel
    images = list(executor.map(preprocess_image, image_paths))

    # Filter out failed loads and build arrays
    batch_images = []
    valid_paths = []

    for img_path, img in zip(image_paths, images):
        if img is not None:
            batch_images.append(img)
            valid_paths.append(img_path)

    if batch_images:
        return np.array(batch_images), valid_paths
    return None, []

def prefetch_batches(batch_list, prefetch_queue, executor, max_prefetch=3):
    """Background thread that prefetches and preprocesses batches"""
    for batch_paths in batch_list:
        # Load batch using parallel I/O
        batch_array, valid_paths = load_batch(batch_paths, executor)
        # Put in queue (blocks if queue is full - backpressure)
        prefetch_queue.put((batch_array, valid_paths))

    # Signal end
    prefetch_queue.put(None)

def filter_faces(input_dir, output_dir, target_gender='Woman', batch_size=64, prefetch_size=3, io_threads=8):
    """
    Filter faces by gender using batched processing with prefetching pipeline

    Args:
        input_dir: Input directory
        output_dir: Output directory
        target_gender: 'Woman' or 'Man'
        batch_size: Images per GPU batch (64 optimal for P4)
        prefetch_size: Number of batches to prefetch (3 = good overlap)
        io_threads: Threads for parallel image loading (8 = good for SSD)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(input_path.glob(ext))

    total = len(image_files)
    print(f"\nGPU {GPU_ID}: Processing {total} images")
    print(f"GPU {GPU_ID}: Batch size: {batch_size}, Prefetch: {prefetch_size} batches, I/O threads: {io_threads}")

    target_idx = 0 if target_gender == 'Woman' else 1

    # Create batches
    batch_list = [image_files[i:i+batch_size] for i in range(0, total, batch_size)]
    print(f"GPU {GPU_ID}: Created {len(batch_list)} batches\n")

    # Setup prefetch pipeline
    prefetch_queue = Queue(maxsize=prefetch_size)
    io_executor = ThreadPoolExecutor(max_workers=io_threads)

    # Start prefetch thread
    prefetch_thread = Thread(
        target=prefetch_batches,
        args=(batch_list, prefetch_queue, io_executor),
        daemon=True
    )
    prefetch_thread.start()

    matched = 0
    processed = 0

    # Main loop: GPU inference on prefetched batches
    while True:
        # Get next prefetched batch (loaded in background while GPU was busy)
        item = prefetch_queue.get()
        if item is None:
            break

        batch_array, valid_paths = item

        if batch_array is not None:
            # GPU INFERENCE on batch
            predictions = gender_model.predict(batch_array, verbose=0)

            # Copy matched files
            for img_path, pred in zip(valid_paths, predictions):
                if np.argmax(pred) == target_idx:
                    try:
                        dest = output_path / img_path.name
                        shutil.copy2(img_path, dest)
                        matched += 1
                    except Exception:
                        pass

        processed += len(valid_paths) if valid_paths else 0

        # Progress
        pct = (processed / total) * 100
        print(f"\rGPU {GPU_ID}: {processed}/{total} ({pct:.1f}%) | Matched: {matched}",
              end='', flush=True)

    # Cleanup
    io_executor.shutdown(wait=True)
    prefetch_thread.join()

    print(f"\n\nGPU {GPU_ID}: ✓ Matched {matched} {target_gender} faces")
    print(f"GPU {GPU_ID}: → Output: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimized batched gender filtering with prefetching')
    parser.add_argument('--input', '-i', required=True, help='Input directory')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--gender', '-g', default='Woman', choices=['Woman', 'Man'],
                       help='Target gender (default: Woman)')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                       help='GPU batch size (default: 64)')
    parser.add_argument('--prefetch', '-p', type=int, default=3,
                       help='Number of batches to prefetch (default: 3)')
    parser.add_argument('--io-threads', '-t', type=int, default=8,
                       help='I/O threads for loading (default: 8)')

    args = parser.parse_args()

    filter_faces(args.input, args.output, args.gender, args.batch_size, args.prefetch, args.io_threads)
