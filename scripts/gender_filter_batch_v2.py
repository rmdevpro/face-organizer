#!/usr/bin/env python3
"""
True batched GPU inference for gender classification
Uses direct model access for maximum throughput
"""

import os
import sys
import argparse
from pathlib import Path
import multiprocessing as mp
import numpy as np
import cv2
import shutil
from queue import Queue
import threading

def setup_gpu(gpu_id):
    """Configure TensorFlow to use specific GPU"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    return gpu_id

def load_and_preprocess_batch(image_paths):
    """Load and preprocess a batch of images for VGGFace gender model"""
    batch_images = []
    valid_paths = []

    for img_path in image_paths:
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Resize to 224x224 (VGGFace input size)
            img = cv2.resize(img, (224, 224))

            # BGR to RGB (OpenCV loads as BGR, model expects RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert to float32
            img = img.astype(np.float32)

            # VGGFace normalization (mean subtraction per channel)
            img[..., 0] -= 93.5940
            img[..., 1] -= 104.7624
            img[..., 2] -= 129.1863

            batch_images.append(img)
            valid_paths.append(img_path)

        except Exception as e:
            continue

    if batch_images:
        return np.array(batch_images), valid_paths
    return None, []

def gpu_worker(gpu_id, input_queue, output_queue, target_gender='Woman', model_weights_path=None):
    """GPU worker that processes batches with true batch inference"""
    setup_gpu(gpu_id)

    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation

    print(f"[GPU {gpu_id}] Building gender model from scratch...", flush=True)

    # Build VGGFace base model architecture
    from deepface.models.facial_recognition.VGGFace import base_model as build_vggface_base
    vgg_base = build_vggface_base()

    # Add gender classification head
    classes = 2
    gender_head = Convolution2D(classes, (1, 1), name="predictions")(vgg_base.layers[-4].output)
    gender_head = Flatten()(gender_head)
    gender_head = Activation("softmax")(gender_head)

    gender_model = Model(inputs=vgg_base.inputs, outputs=gender_head)

    # Load weights
    if model_weights_path and os.path.exists(model_weights_path):
        gender_model.load_weights(model_weights_path)
        print(f"[GPU {gpu_id}] Model loaded from {model_weights_path}", flush=True)
    else:
        print(f"[GPU {gpu_id}] ERROR: Model weights not found!", flush=True)
        return

    print(f"[GPU {gpu_id}] Worker ready", flush=True)

    target_idx = 0 if target_gender == 'Woman' else 1  # Woman=0, Man=1

    while True:
        item = input_queue.get()
        if item is None:  # Poison pill
            break

        batch_id, image_paths = item

        # Load and preprocess entire batch
        batch_array, valid_paths = load_and_preprocess_batch(image_paths)

        results = []

        if batch_array is not None and len(batch_array) > 0:
            try:
                # TRUE BATCH INFERENCE - all images at once!
                predictions = gender_model.predict(batch_array, verbose=0)

                # Process predictions
                for img_path, pred in zip(valid_paths, predictions):
                    predicted_idx = np.argmax(pred)
                    matched = (predicted_idx == target_idx)
                    results.append((img_path, matched))

            except Exception as e:
                print(f"[GPU {gpu_id}] Batch error: {e}", flush=True)
                # Mark all as failed
                results = [(p, False) for p in valid_paths]

        # Add failed images
        failed = set(image_paths) - set(valid_paths)
        results.extend([(p, False) for p in failed])

        output_queue.put((gpu_id, batch_id, results))

    print(f"[GPU {gpu_id}] Worker stopped", flush=True)

def filter_faces_batch(input_dir, output_dir, target_gender='Woman', batch_size=64, num_workers=2):
    """
    Filter faces using true batched multi-GPU processing

    Args:
        input_dir: Input directory
        output_dir: Output directory
        target_gender: 'Woman' or 'Man'
        batch_size: Images per batch (64 is optimal for P4 GPUs)
        num_workers: Number of GPU workers (2 for dual GPU)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(input_path.glob(ext))

    total_images = len(image_files)
    print(f"Found {total_images} images to process")
    print(f"Target: {target_gender}, Batch size: {batch_size}, Workers: {num_workers}")

    if total_images == 0:
        print("No images found!")
        return

    # Create batches
    batches = []
    for i in range(0, total_images, batch_size):
        batch_files = image_files[i:i+batch_size]
        batches.append((i // batch_size, batch_files))

    print(f"Created {len(batches)} batches")

    # Setup queues
    input_queue = mp.Queue(maxsize=num_workers * 3)  # Small buffer for prefetching
    output_queue = mp.Queue()

    # Start GPU workers
    workers = []
    for gpu_id in range(num_workers):
        p = mp.Process(target=gpu_worker, args=(gpu_id, input_queue, output_queue, target_gender))
        p.start()
        workers.append(p)

    print("\nProcessing batches...")

    # Feed batches in background thread
    def feed_batches():
        for batch in batches:
            input_queue.put(batch)
        # Send poison pills
        for _ in range(num_workers):
            input_queue.put(None)

    feeder = threading.Thread(target=feed_batches, daemon=True)
    feeder.start()

    # Collect results
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
                except Exception:
                    total_errors += 1

        processed_batches += 1

        # Progress
        processed_imgs = min(processed_batches * batch_size, total_images)
        pct = (processed_imgs / total_images) * 100
        print(f"\rProgress: {processed_imgs}/{total_images} ({pct:.1f}%) | "
              f"Matched: {total_matched} | GPU {gpu_id}", end='', flush=True)

    print()

    # Cleanup
    feeder.join()
    for w in workers:
        w.join()

    print(f"\n✓ Matched {total_matched} {target_gender} faces")
    print(f"✗ Errors: {total_errors}")
    print(f"→ Output: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch gender filtering with true GPU batching')
    parser.add_argument('--input', '-i', required=True, help='Input directory')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--gender', '-g', default='Woman', choices=['Woman', 'Man'],
                       help='Target gender (default: Woman)')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--workers', '-w', type=int, default=2,
                       help='Number of GPU workers (default: 2)')

    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    filter_faces_batch(
        args.input,
        args.output,
        args.gender,
        args.batch_size,
        args.workers
    )
