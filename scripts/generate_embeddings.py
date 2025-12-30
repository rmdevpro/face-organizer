#!/usr/bin/env python3
"""
Generate face embeddings using InsightFace (ArcFace model)
Handles occlusions and strange angles better than DeepFace
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from tqdm import tqdm

# Set GPU before imports
GPU_ID = int(os.environ.get('GPU_ID', '0'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

import insightface
from insightface.app import FaceAnalysis

print(f"GPU {GPU_ID}: Loading InsightFace model...")

# Initialize InsightFace with ArcFace model
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

print(f"GPU {GPU_ID}: InsightFace loaded")

def extract_embedding(img_path):
    """
    Extract face embedding from image

    Returns:
        tuple: (img_path, embedding) or (img_path, None) if no face detected
    """
    try:
        # Read image
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            return (img_path, None)

        # Detect and analyze face
        faces = app.get(img)

        if len(faces) == 0:
            return (img_path, None)

        # Use the largest face if multiple detected
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)

        # Get the 512-dim embedding (normed_embedding is L2 normalized)
        embedding = faces[0].normed_embedding

        return (img_path, embedding)

    except Exception as e:
        return (img_path, None)

def generate_embeddings(input_dirs, output_file, workers=8):
    """
    Generate embeddings for all images in input directories

    Args:
        input_dirs: List of directories containing face images
        output_file: Path to save embeddings (pickle file)
        workers: Number of parallel workers
    """
    # Collect all image files
    image_files = []
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(input_path.glob(ext))

    total = len(image_files)
    print(f"\nGPU {GPU_ID}: Processing {total} images with {workers} workers\n")

    embeddings = []
    image_paths = []
    no_face_count = 0
    processed = 0

    # Process with thread pool
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        futures = {executor.submit(extract_embedding, img_path): img_path
                   for img_path in image_files}

        # Process results as they complete
        for future in tqdm(as_completed(futures), total=total, desc=f"GPU {GPU_ID}"):
            img_path, embedding = future.result()

            if embedding is not None:
                embeddings.append(embedding)
                image_paths.append(str(img_path))
            else:
                no_face_count += 1

            processed += 1

            # Progress every 1000 images
            if processed % 1000 == 0:
                detection_rate = ((processed - no_face_count) / processed * 100)
                print(f"\rGPU {GPU_ID}: {processed}/{total} ({detection_rate:.1f}% detected)",
                      end='', flush=True)

    # Convert to numpy arrays
    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Save to disk
    print(f"\n\nGPU {GPU_ID}: Saving {len(embeddings)} embeddings to {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings_array,
            'image_paths': image_paths,
            'no_face_count': no_face_count,
            'total_images': total
        }, f)

    detection_rate = (len(embeddings) / total * 100) if total > 0 else 0
    print(f"\nGPU {GPU_ID}: Complete!")
    print(f"  ✓ Detected: {len(embeddings)}/{total} ({detection_rate:.1f}%)")
    print(f"  ✗ No face: {no_face_count}")
    print(f"  → Saved to: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate face embeddings with InsightFace')
    parser.add_argument('--input', '-i', nargs='+', required=True,
                       help='Input directories with face images')
    parser.add_argument('--output', '-o', required=True,
                       help='Output pickle file for embeddings')
    parser.add_argument('--workers', '-w', type=int, default=8,
                       help='Parallel workers (default: 8)')

    args = parser.parse_args()

    generate_embeddings(args.input, args.output, args.workers)
