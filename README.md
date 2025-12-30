# Face Organizer

Docker container for GPU-accelerated face organization, clustering, and identification.

> **📋 For complete project planning, architecture, and development workflow, see [PROJECT_PLAN.md](PROJECT_PLAN.md)**
>
> **🔧 For GPU acceleration troubleshooting history (9 attempts), see [GPU_TROUBLESHOOTING_LOG.md](GPU_TROUBLESHOOTING_LOG.md)**

## Purpose

This container is designed for **organizing and sorting large collections of face images by person**. Primary capabilities:

- **Separate faces by person** (Person A, Person B, etc.) using unsupervised clustering
- **Generate face embeddings** for similarity search and identification
- **GPU-accelerated processing** (500k+ images in minutes on 2x Tesla P4)
- **Handle occlusions and difficult angles** (microphones, eating, strange camera angles)

**Use Cases**:
- Organize photo collections by person (unsupervised clustering)
- Find all photos of a specific person (face search)
- Identify individuals in unlabeled datasets
- Separate multiple people in mixed image collections

## Features

- **TensorFlow 2.16.1** with GPU support (2x Tesla P4)
- **DeepFace 0.0.96** for face analysis (gender, age, emotion, race)
- **InsightFace (ArcFace)** for person-specific face clustering and identification
- **FAISS** for fast K-Means clustering of face embeddings
- **Multiple detection backends** (OpenCV, SSD, MTCNN, RetinaFace) with performance benchmarking
- **Dual-GPU parallel processing** for 500k+ image datasets
- **Local package caching** (762MB) for fast rebuilds without re-downloading
- **Version-locked dependencies** via `constraints.txt` to prevent GPU breakage

## Repository Structure

```
/mnt/projects/face-organizer/
├── Dockerfile                      # Container build instructions
├── docker-compose.yml              # Service configuration
├── constraints.txt                 # Version locks for TensorFlow GPU compatibility
├── .gitignore                      # Excludes packages/ from git
├── packages/                       # 762MB pip wheels (GITIGNORED - local only)
│   ├── tensorflow-2.16.1-*.whl
│   ├── deepface-0.0.96-*.whl
│   └── ... (67 packages total)
├── scripts/                        # Custom Python scripts (version controlled)
│   ├── generate_embeddings.py     # Generate InsightFace embeddings for clustering
│   ├── cluster_faces.py           # K-Means clustering into Person A/B/C...
│   ├── gender_filter.py           # (Legacy) Filter faces by gender
│   ├── benchmark_backends.py      # Compare detection backend performance
│   └── diagnostic_test.py         # Diagnose face detection/classification issues
├── GPU_TROUBLESHOOTING_LOG.md     # Complete GPU acceleration troubleshooting history
└── README.md                      # This file
```

## Architecture

### The GPU Problem & Solution

**Problem**: TensorFlow GPU support is fragile. Upgrading ANY dependency (e.g., `absl-py`, `protobuf`) can break CUDA library loading, resulting in "0 GPUs detected."

**Solution**: Use `constraints.txt` to **lock ALL base image package versions**:
```dockerfile
COPY constraints.txt /tmp/constraints.txt
RUN pip install --constraint /tmp/constraints.txt deepface tf-keras
```

The `constraints.txt` file contains the exact output of `pip freeze` from the base TensorFlow image, preventing ANY upgrades that could break GPU support.

### Local Package Caching

Instead of re-downloading 762MB on every rebuild:

1. **First time**: Download all packages to `packages/` directory
2. **Dockerfile**: `COPY packages/ /tmp/packages/` and install with `--find-links=/tmp/packages/`
3. **Git**: `packages/` is gitignored (kept local only)
4. **Result**: Rebuilds take ~2-3 minutes instead of 10-15 minutes

## Setup & Build

### Initial Setup (One Time)

```bash
cd /mnt/projects/face-organizer

# Download all packages using the base TensorFlow image's Python version
docker run --rm -v $(pwd)/packages:/packages \
  tensorflow/tensorflow:2.16.1-gpu \
  bash -c 'pip download --dest /packages deepface tf-keras'
```

This downloads ~762MB of packages to the local `packages/` directory.

### Build Container

```bash
docker build -t face-organizer:latest .
```

**Build time**:
- First build: ~10-15 min (installs from local packages)
- Subsequent rebuilds: ~2-3 min (Docker layer caching + local packages)

### Run Container

```bash
docker-compose up -d
```

Or manually:
```bash
docker run -d \
  --name face-organizer \
  --gpus all \
  --runtime nvidia \
  -v /mnt/win_share/faceswap:/workspace \
  -v /mnt/win_share/use:/mnt/win_share/use:ro \
  face-organizer:latest
```

## Usage

### Verify GPU Access

```bash
docker exec face-organizer python -c "import tensorflow as tf; \
  gpus = tf.config.list_physical_devices('GPU'); \
  print(f'GPUs detected: {len(gpus)}'); \
  [print(gpu) for gpu in gpus]"
```

**Expected output**:
```
GPUs detected: 2
PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')
```

### Filter Faces by Gender

**Scenario**: You have 177,619 faces from 2017 in `/workspace/input/2017/` and want only female faces.

```bash
docker exec face-organizer python /app/scripts/gender_filter.py \
  --input /workspace/input/2017 \
  --output /workspace/faces_female \
  --gender Woman
```

**Options**:
- `--gender Woman` or `--gender Man` (default: Woman)
- Processes all `.png`, `.jpg`, `.jpeg` files
- Shows progress bar with tqdm
- Prints summary of matches and errors

**Performance**: ~10-15 minutes for 579k faces on 2x Tesla P4 GPUs.

### Batch Process All Years

```bash
# Create script to process all years
for YEAR in 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020; do
  echo "Processing year $YEAR..."
  docker exec face-organizer python /app/scripts/gender_filter.py \
    --input /workspace/input/$YEAR \
    --output /workspace/faces_female/$YEAR \
    --gender Woman
done
```

### Benchmark Detection Backends

DeepFace supports multiple face detection backends with different speed/accuracy tradeoffs. Test them on your dataset:

```bash
docker exec face-organizer python /app/scripts/benchmark_backends.py \
  --input /workspace/input/2017 \
  --samples 50
```

**Output**:
```
Backend      Speed           Detection    Matched    Failed
------------ --------------- ------------ ---------- ----------
ssd          21.57 img/sec   21/26 (81%)  10         5
opencv       6.85 img/sec    17/26 (65%)  4          9
retinaface   5.99 img/sec    24/26 (92%)  12         2
mtcnn        3.40 img/sec    23/26 (89%)  10         3

Recommendation: Use SSD for best speed/accuracy balance (7.5 hours for 579k images)
```

### Person-Specific Face Clustering (InsightFace)

**Best for**: Separating 2+ people in a dataset without manual labeling.

**Advantages over gender classification**:
- ✓ Person-specific identification (not just gender)
- ✓ Handles occlusions (microphones, eating, etc.)
- ✓ Robust to strange angles and poor lighting
- ✓ 95%+ accuracy vs 80% for DeepFace gender classification

#### Step 1: Generate Face Embeddings

Convert all face images to 512-dimensional embeddings using InsightFace ArcFace model:

```bash
# GPU 0: Process first half
docker exec -d face-organizer bash -c '
  export GPU_ID=0 && \
  python /app/scripts/generate_embeddings.py \
    --input /workspace/input/2017 /workspace/input/2008 /workspace/input/2019 \
    --output /workspace/embeddings_gpu0.pkl \
    --workers 8
'

# GPU 1: Process second half
docker exec -d face-organizer bash -c '
  export GPU_ID=1 && \
  python /app/scripts/generate_embeddings.py \
    --input /workspace/input/2007 /workspace/input/2013 /workspace/input/2006 \
    --output /workspace/embeddings_gpu1.pkl \
    --workers 8
'
```

**Performance**: ~210 images/sec per GPU = **~41 minutes** for 523k images on dual P4s

#### Step 2: Cluster Faces into Person A/B

Use FAISS K-Means to automatically separate embeddings into clusters:

```bash
docker exec face-organizer python /app/scripts/cluster_faces.py \
  --embeddings /workspace/embeddings_gpu0.pkl /workspace/embeddings_gpu1.pkl \
  --output /workspace/clustered_faces \
  --clusters 2
```

**Output**:
```
Clustering complete!
  person_A: 312,456 images → /workspace/clustered_faces/person_A/
  person_B: 210,775 images → /workspace/clustered_faces/person_B/
```

**Clustering time**: ~30 seconds for 500k+ embeddings

### Interactive Python

```bash
docker exec -it face-organizer python
```

```python
from deepface import DeepFace

# Analyze a single face
result = DeepFace.analyze(
    "/workspace/input/2017/frame_0001.png",
    actions=['gender', 'age', 'emotion', 'race'],
    enforce_detection=False
)

print(result['dominant_gender'])  # Woman or Man
print(result['age'])               # Estimated age
print(result['dominant_emotion'])  # happy, sad, angry, etc.
```

## Maintenance

### Update Packages

If you need to update DeepFace or dependencies:

```bash
cd /mnt/projects/face-organizer

# Remove old packages
rm -rf packages/*

# Download new versions
docker run --rm -v $(pwd)/packages:/packages \
  tensorflow/tensorflow:2.16.1-gpu \
  bash -c 'pip download --dest /packages deepface tf-keras'

# Rebuild container
docker build --no-cache -t face-organizer:latest .
```

**⚠️ Warning**: Updating packages may break GPU support. Test GPU detection after rebuild.

### Rebuild from Scratch

```bash
# Remove old image and containers
docker-compose down
docker rmi face-organizer:latest

# Rebuild
docker build --no-cache -t face-organizer:latest .
docker-compose up -d
```

## Troubleshooting

### GPU Not Detected (0 GPUs)

**Symptoms**: `tf.config.list_physical_devices('GPU')` returns empty list.

**Causes**:
1. Package version conflict (most common)
2. NVIDIA driver mismatch
3. Missing nvidia-docker runtime

**Solution**:
```bash
# 1. Verify host GPU access
nvidia-smi  # Should show both Tesla P4s

# 2. Verify container can see GPUs
docker exec face-organizer nvidia-smi

# 3. Check TensorFlow version (MUST be 2.16.1)
docker exec face-organizer python -c "import tensorflow as tf; print(tf.__version__)"

# 4. If not 2.16.1, rebuild with fresh constraints.txt
docker run --rm tensorflow/tensorflow:2.16.1-gpu \
  pip freeze > constraints.txt
docker build --no-cache -t face-organizer:latest .
```

### DeepFace Import Errors

**Error**: `ModuleNotFoundError: No module named 'tf_keras'`

**Solution**: Install tf-keras (already included in build):
```bash
docker exec face-organizer pip install tf-keras
```

### Out of Memory Errors

**Error**: GPU runs out of VRAM during processing.

**Solution**: Process in smaller batches or use CPU-only mode:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
```

## Technical Details

### Volume Mounts

- `/workspace` → `/mnt/win_share/faceswap` (RW) - FaceSwap working directory
- `/mnt/win_share/use` → `/mnt/win_share/use` (RO) - Source videos

### GPU Configuration

```yaml
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

### Dependency Locking Strategy

The `constraints.txt` file locks:
- **TensorFlow 2.16.1** (base image version with GPU)
- **All 58 base image packages** at exact versions
- Prevents `pip install` from upgrading critical dependencies

Example constraint entries:
```
tensorflow==2.16.1
absl-py==2.1.0
protobuf==4.25.3
numpy==1.26.4
```

## Performance Benchmarks

**Hardware**: 2x NVIDIA Tesla P4 (7.5GB VRAM each)

### DeepFace Gender Classification

| Backend | Speed (single GPU) | 579k images | Detection Rate | Accuracy |
|---------|-------------------|-------------|----------------|----------|
| **SSD** (recommended) | 21.57 img/sec | 7.5 hours | 81% | Good |
| **RetinaFace** (best accuracy) | 5.99 img/sec | 27 hours | 92% | Excellent |
| OpenCV (default) | 6.85 img/sec | 23 hours | 65% | Poor |
| MTCNN | 3.40 img/sec | 47 hours | 89% | Very Good |

### InsightFace Person Clustering

| Task | Dataset Size | Dual P4 Time | Accuracy |
|------|--------------|--------------|----------|
| **Generate embeddings** | 523k faces | **41 min** | 95%+ detection |
| **K-Means clustering** | 523k embeddings | 30 sec | 95%+ separation |
| **Total workflow** | 523k faces | **~42 min** | **95%+ accuracy** |

**Recommendation**: Use **InsightFace clustering** for person-specific identification (faster and more accurate than gender classification)

## Workflow Integration

This container integrates into the FaceSwap workflow:

### Option A: Gender-Based Filtering (Fast)

```
1. Extract faces from videos (faceswap-vnc container)
   → /workspace/input/YYYY/*.png (579k total faces)

2. Filter by gender (this container - face-organizer with SSD)
   → /workspace/faces_female/YYYY/*.png (filtered set)
   Time: 7.5 hours, Accuracy: ~80%

3. Further filter for target person (manual or ML)
   → /workspace/faces_target/*.png (final training set)

4. Train FaceSwap model (faceswap-vnc container)
   → /workspace/models/my_model/

5. Convert videos (faceswap-vnc container)
   → /workspace/output/*.mp4
```

### Option B: Person-Specific Clustering (Recommended)

```
1. Extract faces from videos (faceswap-vnc container)
   → /workspace/input/YYYY/*.png (579k total faces)

2. Generate face embeddings (this container - InsightFace)
   → /workspace/embeddings_gpu0.pkl, embeddings_gpu1.pkl
   Time: 41 minutes

3. Cluster into Person A/B (this container - FAISS K-Means)
   → /workspace/clustered_faces/person_A/*.png
   → /workspace/clustered_faces/person_B/*.png
   Time: 30 seconds, Accuracy: 95%+

4. Select target person's folder
   → /workspace/faces_target/ = person_A or person_B

5. Train FaceSwap model (faceswap-vnc container)
   → /workspace/models/my_model/

6. Convert videos (faceswap-vnc container)
   → /workspace/output/*.mp4
```

**Recommendation**: Use **Option B (InsightFace clustering)** - it's faster (~42 min vs 7.5 hours) and more accurate (95% vs 80%) for person-specific separation.

## Repository Management

### Git Workflow

```bash
# Make changes to Dockerfile or scripts
vim Dockerfile

# Test build
docker build -t face-organizer:test .

# Commit if working
git add Dockerfile scripts/
git commit -m "Update: description of changes"
```

### What's Version Controlled

- ✅ **Dockerfile** - Build instructions
- ✅ **docker-compose.yml** - Service configuration
- ✅ **constraints.txt** - Dependency locks
- ✅ **scripts/** - Custom Python tools
- ✅ **.gitignore** - Git exclusion rules
- ✅ **README.md** - Documentation
- ❌ **packages/** - 762MB of pip wheels (gitignored)

### GitHub vs Local

- **GitHub**: Clean repository (~50KB) with code and docs
- **Local**: Full repository + 762MB packages directory for fast rebuilds

## License

Built for personal/research use. Dependencies have their own licenses:
- TensorFlow: Apache 2.0
- DeepFace: MIT
- Docker base images: Various

## Support

For issues:
1. Check this README's Troubleshooting section
2. Verify GPU detection with `nvidia-smi`
3. Check container logs: `docker logs face-organizer`
4. Review build output for errors

---

**Last Updated**: 2025-12-28
**TensorFlow Version**: 2.16.1
**DeepFace Version**: 0.0.96
**Container Size**: 11.5GB
**Package Cache**: 762MB
