# DeepFace GPU Container

Docker container for GPU-accelerated face analysis using DeepFace and TensorFlow.

## Purpose

This container is designed for **high-speed face classification and filtering** in FaceSwap workflows. Specifically:

- **Filter extracted faces by gender** (removing unwanted faces)
- **GPU-accelerated processing** (~10-15 min for 579k faces vs hours on CPU)
- **Integrated with FaceSwap pipeline** (shares same volume mounts)

**Use Case**: You've extracted 579k faces from video events (2006-2020) and need to filter by gender to isolate target person(s) for face swap training.

## Features

- **TensorFlow 2.16.1** with working GPU support (2x Tesla P4)
- **DeepFace 0.0.96** for face analysis (gender, age, emotion, race)
- **Local package caching** (762MB) for fast rebuilds without re-downloading
- **Version-locked dependencies** via `constraints.txt` to prevent GPU breakage

## Repository Structure

```
/mnt/projects/deepface-gpu/
├── Dockerfile              # Container build instructions
├── docker-compose.yml      # Service configuration
├── constraints.txt         # Version locks for TensorFlow GPU compatibility
├── .gitignore             # Excludes packages/ from git
├── packages/              # 762MB pip wheels (GITIGNORED - local only)
│   ├── tensorflow-2.16.1-*.whl
│   ├── deepface-0.0.96-*.whl
│   └── ... (67 packages total)
├── scripts/               # Custom Python scripts (version controlled)
│   └── gender_filter.py   # Filter faces by gender
└── README.md             # This file
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
cd /mnt/projects/deepface-gpu

# Download all packages using the base TensorFlow image's Python version
docker run --rm -v $(pwd)/packages:/packages \
  tensorflow/tensorflow:2.16.1-gpu \
  bash -c 'pip download --dest /packages deepface tf-keras'
```

This downloads ~762MB of packages to the local `packages/` directory.

### Build Container

```bash
docker build -t deepface-gpu:latest .
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
  --name deepface-gpu \
  --gpus all \
  --runtime nvidia \
  -v /mnt/win_share/faceswap:/workspace \
  -v /mnt/win_share/use:/mnt/win_share/use:ro \
  deepface-gpu:latest
```

## Usage

### Verify GPU Access

```bash
docker exec deepface-gpu python -c "import tensorflow as tf; \
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
docker exec deepface-gpu python /app/scripts/gender_filter.py \
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
  docker exec deepface-gpu python /app/scripts/gender_filter.py \
    --input /workspace/input/$YEAR \
    --output /workspace/faces_female/$YEAR \
    --gender Woman
done
```

### Interactive Python

```bash
docker exec -it deepface-gpu python
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
cd /mnt/projects/deepface-gpu

# Remove old packages
rm -rf packages/*

# Download new versions
docker run --rm -v $(pwd)/packages:/packages \
  tensorflow/tensorflow:2.16.1-gpu \
  bash -c 'pip download --dest /packages deepface tf-keras'

# Rebuild container
docker build --no-cache -t deepface-gpu:latest .
```

**⚠️ Warning**: Updating packages may break GPU support. Test GPU detection after rebuild.

### Rebuild from Scratch

```bash
# Remove old image and containers
docker-compose down
docker rmi deepface-gpu:latest

# Rebuild
docker build --no-cache -t deepface-gpu:latest .
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
docker exec deepface-gpu nvidia-smi

# 3. Check TensorFlow version (MUST be 2.16.1)
docker exec deepface-gpu python -c "import tensorflow as tf; print(tf.__version__)"

# 4. If not 2.16.1, rebuild with fresh constraints.txt
docker run --rm tensorflow/tensorflow:2.16.1-gpu \
  pip freeze > constraints.txt
docker build --no-cache -t deepface-gpu:latest .
```

### DeepFace Import Errors

**Error**: `ModuleNotFoundError: No module named 'tf_keras'`

**Solution**: Install tf-keras (already included in build):
```bash
docker exec deepface-gpu pip install tf-keras
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

| Task | Count | Time (GPU) | Time (CPU) |
|------|-------|-----------|------------|
| Gender classification | 177k faces | ~10-15 min | ~2-3 hours |
| Full analysis (gender+age+emotion) | 177k faces | ~25-30 min | ~6-8 hours |

## Workflow Integration

This container integrates into the FaceSwap workflow:

```
1. Extract faces from videos (faceswap-vnc container)
   → /workspace/input/YYYY/*.png (579k total faces)

2. Filter by gender (this container - deepface-gpu)
   → /workspace/faces_female/YYYY/*.png (filtered set)

3. Further filter for target person (manual or ML)
   → /workspace/faces_target/*.png (final training set)

4. Train FaceSwap model (faceswap-vnc container)
   → /workspace/models/my_model/

5. Convert videos (faceswap-vnc container)
   → /workspace/output/*.mp4
```

## Repository Management

### Git Workflow

```bash
# Make changes to Dockerfile or scripts
vim Dockerfile

# Test build
docker build -t deepface-gpu:test .

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
3. Check container logs: `docker logs deepface-gpu`
4. Review build output for errors

---

**Last Updated**: 2025-12-28
**TensorFlow Version**: 2.16.1
**DeepFace Version**: 0.0.96
**Container Size**: 11.5GB
**Package Cache**: 762MB
