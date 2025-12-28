# DeepFace GPU Container

Docker container for running DeepFace face analysis with GPU support.

## Features

- TensorFlow 2.16.1 with GPU support
- DeepFace library for face analysis (gender, age, emotion, race)
- NVIDIA GPU acceleration
- Local package caching for fast rebuilds

## Repository Structure

```
/mnt/projects/deepface-gpu/
├── Dockerfile              # Container build instructions
├── docker-compose.yml      # Docker Compose configuration
├── .gitignore             # Git ignore rules
├── packages/              # Downloaded pip packages (GITIGNORED - local only)
├── scripts/               # Custom Python scripts (version controlled)
│   └── gender_filter.py   # Gender-based face filtering
├── configs/               # Configuration files (version controlled)
└── README.md             # This file
```

## Build

First build downloads packages from PyPI:
```bash
cd /mnt/projects/deepface-gpu
python3 -m pip download --dest packages/ blinker deepface opencv-python-headless pillow tqdm
docker build -t deepface-gpu:latest .
```

Subsequent rebuilds use local packages (fast):
```bash
docker build -t deepface-gpu:latest .
```

## Run

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

### Gender Classification Example

```bash
docker exec deepface-gpu python /app/scripts/gender_filter.py \
  --input /workspace/input/2017 \
  --output /workspace/faces_female
```

### Interactive Shell

```bash
docker exec -it deepface-gpu bash
```

### Python Interactive

```bash
docker exec -it deepface-gpu python
>>> from deepface import DeepFace
>>> result = DeepFace.analyze("image.jpg", actions=['gender', 'age'])
>>> print(result)
```

## Package Management

The `packages/` directory contains all downloaded pip wheels and is gitignored to keep GitHub lean while providing fast local rebuilds.

To update packages:
```bash
cd /mnt/projects/deepface-gpu
rm -rf packages/*
python3 -m pip download --dest packages/ blinker deepface opencv-python-headless pillow tqdm
```

## GPU Access

Verify GPU is accessible:
```bash
docker exec deepface-gpu python -c "import tensorflow as tf; print('GPUs:', len(tf.config.list_physical_devices('GPU')))"
```

## Notes

- First build: ~10-15 minutes (downloads packages)
- Rebuild from local packages: ~2-3 minutes
- Packages directory: ~500MB (gitignored, local only)
- Built image size: ~10GB
