# Face Organizer - Project Plan

**Last Updated**: 2025-12-30
**Status**: Initial Development - GPU Acceleration Blocked
**Repository**: https://github.com/rmdevpro/face-organizer

---

## 1. Project Overview

### Purpose
Face Organizer is a GPU-accelerated system for organizing large collections of face images by person using unsupervised clustering. It processes 500k+ images in minutes without requiring manual labeling.

### Core Capabilities
1. **Face Embedding Generation**: Convert face images to 512-dimensional vectors using InsightFace (ArcFace model)
2. **Unsupervised Clustering**: Automatically group faces by person using FAISS K-Means
3. **Face Search**: Find all images of a specific person using similarity search
4. **Dual-GPU Processing**: Parallel processing across 2x Tesla P4 GPUs

### Current Status
🔴 **BLOCKED**: GPU acceleration for InsightFace is not working (see Section 7)
- InsightFace requires ONNX Runtime GPU support
- All attempts (9 total) to enable GPU on Tesla P4 (Pascal/sm_61) with CUDA 12.3 have failed
- Current fallback: CPU processing (5-8 img/sec vs target 400-600 img/sec on GPU)

---

## 2. Git Repository Management

### Repository Structure
```
/mnt/projects/face-organizer/
├── PROJECT_PLAN.md                 # This file - authoritative project guide
├── README.md                        # User-facing documentation
├── GPU_TROUBLESHOOTING_LOG.md      # Historical: GPU acceleration debugging (9 attempts)
├── Dockerfile                       # Container build instructions
├── docker-compose.yml               # Service configuration
├── constraints.txt                  # TensorFlow version locks
├── .gitignore                       # Excludes packages/ directory
├── packages/                        # Local only: 762MB pip wheels (NOT in git)
└── scripts/                         # Python scripts (version controlled)
    ├── generate_embeddings.py      # Primary: InsightFace embedding generation
    ├── cluster_faces.py            # Primary: FAISS K-Means clustering
    ├── gender_filter.py            # Legacy: DeepFace gender filtering
    └── benchmark_backends.py       # Utility: Performance testing
```

### Git Workflow

#### Branch Strategy
- **master**: Stable, working code only
- Feature branches: NOT required for this project (single developer)
- Direct commits to master are acceptable

#### Commit Guidelines
1. **Commit after each complete unit of work**:
   - Dockerfile changes
   - New scripts
   - Documentation updates
   - Bug fixes

2. **Commit message format**:
   ```
   Short summary (50 chars max)

   Detailed explanation if needed:
   - Bullet points for changes
   - Why the change was made
   - Related issues/blockers

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
   ```

3. **Always include**:
   - Working scripts in `scripts/` directory
   - Updated README if functionality changes
   - Troubleshooting notes in GPU_TROUBLESHOOTING_LOG.md if GPU-related

#### What NOT to Commit
❌ `packages/` directory (762MB pip wheels - local cache only)
❌ Running container state/logs
❌ Test images/embeddings
❌ Temporary work files
❌ `.pyc` files or `__pycache__`

#### Commit Checkpoint Process
Execute after:
- ✅ Completing a feature (e.g., new clustering script)
- ✅ Fixing a critical bug
- ✅ Major documentation updates
- ✅ Before switching contexts/tasks
- ✅ End of work session

**Process**:
```bash
cd /mnt/projects/face-organizer
git status                           # Review changes
git add <files>                      # Stage specific files
git commit -m "message"              # Commit with clear message
git push origin master               # Push to GitHub
```

#### Git Repository State Verification
Before starting work, always verify:
```bash
cd /mnt/projects/face-organizer
git status                           # Check for uncommitted changes
git log --oneline -5                 # Review recent commits
git remote -v                        # Verify remote URL
```

---

## 3. Intended Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Face Organizer System                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT: 500k face images (already cropped to faces)         │
│         - Strange angles supported                          │
│         - Occlusions supported (microphones, eating, etc.)  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ STAGE 1: Face Embedding Generation (InsightFace)      │ │
│  │                                                        │ │
│  │  - Model: ArcFace (buffalo_l)                         │ │
│  │  - Backend: ONNX Runtime with CUDAExecutionProvider   │ │
│  │  - Output: 512-dimensional embedding per face         │ │
│  │  - Hardware: 2x Tesla P4 GPUs (parallel processing)   │ │
│  │  - Target Speed: 400-600 img/sec per GPU              │ │
│  │  - Total Time: ~20 minutes for 500k images            │ │
│  │                                                        │ │
│  │  ⚠️ BLOCKED: GPU acceleration not working             │ │
│  │  Current: CPU fallback (5-8 img/sec = 80+ hours)      │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ STAGE 2: Clustering (FAISS K-Means)                   │ │
│  │                                                        │ │
│  │  - Algorithm: K-Means with k=2 (Person A/B)           │ │
│  │  - Input: 500k embeddings (512-dim vectors)           │ │
│  │  - Output: Cluster assignments (0 or 1)               │ │
│  │  - Hardware: Single Tesla P4 GPU                      │ │
│  │  - Speed: <30 seconds for 500k embeddings             │ │
│  │                                                        │ │
│  │  ✅ Should work once embeddings are generated         │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ STAGE 3: File Organization                            │ │
│  │                                                        │ │
│  │  - Sort images into folders based on clusters         │ │
│  │  - Output structure:                                  │ │
│  │    /output/person_A/  (e.g., 312k images)             │ │
│  │    /output/person_B/  (e.g., 210k images)             │ │
│  │    /output/outliers/  (optional: low-confidence)      │ │
│  │                                                        │ │
│  │  ✅ Simple file operations, no blockers               │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  OUTPUT: Images organized by person in separate folders     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Container Base**: TensorFlow 2.16.1 GPU (CUDA 12.3, cuDNN 9.x)
**Hardware**: 2x NVIDIA Tesla P4 (Pascal/sm_61, 8GB VRAM each)

**Key Libraries (Intended)**:
- **InsightFace 0.7.3**: Face recognition model (ArcFace)
- **ONNX Runtime GPU 1.19.2**: Model inference engine ⚠️ NOT WORKING
- **FAISS**: GPU-accelerated K-Means clustering
- **TensorFlow 2.16.1**: Base GPU support ✅ WORKING
- **DeepFace 0.0.96**: Legacy gender classification ✅ WORKING (not used for primary task)

**Key Libraries (Missing from Dockerfile)**:
❌ InsightFace - must be installed manually
❌ FAISS - must be installed manually
❌ ONNX Runtime GPU - must be installed manually

---

## 4. Use Cases

### Primary Use Case: Person Separation (Current Job)
**Scenario**: 500k face images extracted from video footage containing 2 people (Person A and Person B)

**Requirements**:
- Images already cropped to faces
- Many images have occlusions (microphones, eating, hands)
- Strange camera angles present
- No manual labeling available
- Need separation into 2 folders: person_A/ and person_B/

**Process**:
1. Generate embeddings for all 500k images (Stage 1)
2. Run K-Means with k=2 to separate into 2 clusters (Stage 2)
3. Move images to person_A/ and person_B/ folders based on cluster assignment (Stage 3)

**Expected Performance** (when GPU working):
- Stage 1: 20 minutes (dual Tesla P4)
- Stage 2: 30 seconds
- Stage 3: 5 minutes
- **Total: ~25 minutes**

**Current Performance** (CPU fallback):
- Stage 1: 80+ hours (unacceptable)
- Stage 2: N/A (blocked by Stage 1)
- Stage 3: N/A (blocked by Stage 1)

### Secondary Use Case: Photo Library Organization
**Scenario**: Organize 50k personal photos by person

**Process**:
1. Extract faces from photos (use external tool or add face detection stage)
2. Generate embeddings for all faces
3. Run K-Means with k=N (unknown number of people, use elbow method)
4. Manually label one image per cluster
5. Organize photos by identified person

### Tertiary Use Case: Face Search
**Scenario**: Find all photos containing a specific person

**Process**:
1. Generate embeddings for entire photo library (one-time)
2. Generate embedding for reference photo of target person
3. Use FAISS similarity search to find nearest neighbors
4. Return all photos with similarity score > threshold

### Legacy Use Case: Gender Classification
**Scenario**: Filter faces by gender (NOT the primary purpose)

**Status**: ✅ WORKING with DeepFace + TensorFlow
**Note**: This is a capability of the container but NOT the intended primary use

---

## 5. Docker Image Management

### Current State (2025-12-30)

**Running Container**:
- Name: `deepface-gpu` (OLD NAME, still running)
- Image: `ddd616452217` (built Dec 28, 2025 12:28 PM)
- Started: Dec 28, 2025 12:41 PM (running 2+ days)
- Contains: TensorFlow 2.16.1, DeepFace 0.0.96
- Missing: InsightFace, FAISS, ONNX Runtime
- Modifications: Manually installed onnxruntime-gpu 1.19.2, InsightFace 0.7.3 (NOT in image)

**Unused Image**:
- Tag: `deepface-gpu:latest` (OLD NAME)
- Image ID: `43ff5d6a806f` (built Dec 30, 2025 10:55 AM)
- Contains: TensorFlow 2.16.1, DeepFace 0.0.96
- Missing: InsightFace, FAISS, ONNX Runtime
- Status: Never tested

**Problem**:
- Running container has manual modifications (not reproducible)
- Images don't contain required libraries for primary use case
- Image names use old "deepface-gpu" naming

### Proper Image Build Process

**Before building**:
1. Update Dockerfile to install InsightFace, FAISS, ONNX Runtime
2. Test package installation on a test image first
3. Ensure GPU acceleration is verified before committing Dockerfile
4. Commit Dockerfile changes to git

**Build command**:
```bash
cd /mnt/projects/face-organizer
docker build -t face-organizer:latest .
```

**Test command**:
```bash
docker run --rm --gpus all face-organizer:latest \
  python -c "import insightface; import onnxruntime; print(onnxruntime.get_available_providers())"
```

**Deploy command**:
```bash
docker-compose up -d
```

### Image Versioning Strategy

**Tag format**: `face-organizer:YYYY-MM-DD-description`

**Examples**:
- `face-organizer:2025-12-30-base` - Initial build with TensorFlow only
- `face-organizer:2025-12-31-insightface` - After adding InsightFace
- `face-organizer:latest` - Always points to most recent stable build

**When to tag**:
- ✅ After successfully enabling GPU acceleration
- ✅ After adding new major library (InsightFace, FAISS)
- ✅ Before making risky changes (allows rollback)
- ❌ NOT during troubleshooting/experimentation

---

## 6. Development Workflow

### Standard Development Cycle

1. **Plan** (use this document + issues)
2. **Implement** (edit Dockerfile, scripts, etc.)
3. **Test** (in running container or test build)
4. **Document** (update README, this plan, or troubleshooting log)
5. **Commit** (git commit + push)
6. **Build** (docker build - only if Dockerfile changed)
7. **Deploy** (docker-compose up - only if verified working)

### Troubleshooting Workflow

When something doesn't work:

1. **Document in GPU_TROUBLESHOOTING_LOG.md**:
   - What you tried
   - Exact commands/code used
   - Error messages (full output)
   - Analysis of why it failed
   - Next steps to try

2. **Don't modify running container without documenting**:
   - All pip installs must be recorded
   - All configuration changes must be noted
   - Track timestamps of changes

3. **Consult external AI (Gemini CLI)**:
   - Record VERBATIM conversations in troubleshooting log
   - Include full context in queries
   - Document recommendations even if not implemented

4. **Update this plan if blockers change**:
   - Status field in Section 1
   - Current State in Section 5
   - Known Issues in Section 7

### Testing Strategy

**For GPU Acceleration** (currently blocked):
1. Create minimal test script (`verify_gpu.py`)
2. Test with single image first
3. Monitor `nvidia-smi` in parallel terminal
4. Check for GPU memory usage and process
5. Verify speed (should be 400-600 img/sec, not 5-8 img/sec)

**For Clustering**:
1. Test with small dataset first (1000 images)
2. Verify cluster distribution is reasonable
3. Manually check sample images from each cluster
4. Scale to full dataset only after verification

---

## 7. Known Issues & Blockers

### CRITICAL BLOCKER: InsightFace GPU Acceleration

**Issue**: ONNX Runtime cannot use Tesla P4 GPUs for InsightFace inference

**Root Cause**:
- Tesla P4 = Pascal architecture (Compute Capability 6.1)
- Container has CUDA 12.3 + cuDNN 9.x
- Modern ONNX Runtime versions (1.15-1.23) either:
  - Look for CUDA 11 libraries that don't exist in CUDA 12 container
  - Find CUDA 12 but cuDNN kernels fail with compute capability errors
- Pre-built ONNX Runtime binaries are incompatible with Pascal + CUDA 12

**Attempted Solutions** (all failed):
1. onnxruntime-gpu 1.15.1 + NumPy downgrade
2. onnxruntime-gpu 1.17.0 (CUDA 12 support)
3. onnxruntime-gpu 1.23.2 (latest)
4. Symlinks for CUDA 11 libraries
5. onnxruntime-gpu 1.21.0 with explicit provider config
6. onnxruntime-gpu 1.18.0 (recommended by Gemini #1)
7. Installing actual CUDA 11 libraries (incomplete)
8. onnxruntime-gpu 1.18.0 with explicit CUDA provider (recommended by Gemini #2)
9. onnxruntime-gpu 1.19.2 from CUDA 12 repo + HEURISTIC algorithm (recommended by Gemini #3)

**Current Status**:
- CUDAExecutionProvider is ACTIVE (no CPU fallback)
- But cuDNN execution fails: `CUDNN_STATUS_EXECUTION_FAILED_CUDART`
- Error occurs during cudnnConvolutionForward in FusedConv node
- HEURISTIC algorithm selection doesn't help

**Potential Solutions** (not yet tried):
1. **Disable graph optimization** (from Google AI conversation):
   ```python
   import onnxruntime as ort
   sess_options = ort.SessionOptions()
   sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
   ```

2. **Build ONNX Runtime from source** for CUDA 12 + Pascal (high effort)

3. **Use MXNet backend** for InsightFace instead of ONNX Runtime

4. **Use Vision Transformer (ViT)** instead of InsightFace (recommended by Google AI for occlusions)

5. **Downgrade to CUDA 11.8 ONNX Runtime** build (recommended by Gemini)

**Historical Documentation**: See `GPU_TROUBLESHOOTING_LOG.md` for complete details

### Non-Critical Issues

**Missing Libraries in Dockerfile**:
- InsightFace not in base image (must install manually)
- FAISS not in base image (must install manually)
- ONNX Runtime not in base image (must install manually)

**Impact**: Images are not reproducible, cannot deploy to production

**Solution**: Update Dockerfile once GPU acceleration is working

---

## 8. Success Criteria

### Minimum Viable Product (MVP)
- ✅ Dockerfile builds successfully
- ✅ Container can see 2x Tesla P4 GPUs
- ⏳ InsightFace uses GPU acceleration (400+ img/sec)
- ⏳ Process 500k images in <30 minutes
- ⏳ K-Means clustering completes in <1 minute
- ⏳ Images sorted into person_A/ and person_B/ folders
- ⏳ Manual verification shows >90% accuracy

### Production Ready
- ⏳ All dependencies in Dockerfile (reproducible)
- ⏳ Docker image tagged and pushed to registry
- ⏳ Documentation complete (README + this plan)
- ⏳ Scripts handle errors gracefully
- ⏳ GPU monitoring/logging integrated
- ⏳ Can process multiple jobs without restart

### Future Enhancements
- ⏳ Support for N-person clustering (not just 2)
- ⏳ Web UI for cluster visualization
- ⏳ Real-time processing API
- ⏳ Face detection stage (accept full images, not just crops)
- ⏳ TensorRT optimization for maximum speed

---

## 9. References

### Internal Documentation
- `README.md` - User-facing documentation and usage examples
- `GPU_TROUBLESHOOTING_LOG.md` - Complete GPU acceleration debugging history
  - Contains 9 failed attempts with full error logs
  - Includes verbatim Gemini CLI conversations
  - Documents Google AI conversation from Dec 28, 2025

### External Resources
- [InsightFace Documentation](https://github.com/deepinsight/insightface)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [ONNX Runtime CUDA Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [Tesla P4 Specifications](https://www.nvidia.com/en-us/data-center/tesla-p4/)

### Key Conversations
- **Gemini CLI Session** (2025-12-30 20:42): Recommended onnxruntime-gpu==1.18.0 approach
- **Google AI Conversation** (2025-12-28): Vision Transformer recommendation for occlusions
- **Gemini CLI Sessions** (2025-12-30 21:56, 21:59, 22:00): CUDA 12 repo + HEURISTIC approach

---

## 10. Next Actions

### Immediate (Unblock GPU Acceleration)
1. ⏳ Try Google AI's graph optimization disable approach (Attempt 10)
2. ⏳ If that fails, consult Gemini about CUDA 11.8 downgrade
3. ⏳ If that fails, investigate MXNet backend
4. ⏳ If that fails, consider Vision Transformer approach

### After GPU Working
1. ⏳ Update Dockerfile to include working configuration
2. ⏳ Test full 500k image pipeline
3. ⏳ Commit working Dockerfile + scripts
4. ⏳ Tag stable Docker image
5. ⏳ Update README with verified performance numbers

### Long Term
1. ⏳ Clean up old Docker images (12+ failed builds)
2. ⏳ Add CI/CD for automated testing
3. ⏳ Consider GPU upgrade if Pascal incompatibility persists
4. ⏳ Explore TensorRT optimization

---

**Document Owner**: Claude Code
**Review Cycle**: Update after each major change or blocker resolution
