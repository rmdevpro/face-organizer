# DeepFace Embedding Generation Work Log

## !!!!! CPU IS NEVER AN OPTION !!!!!
## MUST USE GPU ACCELERATION - NO EXCEPTIONS
## GOAL: Separate 523k faces into Person A and Person B using GPU

## Session Date: 2025-12-29

### CRITICAL ERROR - Session Started Wrong
- **MISTAKE**: Built new container on .200 (Quadro P1000)
- **CORRECT**: Should use existing container on .210 (2x Tesla P4)
- **Time Wasted**: Entire session diagnosing wrong GPU
- **Container on .210**: Running for 46 hours, has InsightFace/FAISS installed


### Attempt 1 - Failed
- Downgraded ONNX Runtime to 1.15.1, NumPy to 1.26.4
- **PROBLEM**: Still using CPUExecutionProvider (CPU only)
- Speed: 5-8 images/sec (should be 200+ with GPU)
- Need to force CUDAExecutionProvider


### Attempt 2 - Failed
- Created script with explicit CUDA provider options
- **PROBLEM**: ONNX Runtime 1.15.1 is for CUDA 11, container has CUDA 12.3
- Still falling back to CPU (5-8 it/sec)
- Need ONNX Runtime version compatible with CUDA 12.3


### Attempt 3 - Failed
- Upgraded ONNX Runtime to 1.17.0 (supposed to support CUDA 12)
- **PROBLEM**: ONNX Runtime 1.17.0 pip package still looking for CUDA 11 libraries
- Error: `libcublasLt.so.11: cannot open shared object file`
- Container has CUDA 12.3 with `libcublasLt.so.12`, not version 11
- Still using CPUExecutionProvider (5-8 it/sec)
- Need either: CUDA 11 cuBLAS libs, or ONNX Runtime built for CUDA 12


### Attempt 4 - Failed
- Upgraded ONNX Runtime to latest (1.23.2) which supports CUDA 12
- **PROGRESS**: GPU is now being accessed! (GPU=0 in logs)
- **NEW PROBLEM**: cuDNN compute capability mismatch
- Error: `no kernel image is available for execution on the device`
- Error: `CUDNN_FE failure 11: CUDNN_BACKEND_API_FAILED`
- Root cause: ONNX Runtime's cuDNN kernels not compiled for Tesla P4 (compute capability 6.1)
- Tried disabling cuDNN frontend API with env vars - no effect
- This is a fundamental binary incompatibility issue


### Attempt 5 - Expert Guidance (In Progress)
**Root Cause Identified**:
- Tesla P4 = Pascal architecture (Compute Capability 6.1)
- Modern ONNX Runtime 1.21+ defaults to CUDA 12 configs that conflict with Pascal
- Library shadowing: Any `onnxruntime` (CPU) package causes silent CPU fallback

**Solution Strategy (from expert guidance)**:
1. ✓ Clean slate: Uninstall ALL onnxruntime/onnx packages to prevent shadowing
2. Install onnxruntime-gpu==1.21.0 (tested stable for CUDA 12.x + Pascal)
3. Explicit provider config: `providers=['CUDAExecutionProvider'], provider_options=[{'device_id': 0}]`
4. Set environment: `LD_LIBRARY_PATH` to include CUDA/cuDNN paths
5. Disable graph optimization if needed (Pascal workaround): `graph_optimization_level = ORT_DISABLE_ALL`
6. Run dual-GPU parallel: separate scripts with device_id 0 and 1

**Expected Performance**:
- Single P4: 400-600 img/sec (vs current 5-8 CPU)
- Dual P4 parallel: <20 minutes for 500k images (vs 80+ hours CPU)

**Progress**:
- Step 1 ✓: Uninstalled onnxruntime, onnxruntime-gpu, onnx
- Step 2 ✓: Installed onnxruntime-gpu==1.21.0
- Step 3 ✓: Created fixed script with explicit provider configuration
- Step 4 ✗: FAILED - Still cuDNN errors with 1.21.0
- Step 5 ✗: FAILED - Tried 1.18.0, looking for CUDA 11 libs
- Step 6 ✗: FAILED - Created symlinks for CUDA 11 libs, dynamic linker not finding them

**Final Status**: BLOCKED
- All ONNX Runtime versions 1.15-1.23 either:
  - Look for CUDA 11 libraries (libcublasLt.so.11) that don't exist
  - Find CUDA 12 but hit cuDNN compute capability errors (Pascal/sm_61 not supported)
- Symlink workaround failed: Dynamic linker unable to find libcublasLt.so.11 despite:
  - Symlinks created in /usr/local/cuda/lib64 and /usr/local/cuda/targets/x86_64-linux/lib
  - ldconfig run multiple times
  - LD_LIBRARY_PATH explicitly set
  - Symlinks verified to exist

**Root Cause**: Pre-built ONNX Runtime binaries incompatible with Tesla P4 (Pascal, Compute Capability 6.1)

**Possible Solutions (not yet attempted)**:
1. Build ONNX Runtime from source for CUDA 12 + Pascal
2. Use MXNet backend for InsightFace instead of ONNX Runtime
3. Use different face recognition library (not InsightFace)
4. ~~Accept CPU processing (80+ hours for 523k images)~~ ❌ CPU IS NEVER AN OPTION


### Attempt 6 - Symlink Issue Identified (2025-12-30 after power loss)
- Created symlinks for all CUDA 11 libs: libcublas.so.11, libcublasLt.so.11, libcufft.so.10, libcudart.so.11.0
- **NEW DISCOVERY**: Symlinks found but version symbols mismatch
- Error: `version 'libcublas.so.11' not found` - CUDA 12 libs have different symbol versions
- ONNX Runtime 1.18.0 compiled for CUDA 11, expects CUDA 11 version symbols
- Symlinks don't work because internal version tags are incompatible

### Attempt 7 - Installing CUDA 11 Libraries (IN PROGRESS)
- Will install actual CUDA 11 cuBLAS/cuDNN libraries alongside CUDA 12
- This should provide the correct version symbols ONNX Runtime expects
- Target packages: libcublas-11-8, cuda-cudart-11-8


---

## Ecosystem State Discovery (2025-12-30 22:10)

**CRITICAL FINDING: InsightFace GPU Acceleration Has NEVER Worked**

### Docker Image Analysis

**Image ddd616452217** (Dec 28 12:28 PM - base image):
- Built from: Dockerfile + constraints.txt
- Contains: TensorFlow 2.16.1-gpu, DeepFace 0.0.96
- **MISSING**: InsightFace, onnxruntime-gpu, FAISS
- Container a5bdc2d2c544 currently running from this image

**Image 43ff5d6a806f** (Dec 30 10:55 AM - deepface-gpu:latest):
- Rebuilt version
- Contains: TensorFlow 2.16.1-gpu, DeepFace 0.0.96
- **MISSING**: InsightFace, onnxruntime-gpu, FAISS
- NOT running, never tested

**Running Container a5bdc2d2c544**:
- Started: Dec 28 12:41 PM (running 2 days)
- Base: Image ddd616452217
- Modifications (ALL done manually in running container):
  - Dec 30 21:51: onnxruntime-gpu 1.18.0 installed (Attempt 8)
  - Dec 30 21:57: onnxruntime-gpu 1.19.2 installed from CUDA 12 repo (Attempt 9)
  - Dec 30 21:58: InsightFace 0.7.3 installed
- **PROBLEM**: None of these modifications are in the Dockerfile - NOT REPRODUCIBLE

### Dockerfile Reality

**What's Actually Built**:
```dockerfile
FROM tensorflow/tensorflow:2.16.1-gpu
RUN pip install --constraint constraints.txt deepface tf-keras
```

**What's MISSING**:
- InsightFace installation
- ONNX Runtime configuration
- FAISS installation
- Any GPU acceleration setup for InsightFace

### Git Repository State

**Last Commit**: Dec 28 3:19 PM - "Add comprehensive README with full documentation"

**Uncommitted Changes**:
- Modified: README.md (added InsightFace/FAISS features)
- Modified: scripts/gender_filter.py
- Untracked: GPU_TROUBLESHOOTING_LOG.md
- Untracked: 9 new scripts including generate_embeddings.py, cluster_faces.py

**Timeline Mismatch**:
- README committed: Dec 28 3:19 PM (claims "~210 img/sec per GPU")
- Base image built: Dec 28 12:28 PM (BEFORE README commit, NO InsightFace)
- Scripts created: Dec 28-29 (generate_embeddings.py exists)
- InsightFace first installed: Dec 30 21:58 (TODAY, during troubleshooting)

### Evidence Files

**Embedding Output Files** (from Dec 29):
- /mnt/win_share/faceswap/embeddings_gpu0.pkl (222 bytes)
- /mnt/win_share/faceswap/embeddings_gpu1.pkl (222 bytes)
- /mnt/win_share/faceswap/embeddings_all.pkl (222 bytes)

**Analysis**: Real embeddings for 523k images would be 200+ MB. These 222-byte files are empty/failed runs.

### Conclusions

1. **InsightFace GPU acceleration has NEVER worked** - InsightFace was never in the Docker images
2. **README is aspirational documentation** - documents theoretical/planned features, not tested reality
3. **All work is non-reproducible** - modifications only exist in running container
4. **Git repo is out of sync** - scripts exist but aren't committed, README doesn't match Dockerfile
5. **Performance claims are theoretical** - "~210 img/sec" was never measured
6. **Failed embedding runs** - 222-byte files from Dec 29 show it was attempted but failed

### Recommendation

Before continuing troubleshooting:
1. Update Dockerfile to actually install InsightFace
2. Commit all work to git (scripts, troubleshooting log, README updates)
3. Build fresh image from updated Dockerfile
4. THEN continue GPU acceleration troubleshooting with reproducible baseline

---

## Attempt 8 - Gemini's GPU Acceleration Recovery Plan (2025-12-30)

**Source**: Gemini 2.5 Pro advice from session 2025-12-30T20:42

The core issue is InsightFace GPU acceleration compatibility with Tesla P4 (sm_61), CUDA 12.3, and ONNX Runtime.

### Gemini's Recommendation:

1. **Investigate MXNet Backend for InsightFace:** Check `deepface`/`insightface` for MXNet support and compatible MXNet-GPU packages.
2. **Explore DeepFace with TensorFlow:** Confirm TensorFlow's CUDA/cuDNN compatibility in your container and try running DeepFace with its TensorFlow backend.
3. **Examine the README:** Provide the path to the README claiming 200+ images/sec on P4s. This is critical for understanding the original successful setup.
4. **Building ONNX Runtime from Source:** Reserve this as a last resort due to its complexity.

### Detailed Implementation Plan:

**Goal:** Enable GPU acceleration for face embedding extraction to process 500,000 images efficiently.

#### Step 1: Create a Clean Environment

To prevent conflicts, remove all existing versions of ONNX and its runtime:

```bash
pip uninstall -y onnxruntime onnxruntime-gpu onnx
```

#### Step 2: Install a Compatible `onnxruntime-gpu` Version

Newer versions of `onnxruntime-gpu` have dropped support for the Pascal architecture (Tesla P4). Install version `1.18.0` - a robust choice for systems with CUDA 11.x or 12.x:

```bash
pip install onnxruntime-gpu==1.18.0 onnx
```

#### Step 3: Write a Verification Script

Create `verify_gpu.py` to test if the GPU is being correctly identified and used by InsightFace:

```python
import insightface
import numpy as np
import onnxruntime as ort
import cv2

print(f"ONNX Runtime version: {ort.__version__}")
print(f"ONNX Runtime providers: {ort.get_available_providers()}")

try:
    # Explicitly configure for GPU 0
    # This is the most critical step.
    app = insightface.app.FaceAnalysis(
        providers=['CUDAExecutionProvider'],
        provider_options=[{'device_id': 0}]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Create a dummy image to test inference
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = app.get(dummy_image)

    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print("\nSUCCESS: InsightFace is configured with CUDAExecutionProvider.")
        print(f"Processed dummy image. Found {len(faces)} faces.")
        print("GPU should be active. Check nvidia-smi to confirm.")
    else:
        print("\nFAILURE: CUDAExecutionProvider is not available.")

except Exception as e:
    print(f"\nAn error occurred during GPU initialization: {e}")
    print("Please check your CUDA/cuDNN installation and driver compatibility.")
```

#### Step 4: Execute and Verify

Run the script and monitor your GPU usage:

1. In one terminal, run `watch nvidia-smi` to monitor the GPUs.
2. In another terminal, execute the verification script: `python verify_gpu.py`.

You should see a Python process appear in the `nvidia-smi` output, consuming memory on GPU 0. The script's output should confirm that the `CUDAExecutionProvider` is available and used.

#### Next Steps: Parallel Processing

Once verified, you can proceed with the large-scale clustering. To use both Tesla P4s, run two instances of your processing script, modifying the `device_id` in the `provider_options`:

- **Script 1:** `provider_options=[{'device_id': 0}]` (for the first GPU)
- **Script 2:** `provider_options=[{'device_id': 1}]` (for the second GPU)

This will allow you to process the 500,000 images in parallel, significantly reducing the total time.

**Execution Results (2025-12-30 21:51)**:
- Step 1 ✓: Successfully uninstalled onnxruntime-gpu 1.18.0, onnx 1.20.0
- Step 2 ✓: Successfully installed onnxruntime-gpu==1.18.0, onnx 1.20.0
- Step 3 ✓: Created verify_gpu.py script in /app/ directory
- Step 4 ✗: **FAILED** - Same error as Attempt 5

**Error**: `libcufft.so.10: cannot open shared object file: No such file or directory`

**Analysis**:
- ONNX Runtime 1.18.0 HAS CUDAExecutionProvider available
- But falls back to CPUExecutionProvider due to missing CUDA 11 libraries
- Symlinks exist (libcufft.so.10 -> libcufft.so.11) but don't work
- Root cause: Version symbol mismatch between CUDA 11 and CUDA 12 libraries
- ldconfig cache doesn't recognize .so.10 even after update

**Conclusion**: Gemini's onnxruntime-gpu==1.18.0 approach fails for same reason as Attempt 5.

---

## Attempt 9 - Alternative Backends (IN PROGRESS)

**Status**: Gemini's recommendations #1 and #2 suggest alternative backends:
1. **MXNet Backend for InsightFace**
2. **TensorFlow Backend for DeepFace**

These avoid ONNX Runtime entirely, potentially resolving the CUDA version incompatibility.

**Execution Results (2025-12-30 21:56-22:03)**:

### Gemini Consultation #1 (21:56) - CUDA 12 Repository Solution

**Query**: "Context: I'm trying to get GPU acceleration working for InsightFace on Tesla P4 GPUs (Pascal, compute capability 6.1) with CUDA 12.3. Your previous recommendation was to use onnxruntime-gpu==1.18.0, but it failed with 'libcufft.so.10: cannot open shared object file' - it's looking for CUDA 11 libraries but the container has CUDA 12.3. Symlinks don't work due to version symbol mismatch. You also suggested: 1) MXNet backend for InsightFace, 2) TensorFlow backend for DeepFace, 3) Examine README about 200+ img/sec on P4s, 4) Build ONNX Runtime from source. Which approach should I try next and what are the exact steps?"

**Gemini Response (verbatim)**:
```
Based on your situation, here is a recommended sequence of approaches, from most likely to succeed with the least effort, to more involved alternatives. The error with `libcufft.so.10` is unusual for recent `onnxruntime-gpu` versions and suggests a dependency conflict, so we'll start by addressing that.

### Approach 1: Clean Re-installation of ONNX Runtime

A conflict in your Python environment is the most probable cause of the error. The best first step is to create a clean environment and install `onnxruntime-gpu` using a specific package index that provides builds aligned with CUDA 12.

I'll now provide the commands to perform a clean re-installation of ONNX Runtime. This will create a new virtual environment to avoid any conflicts with existing packages, clear the pip cache, and then install `onnxruntime-gpu` from a source that has better support for CUDA 12.x.

**2. Install `onnxruntime-gpu` for CUDA 12:**

Next, run this command to install `onnxruntime-gpu` from a dedicated repository that contains builds compatible with CUDA 12.x:

```bash
pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ onnxruntime-gpu
```

**3. Install `insightface`:**

```bash
pip install insightface
```

**4. Verify the installation:**

After the installation is complete, run the following Python code to verify that `onnxruntime` can access the GPU:

```python
import onnxruntime
print(onnxruntime.get_available_providers())
# This should output among other providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

If the `CUDAExecutionProvider` is listed, the installation was successful. Try running your InsightFace application again. If it still fails, we can proceed to the other approaches.
```

**Actions Taken**:
- Uninstalled: onnxruntime-gpu 1.18.0, insightface 0.7.3
- Cleared pip cache
- Installed from CUDA 12 repo: onnxruntime-gpu 1.19.2, onnx 1.20.0
- Reinstalled insightface 0.7.3

**Result**:
- **PROGRESS**: No more "libcufft.so.10" errors! CUDA 12 libraries now found.
- **PROGRESS**: CUDAExecutionProvider is ACTIVE (not falling back to CPU)
- **NEW PROBLEM**: cuDNN execution error during actual computation
- Error: `CUDNN_STATUS_EXECUTION_FAILED_CUDART` in cudnnConvolutionForward

### Gemini Consultation #2 (21:59) - HEURISTIC Algorithm Solution

**Query**: "Good news: Using onnxruntime-gpu 1.19.2 from the CUDA 12 repository solved the libcufft.so.10 error. CUDAExecutionProvider is now ACTIVE (not falling back to CPU). However, I'm getting a new error during actual GPU computation: 'CUDNN failure 5003: CUDNN_STATUS_EXECUTION_FAILED_CUDART' when running cudnnConvolutionForward. This happens on Tesla P4 (Pascal, sm_61) with CUDA 12.3. The error occurs in FusedConv node during InsightFace model inference. What's the next step to fix this cuDNN execution error?"

**Gemini Response (verbatim)**:
```
This cuDNN error, especially `CUDNN_STATUS_EXECUTION_FAILED_CUDART`, often points to an incompatibility between the GPU architecture (your Tesla P4 is Pascal, `sm_61`), the CUDA version, and the cuDNN library version being used by ONNX Runtime. The Pascal architecture is a bit older, and newer CUDA/cuDNN versions can sometimes have issues.

Let's try a common solution for this kind of problem, which is to adjust the convolution algorithm that cuDNN uses. ONNX Runtime exposes an environment variable to control this.

That `CUDNN_STATUS_EXECUTION_FAILED_CUDART` error (5003) on a Pascal-architecture GPU like the Tesla P4 is a strong indicator of an incompatibility between the modern CUDA/cuDNN version and the older hardware. Essentially, the default convolution algorithm that cuDNN is trying to use is likely failing on your GPU.

The most common and least disruptive way to fix this is to change the algorithm selection method. ONNX Runtime provides an environment variable to control this. Instead of the default exhaustive search, you can tell it to use a safer heuristic approach.

Please try running your inference script again, but this time, prepend your command with `ORT_CUDA_CONV_ALGO_SEARCH=HEURISTIC`.

For example, if your command is `python run_inference.py`, you should run:

```bash
ORT_CUDA_CONV_ALGO_SEARCH=HEURISTIC python run_inference.py
```

Please replace `python run_inference.py` with the actual command you use. Let me know if that resolves the error. If it doesn't, the next step would be to investigate downgrading to an ONNX Runtime build for CUDA 11.8, which has a long history of stability with Pascal GPUs.
```

### Gemini Consultation #3 (22:00) - Provider Options Key

**Query**: "What is the exact key name to set the cuDNN convolution algorithm in ONNX Runtime CUDAExecutionProvider options? I need to add it to provider_options=[{'device_id': 0, ???}]. Just give me the key-value pair like 'key_name': 'HEURISTIC'"

**Gemini Response (verbatim)**:
```
`'cudnn_conv_algo_search': 'HEURISTIC'`
```

**Actions Taken**:
- Environment variable ORT_CUDA_CONV_ALGO_SEARCH=HEURISTIC tried - not picked up by ONNX Runtime
- Added `'cudnn_conv_algo_search': 'HEURISTIC'` to provider_options dictionary
- Setting confirmed in output: `'cudnn_conv_algo_search': 'HEURISTIC'`

**Result**:
- **FAILED**: Still getting same cuDNN 5003 error with HEURISTIC setting

**Analysis**:
- Successfully resolved CUDA library version mismatch (11 vs 12)
- CUDAExecutionProvider is working and trying to use GPU
- But cuDNN kernels failing on Pascal (sm_61) + CUDA 12.3 combination
- HEURISTIC algorithm selection doesn't help
- Root cause: Modern cuDNN (CUDA 12.x era) may not support Pascal GPU architecture

**Next Steps** (from Gemini's original recommendations):
1. ❌ ~~onnxruntime-gpu from CUDA 12 repo~~ (tried, cuDNN incompatibility)
2. ⏳ **Examine README** about 200+ img/sec on P4s (may reveal working configuration)
3. ⏳ MXNet backend for InsightFace
4. ⏳ TensorFlow backend for DeepFace
5. ⏳ Build ONNX Runtime from source for CUDA 12 + Pascal

---

## Google AI Conversation - Alternative Approaches (2025-12-28)

**Context**: Prior conversation with Google AI discussing gender classification and face clustering approaches for 500k images on 2x Tesla P4 GPUs. This conversation occurred BEFORE the current GPU troubleshooting session.

### Key Recommendations from Google AI:

**Model Options Discussed**:
1. **ResNet-152**: 99% accuracy on large facial datasets
2. **FaceNet512**: 97.5% female, 94.2% male accuracy using face embeddings
3. **Vision Transformer (ViT-Base)**: 94.3% accuracy on UTKFace dataset, best for handling occlusions
4. **InceptionV3**: Efficient for production pipelines
5. **ArcFace**: Robust for specialized datasets, effective with face vectors

**For Occlusions and Strange Angles**:
Google AI recommended **Vision Transformer (ViT-Base)** over ResNet-50:
- Reason: ViTs use self-attention to handle occlusions (microphones, eating, hands)
- CNNs (like ResNet) look for local patterns; if mouth is covered, CNN may fail
- ViTs treat image as patches and use self-attention to understand relationships between all face parts
- Can classify based on eyes, forehead, jawline even if mouth is covered

**Tesla P4 Optimization**:
- INT8 Precision: P4 optimized for INT8 (21 TOPs performance)
- Use TensorRT to convert models to INT8 engines for 3x speedup
- Single P4 can process 1,500-2,000 images/sec with TensorRT INT8
- Two P4s in parallel: 500k images in 2-3 minutes

**Face Clustering Approach (Unsupervised)**:
Since images are unlabeled, Google AI recommended:
1. Generate facial embeddings using **InsightFace (ArcFace)**
2. Use **K-Means clustering** with FAISS (GPU-accelerated)
3. Split into Person A and Person B automatically
4. Expected speed: ~400-600 images/sec per P4 for embedding generation

**Deployment Pipeline**:
1. Export fine-tuned PyTorch ViT to ONNX format
2. Convert ONNX to TensorRT INT8 engine using `trtexec`
3. Multi-GPU deployment: spawn one worker per GPU using Multiprocessing or DDP
4. Batch size: 64-128 to saturate 8GB VRAM
5. Alternative: Use NVIDIA FasterTransformer for peak ViT performance

### Google AI's Final Response to Claude's ONNX Runtime Problem:

**User's Question to Google AI**:
"from claude: The problem I'm stuck on is that InsightFace uses ONNX Runtime for model inference, and I cannot get ONNX Runtime to use the Tesla P4 GPUs. Every version I try either: 1. Falls back to CPU (5-8 images/sec = 80+ hours) 2. Tries to use GPU but hits cuDNN compute capability errors"

**Google AI Response (Dec 28, 2025)**:
```
The issue you are facing is a common mismatch between ONNX Runtime (ORT) GPU, CUDA/cuDNN versions, and the Tesla P4's Pascal architecture (Compute Capability 6.1). As of late 2025, modern ORT versions (1.21+) often default to CUDA 12 configurations that can conflict with older Pascal cards if not precisely aligned.

1. The "Clean Slate" Fix (Essential)
ONNX Runtime will silently fall back to CPU if any version of the standard onnxruntime (CPU) package is present.

Action: Uninstall everything first to prevent library shadowing.
pip uninstall -y onnxruntime onnxruntime-gpu onnx

2. Target Version Matching (2025 Best Practice)
For a Tesla P4 (Pascal), the most stable path in 2025 is often using CUDA 11.8 or CUDA 12.1 with specific ORT builds.

Recommended Combo: Use onnxruntime-gpu==1.18.0 or 1.21.0 if you are on CUDA 12.x.
Installation:
pip install onnxruntime-gpu==1.21.0

3. Explicit Provider Configuration
InsightFace's default initialization often fails to pass the correct device ID to the CUDA provider.

Action: Modify your FaceAnalysis initialization to target a specific P4.
import insightface
# For GPU 0
app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider'], provider_options=[{'device_id': 0}])
app.prepare(ctx_id=0, det_size=(640, 640))

4. System Environment Paths
If you still see "cuDNN" errors, ORT likely can't find the .so or .dll files for your drivers.

Linux Fix: Ensure your LD_LIBRARY_PATH includes both CUDA and cuDNN.
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/include:$LD_LIBRARY_PATH

5. Disable Graph Optimization (The "Pascal" Workaround)
Older architectures like the P4 sometimes crash during the "graph optimization" phase of modern ORT.

Action: Disable all optimizations to see if it starts running.
import onnxruntime as ort
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

Why this fixes your 80+ hour problem:
- Throughput: Once the P4 is properly engaged (visible via nvidia-smi), your speed will jump from 5-8 img/sec to roughly 400-600 img/sec per card.
- Dual GPU: Since you have two P4s, run two separate Python scripts—one with device_id: 0 and one with device_id: 1—to process your 500k images in parallel.
- Your total time should drop to under 20 minutes.
```

**Analysis**: Google AI's recommendations align with Gemini's (disable graph optimization, use specific ONNX Runtime versions, explicit provider configuration). However, Google AI specifically mentions onnxruntime-gpu==1.21.0 as stable for Pascal + CUDA 12.x, which we already tried in Attempt 5 and it failed. The graph optimization disabling was not tried yet.

