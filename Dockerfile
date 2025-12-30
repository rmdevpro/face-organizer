FROM tensorflow/tensorflow:2.16.1-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy downloaded packages
COPY packages/ /tmp/packages/
COPY requirements.txt /tmp/requirements.txt

# Install minimal dependencies for face organization
# InsightFace (face embeddings), ONNX Runtime GPU (inference), FAISS (clustering)
# Use local packages where available, but allow PyPI for build dependencies
RUN pip install --upgrade pip && \
    pip install --find-links=/tmp/packages/ --prefer-binary \
    -r /tmp/requirements.txt && \
    rm -rf /tmp/packages/ /tmp/requirements.txt

# Create workspace directory
WORKDIR /workspace

# Scripts are hot-mounted from /mnt/projects/face-organizer/scripts (ADR-025 dev pattern)
# No COPY needed - changes to scripts in git repo are immediately available

CMD ["/bin/bash"]
