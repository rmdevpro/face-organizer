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

# Upgrade pip and install from local packages
RUN pip install --upgrade pip && \
    pip install --no-index --find-links=/tmp/packages/ \
    blinker deepface opencv-python-headless pillow tqdm && \
    rm -rf /tmp/packages/

# Create workspace and scripts directory
WORKDIR /workspace
COPY scripts/ /app/scripts/

CMD ["/bin/bash"]
