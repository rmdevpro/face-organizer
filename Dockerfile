FROM tensorflow/tensorflow:2.16.1-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy downloaded packages and version constraints
COPY packages/ /tmp/packages/
COPY constraints.txt /tmp/constraints.txt

# Install DeepFace with strict version constraints to prevent breaking TensorFlow GPU support
# constraints.txt locks ALL base image packages to their exact versions
RUN pip install --upgrade pip && \
    pip install --find-links=/tmp/packages/ --constraint /tmp/constraints.txt --prefer-binary \
    --ignore-installed blinker \
    deepface tf-keras && \
    rm -rf /tmp/packages/ /tmp/constraints.txt

# Create workspace and scripts directory
WORKDIR /workspace
COPY scripts/ /app/scripts/

CMD ["/bin/bash"]
