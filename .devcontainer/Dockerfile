FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

USER root
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python 3.9
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && sed -i 's| main restricted| main restricted universe multiverse|g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    pkg-config \
    curl \
    git \
    # Python 3.9
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    # Image processing libraries
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libqhull-dev \
    libagg-dev \
    libcairo2-dev \
    libffi-dev \
    # Video processing and codecs
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libfdk-aac-dev \
    libmp3lame-dev \
    libopus-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    # OpenCV and GTK dependencies
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgtk-3-dev \
    libdc1394-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv and set up Python
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspaces/tracklab




