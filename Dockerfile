#FROM nvidia/cuda:10.2-base-ubuntu18.04
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    nano \
    libx11-6 \
    ffmpeg \ 
    libsm6 \
    libxext6 \
    python3.6 \
    python3.6-dev \
    python3-distutils \
    python3-apt \
 && rm -rf /var/lib/apt/lists/*  \
 && curl https://bootstrap.pypa.io/get-pip.py | python3.6 - --user

COPY requirements.txt .

RUN python3.6 -m pip install "torch==1.7.0+cu101" "torchvision==0.8.1+cu101" -f https://download.pytorch.org/whl/torch_stable.html \
&& python3.6 -m pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py36_cu101_pyt170/download.html \
&& python3.6 -m pip install -r requirements.txt

