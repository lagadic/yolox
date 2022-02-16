ARG BASE_IMAGE=nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Nvidia hardware acceleration
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Install a tool to find a fast, up-to-date Ubuntu Archive Mirror.
RUN apt update && apt install --no-install-recommends -y python3-pip &&\
    pip3 install -U pip &&\
    pip3 install apt-mirror-updater &&\
    apt-mirror-updater -a -q &&\
    rm -rf /var/lib/apt/lists/*

# Tensorflow
RUN pip3 install -U pip &&\
    pip3 install -U tensorflow==2.7.0

# OpenCV
RUN apt-mirror-updater -a -q &&\
    apt install --no-install-recommends -y python3-opencv &&\
    pip3 install -U pip &&\
    pip3 install opencv-python &&\
    pip3 install pyyaml &&\
    rm -rf /var/lib/apt/lists/*

# LabelImg
RUN apt-mirror-updater -a -q &&\
    apt install --no-install-recommends -y pyqt5-dev-tools &&\
    pip3 install -U pip &&\
    pip3 install labelImg

# Model conversion
RUN pip3 install tf2onnx

RUN ldconfig
