ARG CUDA_VERSION=12.2.2
ARG OS_VERSION=22.04
ARG CUDNN_VERSION=8

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${OS_VERSION}
LABEL org.opencontainers.image.authors="john.lee@robotics.utias.utoronto.ca"

ENV TZ Canada/Eastern
ENV DEBIAN_FRONTEND noninteractive
ENV CUDA_VERSION=${CUDA_VERSION}
ENV OS_VERSION=${OS_VERSION}
ENV CUDNN_VERSION=${CUDNN_VERSION}

SHELL ["/bin/bash", "-c"]

#? System Packages
COPY apt_packages.txt apt_packages.txt
RUN apt-get update && \
    xargs -a apt_packages.txt apt-get install -y --no-install-recommends && \
    rm apt_packages.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip;

#? Install basic python packages
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt && \
    rm requirements.txt

#? Install PyTorch
RUN pip3 install torch torchvision

WORKDIR /home