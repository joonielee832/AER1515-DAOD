FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04
LABEL org.opencontainers.image.authors="john.lee@robotics.utias.utoronto.ca"

ENV TZ Canada/Eastern
ENV DEBIAN_FRONTEND noninteractive

SHELL ["/bin/bash", "-c"]

#? System Packages
COPY apt_packages.txt apt_packages.txt
RUN apt-get update && \
    xargs -a apt_packages.txt apt-get install -y --no-install-recommends && \
    # apt-get install -y --no-install-recommends \
    #     build-essential git gcc pkg-config libpython3-dev python3-dev \
    #     python3-pip python3-wheel python3-opencv python3-tk ninja-build intel-mkl && \
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
RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

#? Install detectron2
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git@v0.5'
# RUN echo "export DETECTRON2_DATASETS=/home/data" >> ~/.bashrc
ENV DETECTRON2_DATASETS /home/data

WORKDIR /home