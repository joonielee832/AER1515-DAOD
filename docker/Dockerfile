FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
LABEL org.opencontainers.image.authors="john.lee@robotics.utias.utoronto.ca"

ENV TZ Canada/Eastern
ENV DEBIAN_FRONTEND noninteractive

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

#? Install detectron2
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git@v0.5'
RUN echo "export DETECTRON2_DATASETS=/home/data" >> ~/.bashrc

WORKDIR /home