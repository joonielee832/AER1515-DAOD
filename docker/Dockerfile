FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04
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
RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113

#? Install detectron2
RUN pip3 install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

WORKDIR /home