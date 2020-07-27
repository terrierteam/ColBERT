FROM continuumio/anaconda3
#FROM pytorch/conda-cuda

ARG DEBIAN_FRONTEND=noninteractive

COPY conda_environment.txt /tmp/

RUN conda config --add channels conda-forge \
 && conda config --add channels pytorch \
 && conda update conda \
 && conda update -n base -c defaults conda \
 && conda env create --name pythonColbert -f /tmp/conda_environment.txt

RUN apt-get update \
 && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates \
 && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - \
 && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
 && echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list \
 && apt-get purge --autoremove -y curl \
 && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION=10.1.243
ENV CUDA_PKG_VERSION=10-1=10.1.243-1

RUN apt-get update \
 && apt-get install -y --no-install-recommends         cuda-cudart-$CUDA_PKG_VERSION cuda-compat-10-1 \
 && ln -s cuda-10.1 /usr/local/cuda \
 &&     rm -rf /var/lib/apt/lists/*

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
 && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_REQUIRE_CUDA=cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411
ENV NCCL_VERSION=2.4.8

RUN apt-get update \
 && apt-get install -y --no-install-recommends     cuda-libraries-$CUDA_PKG_VERSION cuda-nvtx-$CUDA_PKG_VERSION libcublas10=10.2.1.243-1 libnccl2=$NCCL_VERSION-1+cuda10.1 \
 && apt-mark hold libnccl2 \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
 && apt-get install -y --no-install-recommends         cuda-nvml-dev-$CUDA_PKG_VERSION         cuda-command-line-tools-$CUDA_PKG_VERSION cuda-libraries-dev-$CUDA_PKG_VERSION         cuda-minimal-build-$CUDA_PKG_VERSION         libnccl-dev=$NCCL_VERSION-1+cuda10.1 libcublas-dev=10.2.1.243-1 \
 && rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
ENV CUDNN_VERSION=7.6.5.32

RUN apt-get update \
 && apt-get install -y --no-install-recommends     libcudnn7=$CUDNN_VERSION-1+cuda10.1 libcudnn7-dev=$CUDNN_VERSION-1+cuda10.1 \
 && apt-mark hold libcudnn7 \
 && rm -rf /var/lib/apt/lists/*

ENV LANG=C.UTF-8

RUN  echo "deb http://archive.ubuntu.com/ubuntu bionic multiverse" >>  /etc/apt/sources.list \
 && echo "deb http://archive.ubuntu.com/ubuntu bionic-security multiverse" >>  /etc/apt/sources.list \
 && echo "deb http://archive.ubuntu.com/ubuntu bionic-updates multiverse" >>  /etc/apt/sources.list \
 && echo "deb http://archive.ubuntu.com/ubuntu bionic universe" >>  /etc/apt/sources.list \
 && echo "deb http://archive.ubuntu.com/ubuntu bionic-security universe" >>  /etc/apt/sources.list \
 && echo "deb http://archive.ubuntu.com/ubuntu bionic-updates universe" >>  /etc/apt/sources.list \
 && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32 \
 && apt update

RUN conda run -n pythonColbert /bin/bash -c "pip install -y torch==1.4.0"

#COPY conda_environment.txt /tmp/

#RUN conda env create --name pythonColbert -f /tmp/conda_environment.txt

#SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
