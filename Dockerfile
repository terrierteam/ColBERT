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

RUN APT_INSTALL="apt-get install -y --no-install-recommends" \
 && PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" \
 && GIT_CLONE="git clone --depth 10" \
 &&     rm -rf /var/lib/apt/lists/*            /etc/apt/sources.list.d/cuda.list            /etc/apt/sources.list.d/nvidia-ml.list \
 &&     apt-get update \
 &&     DEBIAN_FRONTEND=noninteractive $APT_INSTALL         build-essential         apt-utils         ca-certificates         wget         git         vim         libssl-dev         curl         unzip         unrar         \
 &&     $GIT_CLONE https://github.com/Kitware/CMake ~/cmake \
 &&     cd ~/cmake \
 &&     ./bootstrap \
 &&     make -j"$(nproc)" install \
 &&     $GIT_CLONE https://github.com/pjreddie/darknet.git ~/darknet \
 &&     cd ~/darknet \
 &&     sed -i 's/GPU=0/GPU=1/g' ~/darknet/Makefile \
 &&     sed -i 's/CUDNN=0/CUDNN=1/g' ~/darknet/Makefile \
 &&     make -j"$(nproc)" \
 &&     cp ~/darknet/include/* /usr/local/include \
 &&     cp ~/darknet/*.a /usr/local/lib \
 &&     cp ~/darknet/*.so /usr/local/lib \
 &&     cp ~/darknet/darknet /usr/local/bin \
 &&     DEBIAN_FRONTEND=noninteractive $APT_INSTALL         software-properties-common         \
 &&     add-apt-repository ppa:deadsnakes/ppa \
 &&     apt-get update \
 &&     DEBIAN_FRONTEND=noninteractive $APT_INSTALL         python3.6         python3.6-dev         python3-distutils-extra         \
 &&     wget -O ~/get-pip.py         https://bootstrap.pypa.io/get-pip.py \
 &&     python3.6 ~/get-pip.py \
 &&     ln -s /usr/bin/python3.6 /usr/local/bin/python3 \
 &&     ln -s /usr/bin/python3.6 /usr/local/bin/python \
 &&     $PIP_INSTALL         setuptools         \
 &&     $PIP_INSTALL         numpy         scipy         pandas         cloudpickle         scikit-image>=0.14.2         scikit-learn         matplotlib         Cython         tqdm         \
 &&     DEBIAN_FRONTEND=noninteractive $APT_INSTALL         sudo         \
 &&     $GIT_CLONE https://github.com/nagadomi/distro.git ~/torch --recursive \
 &&     cd ~/torch \
 &&     bash install-deps \
 &&     sed -i 's/${THIS_DIR}\/install/\/usr\/local/g' ./install.sh \
 &&     ./install.sh \
 &&     DEBIAN_FRONTEND=noninteractive $APT_INSTALL         libboost-all-dev         \
 &&     $PIP_INSTALL         cupy         chainer         \
 &&     $PIP_INSTALL         jupyter         \
 &&     DEBIAN_FRONTEND=noninteractive $APT_INSTALL         libatlas-base-dev         graphviz         \
 &&     $PIP_INSTALL         mxnet-cu101         graphviz         \
 &&     DEBIAN_FRONTEND=noninteractive $APT_INSTALL         protobuf-compiler         libprotoc-dev         \
 &&     $PIP_INSTALL         --no-binary onnx onnx         \
 &&     $PIP_INSTALL         onnxruntime         \
 &&     $PIP_INSTALL         paddlepaddle-gpu         \
 &&     $PIP_INSTALL         future         numpy         protobuf         enum34         pyyaml         typing         \
 &&     $PIP_INSTALL         --pre torch torchvision -f         https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html         \
 &&     $PIP_INSTALL         tensorflow-gpu         \
 &&     DEBIAN_FRONTEND=noninteractive $APT_INSTALL         libblas-dev         \
 &&     wget -qO- https://github.com/Theano/libgpuarray/archive/v0.7.6.tar.gz | tar xz -C ~ \
 &&     cd ~/libgpuarray* \
 && mkdir -p build \
 && cd build \
 &&     cmake -D CMAKE_BUILD_TYPE=RELEASE           -D CMAKE_INSTALL_PREFIX=/usr/local           .. \
 &&     make -j"$(nproc)" install \
 &&     cd ~/libgpuarray* \
 &&     python setup.py build \
 &&     python setup.py install \
 &&     printf '[global]\nfloatX = float32\ndevice = cuda0\n\n[dnn]\ninclude_path = /usr/local/cuda/targets/x86_64-linux/include\n' > ~/.theanorc \
 &&     $PIP_INSTALL         https://github.com/Theano/Theano/archive/master.zip         \
 &&     $PIP_INSTALL         jupyterlab         \
 &&     $PIP_INSTALL         h5py         keras         \
 &&     $GIT_CLONE https://github.com/Lasagne/Lasagne ~/lasagne \
 &&     cd ~/lasagne \
 &&     $PIP_INSTALL         . \
 &&     DEBIAN_FRONTEND=noninteractive $APT_INSTALL         libatlas-base-dev         libgflags-dev         libgoogle-glog-dev         libhdf5-serial-dev         libleveldb-dev         liblmdb-dev         libprotobuf-dev         libsnappy-dev         protobuf-compiler         \
 &&     $GIT_CLONE --branch 4.3.0 https://github.com/opencv/opencv ~/opencv \
 &&     mkdir -p ~/opencv/build \
 && cd ~/opencv/build \
 &&     cmake -D CMAKE_BUILD_TYPE=RELEASE           -D CMAKE_INSTALL_PREFIX=/usr/local           -D WITH_IPP=OFF           -D WITH_CUDA=OFF           -D WITH_OPENCL=OFF           -D BUILD_TESTS=OFF           -D BUILD_PERF_TESTS=OFF           -D BUILD_DOCS=OFF           -D BUILD_EXAMPLES=OFF           .. \
 &&     make -j"$(nproc)" install \
 &&     ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2 \
 &&     $PIP_INSTALL         tensorflow_probability         "dm-sonnet>=2.0.0b0" --pre         \
 &&     apt-get update \
 &&     DEBIAN_FRONTEND=noninteractive $APT_INSTALL         caffe-cuda         \
 &&     DEBIAN_FRONTEND=noninteractive $APT_INSTALL         openmpi-bin         libpng-dev         libjpeg-dev         libtiff-dev         \
 &&     ln -s /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.20 /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.1 \
 &&     ln -s /usr/lib/x86_64-linux-gnu/libmpi.so.20.10.1 /usr/lib/x86_64-linux-gnu/libmpi.so.12 \
 &&     wget --no-verbose -O - https://github.com/01org/mkl-dnn/releases/download/v0.14/mklml_lnx_2018.0.3.20180406.tgz | tar -xzf - \
 &&     cp mklml*/* /usr/local -r \
 &&     wget --no-verbose -O - https://github.com/01org/mkl-dnn/archive/v0.14.tar.gz | tar -xzf - \
 &&     cd *-0.14 \
 && mkdir build \
 && cd build \
 &&     ln -s /usr/local external \
 &&     cmake -D CMAKE_BUILD_TYPE=RELEASE           -D CMAKE_INSTALL_PREFIX=/usr/local           .. \
 &&     make -j"$(nproc)" install \
 &&     $PIP_INSTALL         cntk-gpu         \
 &&     ldconfig \
 &&     apt-get clean \
 &&     apt-get autoremove \
 &&     rm -rf /var/lib/apt/lists/* /tmp/* ~/*

#COPY conda_environment.txt /tmp/

#RUN conda env create --name pythonColbert -f /tmp/conda_environment.txt

#SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
