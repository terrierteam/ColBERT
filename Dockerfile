FROM ufoym/deepo:torch-cu100
#FROM pytorch/conda-cuda

ARG DEBIAN_FRONTEND=noninteractive

RUN wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh

RUN chmod 770 ./Anaconda3-5.3.1-Linux-x86_64.sh

ARG DEBIAN_FRONTEND=noninteractive

RUN yes yes | ./Anaconda3-5.3.1-Linux-x86_64.sh

ENV PATH /yes/bin/:$PATH

RUN conda config --add channels conda-forge \
 && conda config --add channels pytorch \
 && conda update conda

COPY conda_environment.txt /tmp/

RUN conda env create --name pythonColbert -f /tmp/conda_environment.txt

#SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
