FROM ufoym/deepo:torch-cu100
#FROM pytorch/conda-cuda

ARG DEBIAN_FRONTEND=noninteractive

RUN wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh

RUN chmod 770 ./Anaconda3-5.3.1-Linux-x86_64.sh

ARG DEBIAN_FRONTEND=noninteractive

RUN yes yes | ./Anaconda3-5.3.1-Linux-x86_64.sh

ENV PATH /yes/bin/:$PATH

COPY conda_environment.txt /tmp/

RUN conda config --add channels conda-forge \
 && conda config --add channels pytorch \
 && conda install python=3.6.0 \
 && conda update conda \
 && conda env create --name pythonColbert -f /tmp/conda_environment.txt
# && conda update -n base -c defaults conda

#COPY conda_environment.txt /tmp/

#RUN conda env create --name pythonColbert -f /tmp/conda_environment.txt

#SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
