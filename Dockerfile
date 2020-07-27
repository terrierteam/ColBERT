FROM continuumio/anaconda3
#FROM pytorch/conda-cuda

ARG DEBIAN_FRONTEND=noninteractive

COPY conda_environment.txt /tmp/

RUN conda config --add channels conda-forge \
 && conda config --add channels pytorch \
 && conda update conda \
 && conda update -n base -c defaults conda \
 && conda env create --name pythonColbert -f /tmp/conda_environment.txt

#COPY conda_environment.txt /tmp/

#RUN conda env create --name pythonColbert -f /tmp/conda_environment.txt

#SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
