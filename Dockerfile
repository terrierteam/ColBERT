FROM continuumio/anaconda3:latest
#FROM pytorch/conda-cuda

ARG DEBIAN_FRONTEND=noninteractive

RUN conda config --add channels conda-forge \
 && conda config --add channels pytorch \
 && conda update conda

COPY conda_environment.txt /tmp/

RUN conda env create --name pythonColbert -f /tmp/conda_environment.txt

#SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
