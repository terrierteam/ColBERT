#!/bin/bash

conda activate pythonColbert
pip install -y torch==1.4.0
python -m src.train
