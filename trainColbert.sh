#!/bin/bash

export CONDA_DEFAULT_ENV="base"
export CONDA_EXE="/opt/conda/bin/conda"
export CONDA_PREFIX="/opt/conda"
export CONDA_PROMPT_MODIFIER="(base) "
export CONDA_PYTHON_EXE="/opt/conda/bin/python"
export CONDA_SHLVL="1"
export PATH=/opt/conda/bin:/opt/conda/condabin:$PATH

source /etc/profile.d/conda.sh
conda activate pythonColbert
pip install torch==1.4.0
python -m src.train

#conda run -v -n pythonColbert pip install torch==1.4.0
#conda run -v -n pythonColbert python -m src.train &>train.log
