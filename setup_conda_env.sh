#!/bin/bash

if [[ $# == 1 ]]; then 
    conda create -n $1 python=3 ipython-notebook --yes
    source activate $1
    conda install -c bioconda pyfaidx
	conda install -c conda-forge jupyterlab
	conda install -c anaconda scikit-learn
	pip install --upgrade tensorflow
else
    echo "usage: $0 <conda environment name>"
    exit 1
fi
