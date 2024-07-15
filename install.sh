#!/bin/bash

# Ensure the script exits on any error
set -e

# # Install PyTorch with CUDA 11.8
pip3 install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html  

# Install PyTorch Geometric and related libraries
pip3 install torch_geometric
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install other dependencies
pip3 install ase==3.22.1 black==22.3.0 e3nn==0.4.4 matplotlib numba orjson \
             pre-commit==2.10.* pytest lmdb pyyaml submitit \
             syrupy==3.0.6 tensorboard tqdm wandb pymatgen==2023.5.10 requests
