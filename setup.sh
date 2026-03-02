#! /bin/bash
# run the script where the setup.py locates with source command
# e.g. source scripts/docker/maskrcnn_setup.sh maskrcnn_env

env_name=$1

conda create -n ${env_name} python=3.8 -y
conda activate ${env_name}

# cudatoolkit version should be same as that of pytorch
# otherwise it will cause error compiling csrc
conda install gcc=9.5 gxx=9.5 cudatoolkit-dev=11.3 -c conda-forge -y
conda install pycocotools fvcore iopath ninjas absl-py setuptools=59.5.0 -c conda-forge -y
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 mkl=2024.0 -c pytorch -c conda-forge -y

python setup.py build develop
python setup.py clean --all

mkdir cache
wget https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl -P ./cache