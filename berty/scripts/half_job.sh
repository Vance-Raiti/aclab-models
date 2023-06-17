#!/bin/bash -l

#trains BERT with half the layers

# Request 4 CPUs
#$ -pe omp 4

# Request 1 GPU
#$ -l gpus=1
# force cuda capacity 7.0 (basically, ask for a relatively new gpu)
#$ -l gpu_c=7.0
# currently off (previous line already does this), but would  force gpu memory > 16GB
##$ -l gpu_memory=16G

#specify a project
#$ -P aclab

#$ -o /projectnb/aclab/vraiti/nn/basic_transformer/logs/half.out

#$ -e /projectnb/aclab/vraiti/nn/basic_transformer/logs/half.err

cd /projectnb/aclab/vraiti
source start.sh
cd nn/basic_transformer
python3 train.py 6 half_model
