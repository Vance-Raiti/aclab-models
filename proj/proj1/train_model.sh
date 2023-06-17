#!/bin/bash -l


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

#$ -e /projectnb/aclab/vraiti/train_err.out

#$ -o /projectnb/aclab/vraiti/train.out


source /projectnb/aclab/vraiti/start.sh
python3 /projectnb/aclab/vraiti/proj/train_model.py