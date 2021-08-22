#!/bin/bash

#SBATCH --time=60:00:00
#SBATCH -p gpu --gres=gpu:5
#SBATCH -n 15
#SBATCH -N 1
#SBATCH --mem=100G
#SBATCH --account=carney-tserre-condo
#SBATCH -C quadrortx
#SBATCH -J contlearn

# Specify an output file
# #SBATCH -o cont1.out
# #SBATCH -e cont1.err

module load anaconda
source activate newtrck
module load gcc/8.3
module load cuda/10.2

python run.py --config-name=ccv_pf_multi.yaml -m model.name=int_softplus

