#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=math-454
#SBATCH --account=math-454
#SBATCH --mem=10G

module purge
module load gcc cuda openmpi/4.1.3-cuda  

srun VENV/bin/python training2.py --pretrain=bert-base-uncased
