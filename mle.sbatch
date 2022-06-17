#!/bin/bash

#SBATCH --job-name=minib22
#SBATCH --output=mini.out
#SBATCH --error=mini.err
#SBATCH --time=06:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=84
#SBATCH --mem-per-cpu=2000
#SBATCH --constraint=fdr

# Load Python modules
module load python/anaconda-2020.02

python para.py 