#!/bin/bash

#SBATCH --job-name=minib22
#SBATCH --output=mini.out
#SBATCH --error=mini.err
#SBATCH --time=24:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000
#SBATCH --constraint=fdr

# Load Python modules
module load python/anaconda-2020.02

python para.py 