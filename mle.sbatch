#!/bin/bash

#SBATCH --job-name=mini
#SBATCH --output=mini.out
#SBATCH --error=mini.err
#SBATCH --time=04:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=14
#SBATCH --mem-per-cpu=2000
#SBATCH --constraint=fdr

# Load Python modules
module load python/anaconda-2020.02

python para.py 