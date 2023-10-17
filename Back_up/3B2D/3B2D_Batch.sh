#!/bin/bash

#SBATCH --job-name="Anis_thesis"
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=Education-AS-MSc-AP

module load 2023r1 python py-numpy py-pip py-matplotlib intel/oneapi-all

srun python 3B_2D_alt_2d_prec_linear_op.py > alt_prec.log