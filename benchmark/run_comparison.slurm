#!/bin/bash
#SBATCH --job-name=ridge_compare
#SBATCH --output=ridge_compare_%A_%a.out
#SBATCH --error=ridge_compare_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# Load modules
module load R

# Load required modules for GPU and GSL
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12

Rscript compare_large.R

