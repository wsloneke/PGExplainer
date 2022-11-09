#!/bin/bash
# SBATCH --nodes 1
# SBATCH -c 16
# SBATCH -p batch
# SBATCH --time 48:00:00
# SBATCH --mem-per-cpu 6G
# SBATCH --job-name network
# SBATCH --mail-type=ALL
# SBATCH --mail-user=whitney_sloneker@brown.edu

hostname -i
python testk.py
