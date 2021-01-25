#!/bin/bash
#SBATCH -A AST20015
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --job-name=extract-bh
#SBATCH --time=1:20:00
#SBATCH --mail-type=END
#SBATCH --mail-user=yueyingn@andrew.cmu.edu

hostname; pwd; date

for i in 019 029 038 045 046; do
    python extract-BH.py $i
done

#python extract-BH.py 064
