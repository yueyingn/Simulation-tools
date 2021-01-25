#!/bin/bash
#SBATCH -p development
#SBATCH -N 1
#SBATCH --ntasks-per-node 2
#SBATCH --job-name=plot
##SBATCH --output=%x-%j.out
##SBATCH --error=%x-%j.err
#SBATCH --time=00:30:00

hostname; pwd; date
  
for i in 008 010 011 013 015 017 018 019 020 021;
do
    python gas4Mpc.py $i
done

date
