#!/bin/bash
#SBATCH -p normal 
#SBATCH -N 1
#SBATCH --ntasks-per-node 2
#SBATCH --job-name=plot
##SBATCH --output=%x-%j.out
##SBATCH --error=%x-%j.err
#SBATCH --time=1:00:00

hostname; pwd; date
  
for i in 001 002 007 008 010 011 013 015 017 018 019 020 021;
do
    python gas40Mpc.py $i
done

date
