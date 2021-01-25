#!/bin/bash
#SBATCH -p development
#SBATCH -N 40
#SBATCH --ntasks-per-node 2
#SBATCH --job-name=cut-asterix
##SBATCH --output=%x-%j.out
##SBATCH --error=%x-%j.err
#SBATCH --time=1:00:00

hostname; pwd; date
  
module load impi/19.0.7

export OMP_NUM_THREADS=1

for i in 007 008 010 011 013 015; do
    mpirun -n 80 python gascut_mpi.py $i || exit 1
done

date
