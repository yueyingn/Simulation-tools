#!/bin/bash
#SBATCH -p development
#SBATCH -N 20
#SBATCH --ntasks-per-node 2
#SBATCH --job-name=mpitest
#SBATCH --output=%x-%j.out
#SBATCH --time=00:30:00

hostname; pwd; date

module load impi/19.0.5

export OMP_NUM_THREADS=1

mpirun -n 40 python gasID2grid_mpi.py


date

