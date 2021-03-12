#!/bin/bash
#SBATCH -p development
#SBATCH -N 8
#SBATCH --ntasks-per-node 1
#SBATCH --job-name=plot
##SBATCH --output=%x-%j.out
##SBATCH --error=%x-%j.err
#SBATCH --time=00:30:00

hostname; pwd; date

srun -N 1 -n 1 python gas-3d-batch.py 0 18 &> job1.out &
srun -N 1 -n 1 python gas-3d-batch.py 18 28 &> job2.out &
srun -N 1 -n 1 python gas-3d-batch.py 28 44 &> job3.out &
srun -N 1 -n 1 python gas-3d-batch.py 44 65 &> job4.out &
srun -N 1 -n 1 python gas-3d-batch.py 65 91 &> job5.out &
srun -N 1 -n 1 python gas-3d-batch.py 91 112 &> job6.out &
srun -N 1 -n 1 python gas-3d-batch.py 112 121 &> job7.out &
srun -N 1 -n 1 python gas-3d-batch.py 121 129 &> job9.out &

wait

date

