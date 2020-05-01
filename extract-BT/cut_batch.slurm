#!/bin/bash
#SBATCH -p development
#SBATCH -N 20
#SBATCH --ntasks-per-node 1
#SBATCH --job-name=cut
#SBATCH --output=%x-%j.out
#SBATCH --time=2:00:00

hostname; pwd; date

for d in 39 48 53 57 80 114 126 129 130 176 177 192 233 307 310 346 359 372 463 494
##for d in 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
do
   srun -N 1 -n 1 -c 16 --exclusive python3 cut.py $d  &> job_$d.out &
done

wait
  

date

