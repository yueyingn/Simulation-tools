#!/bin/bash
#SBATCH -p development
#SBATCH -N 1
#SBATCH --ntasks-per-node 2
#SBATCH --job-name=paint
#SBATCH --time=00:30:00
#SBATCH --dependency=2203627

hostname; pwd; date

#read -r c1 c2 c3 <<<$(echo 23508 180176 395782)
read -r c1 c2 c3 <<<$(echo 278083 27826 249177) 
 
python gas-tzns-4Mpc.py $c1 $c2 $c3
python gas-tzns-40Mpc.py $c1 $c2 $c3
python dm-40Mpc.py $c1 $c2 $c3
python dm-4Mpc.py $c1 $c2 $c3

date

