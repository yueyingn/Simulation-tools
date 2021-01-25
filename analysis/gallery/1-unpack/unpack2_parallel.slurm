#!/bin/bash
#SBATCH -p normal
#SBATCH -N 2
#SBATCH --ntasks-per-node 1
#SBATCH --job-name=unpack
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=2:00:00

hostname; pwd; date
  
data_root_dir="/home1/06431/yueyingn/scratch/BT3/PART_271_pack"
out_dir="/home1/06431/yueyingn/scratch/BT3/"


#'1/Position':13 - 39
#'0/InternalEnergy':79 - 83

#'0/SmoothingLength':88 - 92
#'0/Metallicity': 92 - 96
#'0/Position':96 - 123

#'0/Mass':162 - 166
#'0/ElectroAbundance':166 - 170
#'0/H2Fraction':83 - 88
#'0/StarFormationRate':172 - 176

seq 83 88 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" &
seq 172 176 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" &

wait
EOF
