#!/bin/bash
#SBATCH -p development
#SBATCH -N 11
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


seq 13 19 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" &
seq 20 25 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" &
seq 26 32 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" &
seq 33 39 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" &

seq 79 83 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" &
seq 88 94 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" &
seq 95 101 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" &
seq 102 108 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" & 
seq 109 115 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" & 
seq 116 123 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" &
seq 162 170 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" &

wait
EOF
