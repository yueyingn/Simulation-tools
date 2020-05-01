#!/bin/bash
#SBATCH -p development
#SBATCH -N 7
#SBATCH --ntasks-per-node 1
#SBATCH --job-name=unpack
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=2:00:00

hostname; pwd; date
  
data_root_dir="/home1/06431/yueyingn/scratch/BT3/PART_271_pack"
out_dir="/home1/06431/yueyingn/scratch/BT3/"

#for i in 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110; do
#    filename="tar_file.$i"
#    echo unpack $filename
#    tar -xvf "$data_root_dir/$filename" --strip-components=7 -C "$out_dir"
#done

seq 70 76 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" &
seq 77 83 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" &
seq 88 92 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" &
seq 96 105 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" &
seq 106 114 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" & 
seq 115 123 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" &
seq 162 170 | srun -N 1 -n 1 xargs -P 1 -d '\n' -I file_num tar -xf $data_root_dir/tar_file.file_num --strip-components=7 -C "$out_dir" &

wait
EOF
