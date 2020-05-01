#!/bin/bash
#SBATCH -p development
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --job-name=unpack
#SBATCH --output=%x-%j.out
#SBATCH --time=2:00:00

hostname; pwd; date
  
data_root_dir="/home1/06431/yueyingn/scratch/BT1/PART_005"
out_dir="/home1/06431/yueyingn/scratch/BT1/PART_005_recover"


for i in 32 33 34 35 36 37 38 39; do
    filename="tar_file.$i"
    echo unpack $filename
    tar -xvf "$data_root_dir/$filename" --strip-components=5 -C "$out_dir"
done

date

