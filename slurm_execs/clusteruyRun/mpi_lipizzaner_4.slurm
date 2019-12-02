#!/bin/bash
#SBATCH --job-name=4_mpi_lipizznaer
#SBATCH --ntasks=17
#SBATCH --mem-per-cpu=8192
#SBATCH --time=16:00:00
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emiliano.perez.mauri@gmail.com

cd ~/hpcFinal/lipizzaner-gan-master/src/
source activate lipizzaner

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/clusteruy/home/eperez/anaconda3/lib/

mpirun -n 17 python main.py train --mpi -f configuration/clusteruy-test/runs/4x4/mnistMPI.yml
