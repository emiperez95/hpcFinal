#!/bin/bash
#SBATCH --job-name=3_mpi_lipizznaer
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=8192
#SBATCH --time=8:00:00
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emiliano.perez.mauri@gmail.com

cd ~/hpcFinal/lipizzaner-gan-master/src/
source activate lipizzaner

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/clusteruy/home/eperez/anaconda3/lib/

mpirun -n 10 python main.py train --mpi -f configuration/clusteruy-test/runs/3x3/mnistMPI.yml
