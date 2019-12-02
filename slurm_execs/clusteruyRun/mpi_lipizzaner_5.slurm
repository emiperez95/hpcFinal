#!/bin/bash
#SBATCH --job-name=5_mpi_lipizznaer
#SBATCH --ntasks=26
#SBATCH --mem-per-cpu=8192
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emiliano.perez.mauri@gmail.com

cd ~/hpcFinal/lipizzaner-gan-master/src/
source activate lipizzaner

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/clusteruy/home/eperez/anaconda3/lib/

mpirun -n 26 python main.py train --mpi -f configuration/clusteruy-test/runs/5x5/mnistMPI.yml
