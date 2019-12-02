#!/bin/bash
#SBATCH --job-name=mpi_lipizzaner_test
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=0:30:00
#SBATCH --partition=normal
#SBATCH --qos=rapida
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emiliano.perez.mauri@gmail.com

cd ~/hpcFinal/lipizzaner-gan-master/src/
source activate lipizzaner

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/clusteruy/home/eperez/anaconda3/lib/

mpirun -n 5 python main.py train --mpi -f configuration/clusteruy-test/mnistMPI_test.yml
