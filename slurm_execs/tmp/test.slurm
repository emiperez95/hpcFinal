#!/bin/bash
#SBATCH --job-name=sh_test
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8192
#SBATCH --time=2:30:00
#SBATCH --partition=besteffort
#SBATCH --qos=besteffort
#SBATCh --tmp=20G

cd ~/editing/hpcFinal/lipizzaner-gan-master/src/
source activate lipizzaner

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/clusteruy/home/eperez/anaconda3/lib/

mpirun -n 5 python ~/all_test.py
mpirun -n 5 python main.py train --mpi -f configuration/clusteruy-test/mnistMPI_35_nogpu.yml

cp -ru /scratch/output ~/output_cpu
