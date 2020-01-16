#!/bin/bash
#SBATCH --job-name=sh_test
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8192
#SBATCH --time=2:30:00
#SBATCH --partition=besteffort
#SBATCH --qos=besteffort
#SBATCH --tmp=40G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emiliano.perez.mauri@gmail.com

mkdir /scratch/eperez/output
sbcast -v /scratch/eperez/output
sbcast -v ~/editing/hpcFinal/lipizzaner-gan-master/src/output/data /scratch/eperez/output/data
#cp -ru ~/editing/hpcFinal/lipizzaner-gan-master/src/output/data /scratch/eperez/output/data

cd ~/editing/hpcFinal/lipizzaner-gan-master/src/
source activate lipizzaner

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/clusteruy/home/eperez/anaconda3/lib/

mpirun -n 5 python ~/all_test.py
mpirun -n 5 kernprof -l  main.py train --mpi -f configuration/clusteruy-test/mnistMPI_35_nogpu.yml

mv main.py.lprof main_cpu_b.py.lprof
cp -ru /scratch/output ~/output_cpu
