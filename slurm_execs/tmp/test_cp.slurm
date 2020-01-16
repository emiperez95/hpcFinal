#!/bin/bash
#SBATCH --job-name=sh_test
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8192
#SBATCH --time=2:30:00
#SBATCH --partition=besteffort
#SBATCH --qos=besteffort
#SBATCH --tmp=20G

pwd
ls /scratch/eperez/
cp -r ~/editing/hpcFinal /scratch/eperez/hpcFinal2
pwd
cd /scratch/eperez/hpcFinal2/lipizzaner-gan-master/src/
pwd
source activate lipizzaner

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/clusteruy/home/eperez/anaconda3/lib/

#mpirun -n 5 python ~/all_test.py
pwd
ls /scratch/eperez/hpcFinal2/lipizzaner-gan-master/src/
mpirun -n 2 ls
mpirun -n 5 python /scratch/eperez/hpcFinal2/lipizzaner-gan-master/src/main.py train --mpi -f configuration/clusteruy-test/mnistMPI_35_nogpu.yml
mv /scratch/eperez/hpcFinal/lipizzaner-gan-master/src/output/statistics.txt ~/stats_res/statistics.txt
