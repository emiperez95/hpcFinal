#!/bin/bash
#SBATCH --job-name=scratch_test
#SBATCH --nodes=5
#SBATCH --mem-per-cpu=512
#SBATCH --time=0:30:00
#SBATCH --partition=normal
#SBATCH --qos=rapida
#SBATCH --tmp=40G

#User variabels (modify these)
USR=eperez
SOURCE=~/editing/hpcFinal/lipizzaner-gan-master/src
OUTPUT=~/editing/hpcFinal/slurm_execs/cpu_test/
USE_PROFILING=0

#Program variables
LOCAL_DATA=/scratch/$USR/output/data
SOURCE_DATA=$SOURCE/output/data

#Moving data pre-execution
srun mkdir -p $LOCAL_DATA/raw
srun mkdir -p $LOCAL_DATA/processed
files_to_move=(
	raw/t10k-images-idx3-ubyte
	raw/t10k-labels-idx1-ubyte
	raw/train-images-idx3-ubyte
	raw/train-labels-idx1-ubyte
	processed/test.pt
	processed/training.pt
)
for i in ${files_to_move[@]}; do 
	sbcast -v $SOURCE_DATA/$i $LOCAL_DATA/$i
done
#Making correct enviroment
cd $SOURCE
source activate lipizzaner
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/clusteruy/home/$USR/anaconda3/lib/
echo "=============EXITO===================="
<<COMMENT1
if $USE_PROFILING 
then
	mpirun -n 5 kernprof -l  main.py train --mpi -f configuration/clusteruy-test/cpu_test/mnistMPI.yml
	mv main.py.lprof $OUTPUT
else
	mpirun -n 5  main.py train --mpi -f configuration/clusteruy-test/cpu_test/mnistMPI.yml
fi
COMMENT1

echo "exito denuevo"
mpirun -n 5  python  main.py train --mpi -f configuration/clusteruy-test/cpu_test/mnistMPI.yml
cp -ru /scratch/$USR/output $OUTPUT
