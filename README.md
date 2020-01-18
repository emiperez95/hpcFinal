## Instalation instructions for clusteruy

Once in login.cluster.uy, do the following:

1. Install conda on server

2. Set up enviroment in compilation server
```
conda env create -f ./lipizzaner-gan-master/src/helper_files/environment_mpi.yml
source activate lipizzaner
```

3. Configure local settings:
  - On ./slurm_execs/cpu_tests/cpu_run.slurm:
    - USR: your cluster username
    - SOURCE: Complete location of src folder (./lipizzaner-gan-master/src from this file)
    - OUTPUT: Desired output folder for run results
    - USE_PROFILING: Not in use at the moment
  - On ./lipizzaner-gan-master/src/configuration/clusteruy-test/cpu_test/generalMPI.yml
    - Change output dir to /scratch/<USR>/output, using yout cluster username
  
4. Configure run settings: 
  The defailt run setting is for 2x2 grid, to change the size:
  - On ./slurm_execs/cpu_tests/cpu_run.slurm:
    - Change number of nodes to be used to total cell ammounts plus one (example: if 4x4 grid desired, you need 17 nodes)
    ```
    #SBATCH --nodes=<NUMBER_OF_NODES>
    ```
    - Change the number of nodes to be given to mpi for execution:
    ```
    mpirun -n <NUMBER_OF_NODES>  python  main.py train --mpi -f configuration/clusteruy-test/cpu_test/mnistMPI.yml
    ```
    -You might need to change other sbatch settings as you see fit
    
  - On ./lipizzaner-gan-master/src/configuration/clusteruy-test/cpu_test/mnistlMPI.yml
    - Change grid x_size and y_size to desired ammounts.
    

