from mpi4py import MPI
# import platform #Get node name
import time #sleep

from controllers.master import Master
from controllers.slave import Slave
from test_pkg.testData import Test_Data


import pprint #Pretty printing


'''
Roles a designar:
    -Master
        .Crear la grilla
        .Checkear que los diferentes miembros siguen vivos
    -Logger
    -Miembros del grid
    -Master replace
        .Checkear que el master sigue vivo

Ideas:
    -Que los logs se levanten a la nube, asi no dependemos de quien es el log

Tags: (comm (->World))
    -Tag1: Tasks given by master
    -Tag2: Log matters
'''

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # TESTING:
    node_name = Test_Data().rank2name[rank]
    # 

    cc = {
        "mpi" : {
            "root" : 0,  #Initial root process
            "size" : size,
            "rank" : rank,
            "comm" : comm
        },

        "grid" : {
            "x_size" : 4,
            "y_size" : 4,
            "len" : 0
        },

        "sys_info" : {
            "node_name": node_name,
            "rank": rank
        }
    }
    cc["grid"]["len"] = cc["grid"]["x_size"] * cc["grid"]["y_size"]


    if rank == cc["mpi"]["root"]:
        Master(cc)
        
    else:
        Slave(cc)
        # slave_init()
        # slave_execution()
        # data = None