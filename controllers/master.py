from .topology import Topology_manager
from .grid import Grid_manager

from time import sleep

class Master():
    def __init__(self, cc):
        comm = cc["mpi"]["comm"]
        self.cc = cc
        self.__size_check()
        self.__recieve_all_pu_info()

        self.topology = Topology_manager(cc)
        self.grid = Grid_manager(cc)

        while self.grid.empty_spaces() > 0:
            worker = self.topology.get_best_worker()
            self.grid.assign_worker(worker)   
            self.topology.assign_worker(worker)
            comm.isend({"task" : "work"}, dest=worker, tag=1)

        self.__mpi_close_all()

    def __recieve_all_pu_info(self):
        comm = self.cc["mpi"]["comm"]
        nodes_top = comm.gather(self.cc["sys_info"],\
                                root=self.cc["mpi"]["root"])
        self.cc["nodes_rank"] = nodes_top

    def __size_check(self):
        size = self.cc["mpi"]["size"]
        req_length = self.cc["grid"]["len"]+2
        if (req_length) > size:
            message = '''
            Not enough procesing units to make given grid size.
            Number should be greater than {}.
            '''.format(req_length)
            self.__mpi_close_all()
            raise Exception(message)

    def __mpi_close_all(self):
        size = self.cc["mpi"]["size"]
        comm = self.cc["mpi"]["comm"]
        for i in range(size):
            comm.isend({"task": "stop"}, dest=i, tag=1)