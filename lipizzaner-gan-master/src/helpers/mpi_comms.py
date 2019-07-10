
from helpers.singleton import Singleton
from helpers.configuration_container import ConfigurationContainer

from mpi4py import MPI

@Singleton
class CommsManager():
    def __init__(self):
        cc = ConfigurationContainer.instance()
        self.root = cc.settings['general']['distribution']['root']
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.sys_info = {
            "node" : TestData.nodes[self.rank],
            "rank" : self.rank
        }
    
    def start_comms(self):
        data = None
        data = self.comm.gather(self.sys_info, root=self.root)
        self.nodes_info = data

    def send(self, message, dest, tag=1):
        self.comm.isend(message, dest=dest, tag=tag)

    def start_worker(self, pu, grid):
        self.comm.send({"task" : "work", "grid" : grid}, dest=pu, tag=1)

    def recieve_root(self, tag=1):
        return self.comm.recv(source=self.root, tag=tag)

    def close_all(self):
        for i in range(self.size):
            self.comm.isend({"task": "stop"}, dest=i, tag=1)



        
class TestData():
    nodes = {
            0: "machine1",
            1: "machine1",
            2: "machine2",
            3: "machine2",
            4: "machine3",
            5: "machine3",
            6: "machine4",
            7: "machine4",
            8: "machine5",
            9: "machine5",
            10: "machine6",
            11: "machine6",
            12: "machine7",
            13: "machine7",
            14: "machine8",
            15: "machine8",
            16: "machine9",
            17: "machine9",
            18: "machine10",
            19: "machine10",
        }   

