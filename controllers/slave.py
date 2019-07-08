import time #sleep
from .logger import Logger


class Slave():

    def __init__(self, cc):
        self.cc = cc
        self.lg = Logger(cc)
        self.__send_pu_info()
        self.__run()

    def __send_pu_info(self):
        comm = self.cc["mpi"]["comm"]
        comm.gather(self.cc["sys_info"],\
                    root=self.cc["mpi"]["root"])
    
    def __process_data(self, task):
        # Processing

        self.lg.to_log(self.cc["sys_info"].__str__())
        time.sleep(10)
        # Request neighbour info?

        # More processing
        # time.sleep(10)

    def __run(self):
        comm = self.cc["mpi"]["comm"]
        while(1):
            data = comm.recv(source=0, tag=1)
            task = data["task"]
            if task == "stop":
                self.lg.to_log("Stop signal recieved")
                break
            elif task == "work":
                self.lg.to_log("Work signal recieved")
                self.__process_data(data)
                self.lg.to_log("Finished work")
            elif task == "alive":
                # Aca se deberia responder con un comm
                pass