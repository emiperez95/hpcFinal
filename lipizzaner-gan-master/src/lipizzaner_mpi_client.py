# Debugging
import logging


from helpers.mpi_comms import CommsManager
from helpers.configuration_container import ConfigurationContainer
from helpers.grid import GridManager

import sys
import pprint
from time import sleep

class LipizzanerMpiClient:

    _logger = logging.getLogger(__name__)

    def __init__(self):
        self.comms = CommsManager.instance()
        self.comms.start_comms()
        self.grid = GridManager.instance()

        self.run()

    def run(self):
        while(1):
            data = self.comms.recieve_root()
            task = data["task"]
            if task == "stop":
                self.to_log("Stop signal recieved")
                break
            elif task == "work":
                print("recieved")
                sys.stdout.flush()
                self.to_log("Work signal recieved")

                self.grid.load_grid(data["grid"])
                self.__process_data()

                self.to_log("Finished work")
            elif task == "alive":
                # Aca se deberia responder con un comm
                pass

    def __process_data(self):
        neightbours_list = self.grid.get_neightbours(self.comms.rank)
        self.to_log(neightbours_list.__str__())
        sleep(10)

    def to_log(self, message):
        print("{}) ".format(self.comms.rank) + message)
        sys.stdout.flush()
