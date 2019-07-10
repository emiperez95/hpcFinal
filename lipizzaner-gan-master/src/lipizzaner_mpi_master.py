# Debugging
import logging


from helpers.mpi_comms import CommsManager
from helpers.configuration_container import ConfigurationContainer

from helpers.grid import GridManager
from helpers.topology import TopologyManager


import pprint
from time import sleep

class LipizzanerMpiMaster:

    _logger = logging.getLogger(__name__)

    def __init__(self):
        self.cc = ConfigurationContainer.instance()
        self.comms = CommsManager.instance()
        self.comms.start_comms()
        # Falta el size check

        self.topology = TopologyManager.instance()
        self.grid = GridManager.instance()
        while self.grid.empty_spaces() > 0:
            worker = self.topology.get_best_worker()
            self.grid.assign_worker(worker)   
            self.topology.assign_worker(worker)
        
        print(self.grid.grid)
        for pu in self.grid.grid_to_list():
            self.comms.start_worker(pu, self.grid.grid)
            sleep(1)

        # End program
        self.comms.close_all()
