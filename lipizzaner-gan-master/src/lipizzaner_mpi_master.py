import logging
import signal
import time
from time import sleep

from helpers.db_logger import DbLogger
from helpers.configuration_container import ConfigurationContainer
from helpers.reproducible_helpers import set_random_seed
from helpers.topology import TopologyManager
from distribution.comms_manager import CommsManager
from distribution.grid_manager import GridManager

# import pprint

class LipizzanerMpiMaster:

    _logger = logging.getLogger(__name__)

    def __init__(self):
        self.cc = ConfigurationContainer.instance()
        self.comms = CommsManager.instance()
        self.comms.start_comms()
        # TODO: Falta el size check
        self.topology = TopologyManager.instance()
        self.grid = GridManager.instance()
        self.experiment_id = None

        set_random_seed(self.cc.settings['general']['seed'],
                        self.cc.settings['trainer']['params']['score']['cuda'])
        self._logger.info("Seed used in master: {}".format(self.cc.settings['general']['seed']))

        signal.signal(signal.SIGINT, self._sigint)
        self._start_experiments()
        
        # TODO: Check for end
        # TODO: gather results

        # End program
        # self.comms.stop_running_experiments()

    def _start_experiments(self):
        self.cc.settings['general']['distribution']['start_time'] = time.strftime('%Y-%m-%d_%H-%M-%S')

        # If DB logging is enabled, create a new experiment and attach its ID to settings for clients
        db_logger = DbLogger()
        if db_logger.is_enabled:
            self.experiment_id = db_logger.create_experiment(self.cc.settings)
            self.cc.settings['general']['logging']['experiment_id'] = self.experiment_id

        while self.grid.empty_spaces() > 0:
            worker = self.topology.get_best_worker()
            self.grid.assign_worker(worker)   
            self.topology.assign_worker(worker)
        
        # print(self.grid.grid)
        for proc_unit in self.grid.grid_to_list():
            self.cc.settings["general"]["distribution"]["grid"]["config"] = self.grid.grid
            self.comms.start_worker(proc_unit, self.cc.settings)
            # sleep(1)
    
    def _sigint(self, signal, frame):
        self._terminate(stop_clients=True)

    def _terminate(self, stop_clients=True, return_code=-1):
        try:
            if stop_clients:
                self._logger.info('Stopping clients...')
                self.comms.stop_running_experiments()
        finally:
            db_logger = DbLogger()
            if db_logger.is_enabled and self.experiment_id is not None:
                db_logger.finish_experiment(self.experiment_id)

            exit(return_code)


    
    