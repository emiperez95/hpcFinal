# Debugging
import logging
import os
import traceback
from threading import Thread, Lock, Event
# import sys
import pprint
# from time import sleep

from distribution.comms_manager import CommsManager
from distribution.concurrent_populations import ConcurrentPopulations
from distribution.state_encoder import StateEncoder
from distribution.neighbourhood import Neighbourhood
from distribution.grid import Grid
from helpers.configuration_container import ConfigurationContainer
from helpers.log_helper import LogHelper
from helpers.or_event import or_event
from lipizzaner import Lipizzaner

class LipizzanerMpiClient():
    is_busy = False
    is_finished = False

    _stop_event = None
    _finish_event = None
    _lock = Lock()

    _logger = logging.getLogger(__name__)

    def __init__(self):
        self.comms = CommsManager.instance()
        self.comms.start_comms()
        self.grid = Grid.instance()
        LipizzanerMpiClient.is_busy = False
        LipizzanerMpiClient.is_finished = False

        self.run()

    def run(self):
        while(1):
            self._logger.info("{}) Waiting for task".format(self.comms.rank))
            data = self.comms.recv_task()
            task = data["task"]
            func = getattr(self, "_task_"+task)
            result = func(data)
            # result = globals()["_task_"+task](data)
            if result:
                if isinstance(result, int):
                    break
                else:
                    self.comms.isend(result, data["source"])
        self._logger.info("Exiting...")

    # ===================================================
    #                Execution opers
    # ===================================================

    def _task_run(self, data):
        config = data["config"]
        LipizzanerMpiClient._lock.acquire()
        if LipizzanerMpiClient.is_busy:
            LipizzanerMpiClient._lock.release()
            return 'Client is currently busy.'
        
        LipizzanerMpiClient.is_finished = False
        LipizzanerMpiClient.is_busy = True
        LipizzanerMpiClient._lock.release()
        LipizzanerMpiClient._stop_event = Event()
        LipizzanerMpiClient._finish_event = Event()
        worker_thread = Thread(target=LipizzanerMpiClient._run_lipizzaner, args=(config,))
        worker_thread.start()

        return None

    def _task_stop(self, data):
        LipizzanerMpiClient._lock.acquire()

        if LipizzanerMpiClient.is_busy:
            self._logger.warning('Received stop signal from master, experiment will be quit.')
            LipizzanerMpiClient._stop_event.set()
        else:
            self._logger.warning('Received stop signal from master, but no experiment is running.')

        LipizzanerMpiClient._lock.release()
        return None
    
    def _task_close(self, data):
        return 1

    def _task_status(self, data):
        result = {
            'busy': LipizzanerMpiClient.is_busy,
            'finished': LipizzanerMpiClient.is_finished
        }
        return result

    def _task_results(self, data):
        # LipizzanerMpiClient._lock.acquire()
        if LipizzanerMpiClient.is_busy:
            self._logger.info('Sending neighbourhood results to master')
            worker_thread = Thread(target=LipizzanerMpiClient._gather_results, args=(data["source"],))
            worker_thread.start()
            # response = self._gather_results()
        else:
            self._logger.warning('Master requested results, but no experiment is running.')
            response = None

        # LipizzanerMpiClient._lock.release()
            
        return None

    def _task_generators_best(self, data):
        populations = ConcurrentPopulations.instance()

        if populations.generator is not None:
            best_individual = sorted(populations.generator.individuals, key=lambda x: x.fitness)[0]
            parameters = [self._individual_to_dict(best_individual)]
        else:
            parameters = []
        return parameters

    def _task_generators(self, data):
        populations = ConcurrentPopulations.instance()

        populations.lock()
        if populations.generator is not None:
            parameters = [self._individual_to_dict(i) for i in populations.generator.individuals]
        else:
            parameters = []
        populations.unlock()
        return parameters

    def _task_discriminators(self, data):
        populations = ConcurrentPopulations.instance()

        populations.lock()
        if populations.discriminator is not None:
            parameters = [self._individual_to_dict(i) for i in populations.discriminator.individuals]
        else:
            parameters = []
        populations.unlock()
        return parameters

    def _task_discriminators_best(self, data):
        populations = ConcurrentPopulations.instance()

        not_finished = not LipizzanerMpiClient.is_finished
        if not_finished:
            populations.lock()
        
        if populations.discriminator is not None:
            best_individual = sorted(populations.discriminator.individuals, key=lambda x: x.fitness)[0]
            parameters = [self._individual_to_dict(best_individual)]
        else:
            parameters = []
        
        if not_finished:
            populations.unlock()
        return parameters
    
    # ===================================================
    #                New data opers
    # ===================================================
    def _task_gen_disc(self, data):
        params = {
            "generator" : self._task_generators(data),
            "discriminator" : self._task_discriminators(data)
        }
        return params

    # ===================================================
    #                Comms opers
    # ===================================================
    def _task_new_comm(self, data):
        self.comms.new_comm(data["color"], data["key"])


    # ===================================================
    #                Static aux funcs
    # ===================================================
    @staticmethod
    def _run_lipizzaner(config):
        cc = ConfigurationContainer.instance()
        cc.settings = config
        
        grid = Grid.instance()  
        grid.load_grid()

        output_base_dir = cc.output_dir
        LipizzanerMpiClient._set_output_dir(cc)

        if 'logging' in cc.settings['general'] and cc.settings['general']['logging']['enabled']:
            LogHelper.setup(cc.settings['general']['logging']['log_level'], cc.output_dir)
        message = 'Distributed training recognized, set log directory to {}'.format(cc.output_dir)
        LipizzanerMpiClient._logger.info(message)

        try:
            lipizzaner = Lipizzaner()
            lipizzaner.run(cc.settings['trainer']['n_iterations'], LipizzanerMpiClient._stop_event)
            LipizzanerMpiClient.is_finished = True

            # Wait until master finishes experiment, i.e. collects results, or experiment is terminated
            or_event(LipizzanerMpiClient._finish_event, LipizzanerMpiClient._stop_event).wait()
        except Exception as ex:
            LipizzanerMpiClient.is_finished = True
            LipizzanerMpiClient._logger.critical('An unhandled error occured while running Lipizzaner: {}'.format(ex))
            traceback.print_exc()
            raise ex
        finally:
            LipizzanerMpiClient.is_busy = False
            LipizzanerMpiClient._logger.info('Finished experiment, waiting for new requests.')
            cc.output_dir = output_base_dir
            ConcurrentPopulations.instance().lock()




    @staticmethod
    def _individual_to_dict(individual):
        individual_dict = {
            'id': individual.id,
            'parameters': individual.genome.encoded_parameters,
            'learning_rate': individual.learning_rate,
            'optimizer_state': StateEncoder.encode(individual.optimizer_state)
        }
        if individual.iteration is not None:
            individual_dict['iteration'] = individual.iteration

        return individual_dict

    @classmethod
    def _set_output_dir(cls, cc):
        output = cc.output_dir
        dataloader = cc.settings['dataloader']['dataset_name']
        start_time = cc.settings['general']['distribution']['start_time']

        cc.output_dir = os.path.join(output, 'distributed', dataloader, start_time, str(os.getpid()))
        os.makedirs(cc.output_dir, exist_ok=True)
    
    @staticmethod
    def _gather_results(source):
        LipizzanerMpiClient._lock.acquire()
        neighbourhood = Grid.instance()
        cc = ConfigurationContainer.instance()
        results = {
            'generators': neighbourhood.best_generator_parameters,
            'discriminators': neighbourhood.best_discriminator_parameters,
            'weights_generators': neighbourhood.mixture_weights_generators
        }
        if cc.settings['trainer']['name'] == 'with_disc_mixture_wgan' \
            or cc.settings['trainer']['name'] == 'with_disc_mixture_gan':
            results['weights_discriminators'] = neighbourhood.mixture_weights_discriminators
        else:
            results['weights_discriminators'] = 0.0
        LipizzanerMpiClient._finish_event.set()
        LipizzanerMpiClient._lock.release()

        comms = CommsManager.instance()
        comms.isend(results, source)
        # return results