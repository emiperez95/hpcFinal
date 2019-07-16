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
        self.is_busy = False
        self.is_finished = False

        self.run()

    def run(self):
        while(1):
            self._logger.warn("{}) Waiting for task".format(self.comms.rank))
            data = self.comms.recv_task()
            task = data["task"]
            func = getattr(self, "_task_"+task)
            result = func(data)
            # result = globals()["_task_"+task](data)
            if result:
                self.comms.isend(result, data["source"])
                if type(result) == "int":
                    break

    # def __process_data(self):
    #     neightbours_list = self.grid.get_neightbours(self.comms.rank)
    #     self.to_log(neightbours_list.__str__())
    #     sleep(10)

    # def to_log(self, message):
    #     print("{}) ".format(self.comms.rank) + message)
    #     sys.stdout.flush()

    def _task_run(self, data):
        config = data["config"]
        self._lock.acquire()
        if self.is_busy:
            self._lock.release()
            return 'Client is currently busy.'
        
        self.is_finished = False
        self.is_busy = True
        self._lock.release()
        self._stop_event = Event()
        self._finish_event = Event()
        worker_thread = Thread(target=LipizzanerMpiClient._run_lipizzaner, args=(config,))
        worker_thread.start()

        return None

    def _task_stop(self, data):
        self._lock.acquire()

        if self.is_busy:
            self._logger.warning('Received stop signal from master, experiment will be quit.')
            self._stop_event.set()
        else:
            self._logger.warning('Received stop signal from master, but no experiment is running.')

        self._lock.release()
        return None
    
    def _task_close(self, data):
        return "end"

    def _task_status(self, data):
        result = {
            'busy': self.is_busy,
            'finished': self.is_finished
        }
        return result

    def _task_results(self, data):
        self._lock.acquire()
        if self.is_busy:
            self._logger.info('Sending neighbourhood results to master')
            response = self._gather_results()
            self._finish_event.set()
        else:
            self._logger.warning('Master requested results, but no experiment is running.')
            response = None

        self._lock.release()
            
        return response

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

        populations.lock()
        if populations.discriminator is not None:
            best_individual = sorted(populations.discriminator.individuals, key=lambda x: x.fitness)[0]
            parameters = [self._individual_to_dict(best_individual)]
        else:
            parameters = []
        populations.unlock()
        return parameters
    
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
            LipizzanerMpiClient._is_finished = True

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
    def _gather_results():
        neighbourhood = Neighbourhood.instance()
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

        return results