import logging
import os
import signal
import time
import traceback
from time import sleep
from multiprocessing import Event
import torch
from torch.autograd import Variable

from helpers.db_logger import DbLogger
from helpers.configuration_container import ConfigurationContainer
from helpers.reproducible_helpers import set_random_seed
from helpers.topology import TopologyManager
from helpers.heartbeat_mpi import Heartbeat
from distribution.comms_manager import CommsManager
from distribution.grid_manager import GridManager
from training.mixture.mixed_generator_dataset import MixedGeneratorDataset
from training.mixture.score_factory import ScoreCalculatorFactory

# import pprint
GENERATOR_PREFIX = 'generator-'
DISCRIMINATOR_PREFIX = 'discriminator-'

class LipizzanerMpiMaster:

    _logger = logging.getLogger(__name__)

    @profile
    def __init__(self):
        self.cc = ConfigurationContainer.instance()
        self.comms = CommsManager.instance()
        self.comms.start_comms()
        self.topology = TopologyManager.instance()
        self.grid = GridManager.instance()

        self._size_error(self.comms, self.grid)
        self.heartbeat_event = None
        self.heartbeat_thread = None
        self.experiment_id = None

        set_random_seed(self.cc.settings['general']['seed'],
                        self.cc.settings['trainer']['params']['score']['cuda'])
        self._logger.info("Seed used in master: {}".format(self.cc.settings['general']['seed']))

        self.heartbeat_event = Event()
        self.heartbeat_thread = Heartbeat(self.heartbeat_event, False)

        signal.signal(signal.SIGINT, self._sigint)

        self._load_dataset()

        self._start_experiments()

        self.heartbeat_thread.start()
        self._logger.info("Started heartbeat")
        self.heartbeat_thread.join()

        if self.heartbeat_thread.success:
            self._logger.info("Heartbeat stopped with success")
            self._gather_results()
            self._terminate(stop_clients=True, return_code=0)
        else:
            self._logger.info("Heartbeat stopped with error")
            self._terminate(stop_clients=False, return_code=-1)
        # self._terminate()
    @profile
    def _start_experiments(self):
        self.cc.settings['general']['distribution']['start_time'] = time.strftime('%Y-%m-%d_%H-%M-%S')

        # If DB logging is enabled, create a new experiment and attach its ID to settings for clients
        db_logger = DbLogger()
        if db_logger.is_enabled:
            self.experiment_id = db_logger.create_experiment(self.cc.settings)
            self.cc.settings['general']['logging']['experiment_id'] = self.experiment_id

        while self.grid.empty_spaces() > 0:
            worker = self.topology.get_best_worker()
            w_id = self.grid.assign_worker(worker)   
            self.topology.assign_worker(worker)
            
            self._logger.info("Asigned worker {} to wid {}".format(worker, w_id))
            self.comms.send_task("new_comm", worker, data={"color" : 0, "key" : w_id})

        for off_pu in self.topology.inactive_pu:
            self._logger.info("Asigned worker {} to rest".format(off_pu))
            self.comms.send_task("new_comm", off_pu, data={"color" : 1, "key" : off_pu})
        
        self.comms.new_comm_master(2, 0)
        
        # print(self.grid.grid)
        for proc_unit in self.grid.grid_to_list():
            self.cc.settings["general"]["distribution"]["grid"]["config"] = self.grid.grid
            self.comms.start_worker(proc_unit, self.cc.settings)
            #sleep(2)
    
    # Pre downloading the dataset for distributed sistems.
    # Otherwise, the parallel datasets would overlap each other and fail.
    def _load_dataset(self):
         if ( 'distributed_filesystem' in self.cc.settings['general']['distribution'] \
            and self.cc.settings['general']['distribution']['distributed_filesystem'] == True ):
                self._logger.info('Downloading dataset...')
                self.cc.create_instance(self.cc.settings['dataloader']['dataset_name']).load()
                self._logger.info('Dataset downloaded.')



    def _sigint(self, signal, frame):
        self._terminate(stop_clients=True)

    def _terminate(self, stop_clients=True, return_code=-1):
        try:
            if stop_clients:
                self._logger.info('Stopping clients...')
                self.comms.close_all() #stop_running_experiments()
        finally:
            db_logger = DbLogger()
            if db_logger.is_enabled and self.experiment_id is not None:
                db_logger.finish_experiment(self.experiment_id)

            exit(return_code)
    
    @profile
    def _gather_results(self):
        self._logger.info('Collecting results from clients...')

        # Initialize node client
        dataloader = self.cc.create_instance(self.cc.settings['dataloader']['dataset_name'])
        # network_factory = self.cc.create_instance(self.cc.settings['network']['name'], dataloader.n_input_neurons)
        node_client = self.comms
        db_logger = DbLogger()

        results = node_client.general_gather_results()

        self._logger.info('Results scoring')
        scores = []
        for (node, generator_pop, discriminator_pop, weights_generator, _) in results:
            # self._logger.info("Getting result from {}".format(node))
            # node_name = '{}:{}'.format(node['address'], node['port'])
            node_name = node
            try:
                output_dir = self.get_and_create_output_dir(node)

                for generator in generator_pop.individuals:
                    source = generator.source['id']
                    filename = '{}{}.pkl'.format(GENERATOR_PREFIX, source)
                    self._logger.info("Filename: {}".format(filename))
                    torch.save(generator.genome.net.state_dict(),
                               os.path.join(output_dir, 'generator-{}.pkl'.format(source)))

                    with open(os.path.join(output_dir, 'mixture.yml'), "a") as file:
                        # self._logger.error("Weights generator: " + weights_generator.__str__())
                        file.write('{}: {}\n'.format(filename, weights_generator[source]))

                for discriminator in discriminator_pop.individuals:
                    source = discriminator.source['id']
                    filename = '{}{}.pkl'.format(DISCRIMINATOR_PREFIX, source)
                    self._logger.info("Filename: {}".format(filename))
                    torch.save(discriminator.genome.net.state_dict(),
                               os.path.join(output_dir, filename))

                # # Save images
                self._logger.error("Data recieved\nGen pop {} \nWeights gen {}"
                                    .format(generator_pop, weights_generator))
                dataset = MixedGeneratorDataset(generator_pop,
                                                weights_generator,
                                                self.cc.settings['master']['score_sample_size'],
                                                self.cc.settings['trainer']['mixture_generator_samples_mode'])
                image_paths = self.save_samples(dataset, output_dir, dataloader)
                self._logger.info('Saved mixture result images of client {} to target directory {}.'
                                  .format(node_name, output_dir))

                # Calculate inception or FID score
                score = float('-inf')
                if self.cc.settings['master']['calculate_score']:
                    calc = ScoreCalculatorFactory.create()
                    self._logger.info('Score calculator: {}'.format(type(calc).__name__))
                    self._logger.info('Calculating score score of {}. Depending on the type, this may take very long.'
                                      .format(node_name))

                    score = calc.calculate(dataset)
                    self._logger.info('Node {} with weights {} yielded a score of {}'
                                      .format(node_name, weights_generator, score))
                    scores.append((node, score))

                if db_logger.is_enabled and self.experiment_id is not None:
                    db_logger.add_experiment_results(self.experiment_id, node_name, image_paths, score)
            except Exception as ex:
                self._logger.error('An error occured while trying to gather results from {}: {}'.format(node_name, ex))
                traceback.print_exc()

        if self.cc.settings['master']['calculate_score'] and scores:
            best_node = sorted(scores, key=lambda x: x[1], reverse=ScoreCalculatorFactory.create().is_reversed)[-1]
            self._logger.info('Best result: pu{} = {}'.format(best_node[0], best_node[1]))

    def get_and_create_output_dir(self, node):
            directory = os.path.join(self.cc.output_dir, 'master', self.cc.settings['general']['distribution']['start_time'],
                                    'pu{}'.format(node))
            os.makedirs(directory, exist_ok=True)
            return directory

    def save_samples(self, dataset, output_dir, image_specific_loader, n_images=10, batch_size=100):
        image_format = self.cc.settings['general']['logging']['image_format']
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        loaded = image_specific_loader.load()
        paths = []


        for i, data in enumerate(dataloader):
            shape = loaded.dataset.train_data.shape if hasattr(loaded.dataset, 'train_data') else None
            path = os.path.join(output_dir, 'mixture-{}.{}'.format(i + 1, image_format))
            image_specific_loader.save_images(Variable(data), shape, path)
            paths.append(path)

            if i + 1 == n_images:
                break

        return paths

    def _size_error(self, comms, grid):
        pu_size = comms.size
        grid_size = grid.grid_x * grid.grid_y
        if grid_size+1 > pu_size:
            self._logger.error("Not enough Procesing Units to execute this grid"\
                "at least {} PUs are required.".format(grid_size+1))
            comms.close_all()
            exit(-1)
