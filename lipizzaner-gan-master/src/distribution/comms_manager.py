import time
from concurrent.futures import as_completed, ThreadPoolExecutor

from distribution.node_client import NodeClient
from distribution.state_encoder import StateEncoder
from helpers.log_helper import logging
from helpers.singleton import Singleton
from helpers.configuration_container import ConfigurationContainer
from helpers.individual import Individual
from helpers.population import Population, TYPE_GENERATOR, TYPE_DISCRIMINATOR

# import mpi4py
# mpi4py.rc.recv_mprobe = False
# import dill
from mpi4py import MPI
# MPI.pickle.__init__(dill.dumps, dill.loads)

# MPI.pickle.dumps = dill.dumps
# MPI.pickle.loads = dill.loads


TIMEOUT_SEC_DEFAULT = 10
MAX_HTTP_CLIENT_THREADS = 5

# TODO: Mantener un pu con el backup de los demas, por si se apaga uno corriendo, 
# este puede tomar su lugar y designar a uno nuevo para operar

@Singleton
class CommsManager(NodeClient):
    _logger = logging.getLogger(__name__)
    def __init__(self, network_factory=None):
        self.cc = ConfigurationContainer.instance()
        self.root = self.cc.settings['general']['distribution']['root']
        self.local = MPI.COMM_WORLD
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.sys_info = {
            "node" : TestData.nodes[self.rank],
            "rank" : self.rank
        }

        # Network factory
        if not network_factory:
            dataset_name = self.cc.settings['dataloader']['dataset_name']
            dataloader = self.cc.create_instance(dataset_name)
            self.network_factory = self.cc.create_instance(\
                                        self.cc.settings['network']['name'],\
                                        dataloader.n_input_neurons)
        else: self.network_factory = network_factory
    # ===================================================
    #                    New functions
    # ===================================================
    def start_comms(self):
        data = None
        data = self.comm.gather(self.sys_info, root=self.root)
        self.nodes_info = data

    def isend(self, message, dest):
        r_dest = self._parse_node(dest)
        self.comm.send(message, dest=r_dest, tag=1)
        self._logger.info("Sent message to {} ,tag 1".format(r_dest))

    def send_task(self, task, dest, data=None):
        r_dest = self._parse_node(dest)
        if data:
            data["task"] = task
            self.comm.send(data, dest=r_dest, tag=3)
        else:
            self.comm.send({"task" : task}, dest=r_dest, tag=3)
        self._logger.info("Sent task ({}) to {}, tag 3".format(task, r_dest))

    def recv(self, source=None):
        if source:
            r_source = self._parse_node(source)
            data = self.comm.recv(source=r_source, tag=1)
            self._logger.info("Recieved data from {}, tag 1".format(r_source))
        else:
            status = MPI.Status()
            data = self.comm.recv(source=MPI.ANY_SOURCE, status=status, tag=1)
            data["source"] = status.Get_source()
            self._logger.info("Recieved data from {}, tag 1".format(data["source"]))
        return data

    def recv_task(self, source=None):
        if source:
            r_source = self._parse_node(source)
            data = self.comm.recv(source=r_source, tag=3)
            self._logger.info("Recieved task ({}) from {}, tag 3".format(data["task"],r_source))
        else:
            status = MPI.Status()
            data = self.comm.recv(source=MPI.ANY_SOURCE, status=status, tag=3)
            data["source"] = status.Get_source()
            self._logger.info("Recieved task ({}) from {}, tag 3".format(data["task"],data["source"]))
        return data

    
    def start_worker(self, pu, grid):
        r_pu = self._parse_node(pu)
        self.comm.send({"task" : "run", "config" : self.cc.settings}, dest=r_pu, tag=3)
        self._logger.info("Started worker {}, tag 3".format(r_pu))

    # def recieve_root(self):
    #     return self.comm.recv(source=self.root)

    def close_all(self):
        for i in range(self.size):
            self.comm.isend({"task": "close"}, dest=i, tag=3)

    def _parse_node(self, node):
        if isinstance(node, int):
            return node
        elif isinstance(node, float):
            return int(node)
        return int(node["id"])

    # ===================================================
    #                Comms management
    # ===================================================

    def new_comm(self, color, key):
        '''
        Create 2 comms:
            Local: Intended for PUs in grid
            Global: same as local plus master
        '''
        self.local = MPI.COMM_WORLD.Split(color, key)
        self.general = MPI.COMM_WORLD.Split(color, key+1)
        # self.local_rank = key
        size = self.local.Get_size()
        self._logger.info("New comm {}, rank {} out of {}".format(color, key, size))
    
    def new_comm_master(self, color, key):
        '''
        Same as new_comm but intended for master only
        '''
        self.local = MPI.COMM_WORLD.Split(color, key)
        self.general = MPI.COMM_WORLD.Split(0, 0)
        # self.local_rank = key
        size = self.local.Get_size()
        self._logger.info("New comm {}, rank {} out of {}".format(color, key, size))


    # ===================================================
    #                Comm operations
    # ===================================================
    @property
    def local_rank(self):
        '''
        Local rank getter for MPI comm
        '''
        return self.local.Get_rank()

    @property
    def general_rank(self):
        '''
        General rank getter for MPI comm
        '''
        return self.general.Get_rank()

    def local_all_gather(self, send_data):
        ret_data = self.local.allgather(send_data)
        self._logger.info("Allgather on local comm")
        return ret_data

    def general_gather(self, send_data):
        '''
        Gather operation to send all the results data to master.
        '''
        self.general.gather(send_data, root=0)
        self._logger.info("Gather on general comm")

    def general_gather_master(self):
        '''
        Same as general_gather but this operation removes first
        element from array and returns the result.
        '''
        ret_data = self.general.gather(None, root=0)
        self._logger.info("Gather on general comm by master")
        ret_data.pop(0)
        return ret_data

    def general_gather_results(self):
        res_data = self.general_gather_master()

        formatted_list = []
        for i, data in enumerate(res_data):
            formatted_list.append((i,
                        self._create_population(data['generators'],
                                            self.network_factory.create_generator,
                                            TYPE_GENERATOR),
                        self._create_population(data['discriminators'],
                                            self.network_factory.create_discriminator,
                                            TYPE_DISCRIMINATOR),
                        data['weights_generators'],
                        data['weights_discriminators'])),

        return formatted_list


    # ===================================================
    #         Redefined comunication functions
    # ===================================================

    def get_all_generators(self, nodes, timeout_sec=TIMEOUT_SEC_DEFAULT):
        """
        Concurrently loads all current generator individuals from the given nodes.
        Returns when all are loaded, or raises TimeoutError when timeout is reached.
        """
        generators = self.load_generators_from_api(nodes, timeout_sec)
        for a in generators:
            try:
                self._logger.info("Parsing gen: {}".format(a.keys()))
            except:
                self._logger.warn("Parsing gen: {}".format(a))
                
        return [self._parse_individual(gen, self.network_factory.create_generator)
                for gen in generators] #TODO: if self._is_json_valid(gen)

    def get_all_discriminators(self, nodes, timeout_sec=TIMEOUT_SEC_DEFAULT):
        """
        Concurrently loads all current discriminator individuals from the node specified by 'addresses'
        Returns when all are loaded, or raises TimeoutError when timeout is reached.
        """
        discriminators = self.load_discriminators_from_api(nodes, timeout_sec)
        for a in discriminators:
            try:
                self._logger.info("Parsing disc: {}".format(a.keys()))
            except:
                self._logger.warn("Parsing disc: {}".format(a))

        return [self._parse_individual(disc, self.network_factory.create_discriminator)
                for disc in discriminators] #TODO: if self._is_json_valid(disc)]

    def get_best_generators(self, nodes, timeout_sec=TIMEOUT_SEC_DEFAULT):
        generators = self.load_best_generators_from_api(nodes, timeout_sec)
        for a in generators:
            try:
                self._logger.info("Parsing b gen: {}".format(a.keys()))
            except:
                self._logger.warn("Parsing b gen: {}".format(a))

        return [self._parse_individual(gen, self.network_factory.create_generator)
                for gen in generators] #TODO: if self._is_json_valid(gen)]

    def load_best_generators_from_api(self, nodes, timeout_sec=TIMEOUT_SEC_DEFAULT):
        return self._load_parameters_concurrently(nodes, 'generators_best', timeout_sec)

    def load_generators_from_api(self, nodes, timeout_sec=TIMEOUT_SEC_DEFAULT):
        return self._load_parameters_concurrently(nodes, 'generators', timeout_sec)

    def load_best_discriminators_from_api(self, nodes, timeout_sec=TIMEOUT_SEC_DEFAULT):
        return self._load_parameters_concurrently(nodes, 'discriminators_best', timeout_sec)

    def load_discriminators_from_api(self, nodes, timeout_sec=TIMEOUT_SEC_DEFAULT):
        return self._load_parameters_concurrently(nodes, 'discriminators', timeout_sec)

    def _load_results(self, node, timeout_sec):
        try:
            self.send_task("results", node)
            self._logger.info("Waiting for results {}".format(node))
            return self.recv(node)
        except Exception as ex:
            self._logger.error('Error loading results from {}: {}.'.format(node, ex))
            return None

    def gather_results(self, nodes, timeout_sec=TIMEOUT_SEC_DEFAULT):
        """
        Gathers the final results from each client node, and therefore finishes their experiment runs.
        :return: A list of result tuples: [(node, generator_population, discriminator_population)]
        """

        results = []
        # with ThreadPoolExecutor(max_workers=MAX_HTTP_CLIENT_THREADS) as executor:

        #     futures = {executor.submit(self._load_results, node, timeout_sec): node for node in nodes}
        #     for future in as_completed(futures):
        #         # Result has the form { 'discriminators': [[],  [], ..], 'generators': [[], [], ..] }
        #         node = futures[future]
        #         result = future.result()
        #         if result is not None:
        #             results.append((node,
        #                             self._create_population(result['generators'],
        #                                                     self.network_factory.create_generator,
        #                                                     TYPE_GENERATOR),
        #                             self._create_population(result['discriminators'],
        #                                                     self.network_factory.create_discriminator,
        #                                                     TYPE_DISCRIMINATOR),
        #                             result['weights_generators'],
        #                             result['weights_discriminators'])),

        for node in nodes:
            result = self._load_results(node, timeout_sec)
            results.append((node,
                                    self._create_population(result['generators'],
                                                            self.network_factory.create_generator,
                                                            TYPE_GENERATOR),
                                    self._create_population(result['discriminators'],
                                                            self.network_factory.create_discriminator,
                                                            TYPE_DISCRIMINATOR),
                                    result['weights_generators'],
                                    result['weights_discriminators'])),

        return results

    def get_client_statuses(self, active_nodes):
        # Solo lo usa hearbeat
        statuses = []
        for client in active_nodes:
            try:
                self.send_task("status", client)
                resp = self.recv(client)
                resp['address'] = client
                resp['alive'] = True
                statuses.append(resp)
            except Exception:
                statuses.append({
                    'busy': None,
                    'finished': None,
                    'alive': False,
                    'address': client,
                    'port': client
                })

        return statuses

    def stop_running_experiments(self, except_for_clients=None):
        for i in range(self.size):
            self.comm.send_task("close", dest=i)

    def _load_parameters_async(self, node, task, timeout_sec):
        # try:
        start = time.time()

        self.send_task(task, node)
        resp = self.recv(node)
 
        stop = time.time()
        self._logger.info('Loading parameters from node {} took {} seconds'.format(node, stop - start))
        try:
            for n in resp:
                n['source'] = node
        except:
            self._logger.error("Error on resp {}".format(resp))
            raise
        return resp
        # except Exception as ex:
        #     self._logger.error('Error loading parameters from node {}: {}.'.format(node, ex))
        #     return []

    def _load_parameters_concurrently(self, nodes, path, timeout_sec):
        """
        Returns a list of parameter lists
        """

        all_parameters = []
        with ThreadPoolExecutor(max_workers=MAX_HTTP_CLIENT_THREADS) as executor:
            futures = [executor.submit(self._load_parameters_async, node, path, timeout_sec) for node in nodes]
            for future in as_completed(futures):
                all_parameters.extend(future.result())
        return all_parameters

    @staticmethod #
    def _parse_individual(data, create_genome):

        return Individual.decode(create_genome,
                                 data['parameters'],
                                 is_local=False,
                                 learning_rate=data['learning_rate'],
                                 optimizer_state=StateEncoder.decode(data['optimizer_state']),
                                 source=data['source'],
                                 id=data['id'],
                                 iteration=data.get('iteration', None))

    @staticmethod
    def _create_population(all_parameters, create_genome, population_type):
        individuals = [Individual.decode(create_genome, parameters['parameters'],
                                         source=parameters['source'])
                       for parameters in all_parameters if parameters and len(parameters) > 0]
        return Population(individuals, float('-inf'), population_type)


# ===================================================
#                Testing vars
# ===================================================
# TODO: Remove this in prod

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

