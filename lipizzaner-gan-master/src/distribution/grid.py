import numpy as np
from collections import OrderedDict

from helpers.singleton import Singleton
from helpers.configuration_container import ConfigurationContainer
from helpers.population import Population, TYPE_GENERATOR, TYPE_DISCRIMINATOR
from helpers.log_helper import logging
from helpers.class_dict_converter import individual_to_dict

from distribution.concurrent_populations import ConcurrentPopulations
from distribution.comms_manager import CommsManager
from distribution.neighbourhood import Neighbourhood

'''
    Class that provides the visualization of the actual grid a procesing unit
is on. It requires that a grid is loaded in the settings, otherwise it will
fail.
    The grid can be reloaded from the settings by using load_grid. This allows
to have changing neighbours or diferent structures without having to modify
anything else.
'''

@Singleton
class Grid():
    _logger = logging.getLogger(__name__)
    def __init__(self):
        self.cc = ConfigurationContainer.instance()
        self.grid_x = self.cc.settings["general"]["distribution"]["grid"]["x_size"]
        self.grid_y = self.cc.settings["general"]["distribution"]["grid"]["y_size"]
        self.grid_size = self.grid_x * self.grid_y

        # Modifications from Neighbourhood __init__
        self.concurrent_populations = ConcurrentPopulations.instance()
        self.node_client = CommsManager.instance()
        self.grid = np.array([])

        self._mixture_weights_generators = None
        self._mixture_weights_discriminators = None

    def __str__(self):
        return self.grid.__str__()

    @property
    def local_node(self):
        '''
        Return self rank as a dict,
        Used for compatibility.
        '''
        return {"id" : self.node_client.rank}

    @property
    def local_rank(self):
        return self.node_client.local_rank

    @property
    def grid_position(self):
        '''Return x and y positions in grid.'''
        lmb_func = lambda a: (a[0][0], a[1][0])
        return lmb_func(np.where(self.grid == self.local_node["id"]))

    @property
    def cell_number(self):
        '''Return position in flattened grid.'''
        grid_pos = self.grid_position
        return grid_pos[1] * self.grid_y + grid_pos[0]

    @property
    def neighbours(self):
        '''Return id dict of neighbours'''
        pu_id = self.local_node["id"]
        return [{"id" : pu_id} for pu_id in self.get_neighbours(pu_id)]

    @property
    def all_nodes(self):
        '''Return neighbours + self list.'''
        return self.neighbours + [{"id": self.local_rank}]

    @property
    def mixture_weights_generators(self):
        if self.grid.size != 0 and not self._mixture_weights_generators:
            self._mixture_weights_generators = self._init_mixture_weights()
        return self._mixture_weights_generators

    @mixture_weights_generators.setter
    def mixture_weights_generators(self, value):
        self._mixture_weights_generators = value

    @property
    def mixture_weights_discriminators(self):
        if self.grid.size != 0 and not self._mixture_weights_discriminators \
                and (self.cc.settings['trainer']['name'] == 'with_disc_mixture_wgan' \
                or self.cc.settings['trainer']['name'] == 'with_disc_mixture_gan'):
            self._mixture_weights_discriminators = self._init_mixture_weights()
        return self._mixture_weights_discriminators

    @mixture_weights_discriminators.setter
    def mixture_weights_discriminators(self, value):
        self._mixture_weights_discriminators = value

    def rank_to_wid(self, rank):
        lmb_func = lambda a: (a[0][0], a[1][0])
        x, y =  lmb_func(np.where(self.grid == rank))
        return x * self.grid_y + y

    # def wid_to_rank(self, rank):
    #     return self.grid[rank%self.grid_x, rank%self.grid_y]

    def load_grid(self):
        '''Loads grid from settings'''
        self.grid = self.cc.settings["general"]["distribution"]["grid"]["config"]

    def get_neighbours(self, processing_unit):
        '''
        Params:
            processing_unit: Procesing unit id.
        Return:
            List of neighbours of processing_unit.
        '''
        x, y = self.__pos_pu(processing_unit)
        neighs = [(x+1, y), (x, y+1), (x-1, y), (x, y-1)]
        neighs_list = []
        for nei in neighs:
            rank = self.grid[nei[0]%self.grid_x, nei[1]%self.grid_y]
            neighs_list.append(self.rank_to_wid(rank))
        pu_pos = self.local_rank
        return list(set(neighs_list) - set([pu_pos]))

    def __pos_pu(self, processing_unit):
        for i, row in enumerate(self.grid):
            for j, col in enumerate(row):
                if col == processing_unit:
                    return i, j
        return None, None

    def _set_source(self, population):
        for individual in population.individuals:
            individual.source = self.local_rank
        return population


    # ==========================================================
    #                 Local neighbourhood opers
    # ==========================================================    

    def all_disc_gen_local(self, local_gen, local_disc):
        '''
            Get all generator and discriminators from all neighbours
        and create the Populations with them.
        Return:
            gen_pop: Population of all generators from neighbours
            disc_pop: Population of all discriminators from neighbours
        '''
        # TODO: Check if encoding is necesary or if pickle is enough
        send_data = (local_gen, local_disc)
        data = self.node_client.local_all_gather(send_data)

        generators = []
        discriminators = []
        # lamda_separator = lambda d: data[0], data[1]
        for sender_wid, elem in enumerate(data):
            if sender_wid in self.get_neighbours(self.node_client.rank):
                for gen_indiv in elem[0].individuals:
                    gen_indiv.source = sender_wid

                for disc_indiv in elem[1].individuals:
                    disc_indiv.source = sender_wid
                generators += elem[0].individuals
                discriminators += elem[1].individuals
            elif sender_wid == self.local_rank:
                for gen_indiv in local_gen.individuals:
                    gen_indiv.source = sender_wid
                generators += local_gen.individuals

                for disc_indiv in local_disc.individuals:
                    disc_indiv.source = sender_wid
                discriminators += local_disc.individuals

            
        gen_pop = Population(individuals=generators,
                          default_fitness=local_gen.default_fitness,
                          population_type=TYPE_GENERATOR)
        disc_pop = Population(individuals=discriminators,
                          default_fitness=local_disc.default_fitness,
                          population_type=TYPE_DISCRIMINATOR)
        return gen_pop, disc_pop


    def best_generators_local(self):
        local_population = self.local_generators
        best_local_individual = sorted(local_population.individuals, key=lambda x: x.fitness)[0]

        all_best = self.node_client.local_all_gather(best_local_individual)
        for sender_wid, indiv in enumerate(all_best):
            indiv.source = sender_wid

        return Population(individuals=all_best,
                          default_fitness=local_population.default_fitness,
                          population_type=TYPE_GENERATOR)

    def get_all_parameters_local(self):
        local_population_gen = self.local_generators
        best_individual_gen = sorted(local_population_gen.individuals, key=lambda x: x.fitness)[0]
        best_parsed_gen = individual_to_dict(best_individual_gen)
        best_parsed_gen["source"] = {"id": self.node_client.local_rank}

        local_population_disc = self.local_discriminators
        best_individual_disc = sorted(local_population_disc.individuals, key=lambda x: x.fitness)[0]
        best_parsed_disc = individual_to_dict(best_individual_disc)
        best_parsed_disc["source"] = {"id": self.node_client.local_rank}

        send_data = (best_parsed_gen, best_parsed_disc)
        data_arr = self.node_client.local_all_gather(send_data)

        results = {
            'generators': [],
            'discriminators': [],
        }
        neighs = self.get_neighbours(self.local_node["id"]) + [self.local_rank]
        for i, touple in enumerate(data_arr):
            if i in neighs:
                results['generators'].append(touple[0])
                results['discriminators'].append(touple[1])
        
        return results

        # Missing data postprocesing and return


    # def best_discriminators_local(self):
    #     local_population = self.local_discriminators
    #     best_local_individual = sorted(local_population.individuals, key=lambda x: x.fitness)[0]

    #     all_best = self.node_client.local_all_gather(best_local_individual)
    #     for sender_wid, indiv in enumerate(all_best):
    #         indiv.source = sender_wid

    #     return Population(individuals=all_best,
    #                       default_fitness=local_population.default_fitness,
    #                       population_type=TYPE_DISCRIMINATOR)

    # ==========================================================
    #                   From neighbourhood
    # ==========================================================
    @property
    def local_generators(self):
        # Return local individuals for now, possibility to split up gens and discs later
        return self._set_source(self.concurrent_populations.generator)

    @property
    def local_discriminators(self):
        # Return local individuals for now, possibility to split up gens and discs later
        return self._set_source(self.concurrent_populations.discriminator)

    @property
    def all_generators(self):
        neighbour_individuals = self.node_client.get_all_generators(self.neighbours)
        local_population = self.local_generators

        return Population(individuals=neighbour_individuals + local_population.individuals,
                          default_fitness=local_population.default_fitness,
                          population_type=TYPE_GENERATOR)

    @property
    def best_generators(self):
        best_neighbour_individuals = self.node_client.get_best_generators(self.neighbours)
        local_population = self.local_generators
        best_local_individual = sorted(local_population.individuals, key=lambda x: x.fitness)[0]

        return Population(individuals=best_neighbour_individuals + [best_local_individual],
                          default_fitness=local_population.default_fitness,
                          population_type=TYPE_GENERATOR)

    @property
    def all_discriminators(self):
        neighbour_individuals = self.node_client.get_all_discriminators(self.neighbours)
        local_population = self.local_discriminators

        return Population(individuals=neighbour_individuals + local_population.individuals,
                          default_fitness=local_population.default_fitness,
                          population_type=TYPE_DISCRIMINATOR)

    @property
    def all_generator_parameters(self):
        neighbour_generators = self.node_client.load_generators_from_api(self.neighbours)
        local_parameters = [i.genome.encoded_parameters for i in self.local_generators.individuals]
        return local_parameters + [n['parameters'] for n in neighbour_generators]

    @property
    def all_discriminator_parameters(self):
        neighbour_discriminators = self.node_client.load_discriminators_from_api(self.neighbours)
        local_parameters = [i.genome.encoded_parameters for i in self.local_discriminators.individuals]
        return local_parameters + [n['parameters'] for n in neighbour_discriminators]

    @property
    def best_generator_parameters(self):
        return self.node_client.load_best_generators_from_api(self.neighbours + [self.local_node])

    @property
    def best_discriminator_parameters(self):
        return self.node_client.load_best_discriminators_from_api(self.neighbours + [self.local_node])

    def _init_mixture_weights(self):
        node_ids = [node['id'] for node in self.all_nodes]
        default_weight = 1 / len(node_ids)
        # Warning: Feature of order preservation in Dict is used in the mixture_weight
        #          initialized here because further code involves converting it to list
        # According to https://stackoverflow.com/a/39980548, it's still preferable/safer
        # to use OrderedDict over Dict in Python 3.6
        ordered_dict = OrderedDict({n_id: default_weight for n_id in node_ids})
        return ordered_dict