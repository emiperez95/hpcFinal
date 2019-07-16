import numpy as np
from collections import OrderedDict

from helpers.singleton import Singleton
from helpers.configuration_container import ConfigurationContainer
from helpers.population import Population, TYPE_GENERATOR, TYPE_DISCRIMINATOR

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
        return self.neighbours + [self.local_node]

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
            self.mixture_weights_discriminators = self._init_mixture_weights()
        return self._mixture_weights_discriminators

    @mixture_weights_discriminators.setter
    def mixture_weights_discriminators(self, value):
        self._mixture_weights_discriminators = value


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
            neighs_list.append(self.grid[nei[0]%self.grid_x, nei[1]%self.grid_y])
        pu_pos = self.__pos_pu(processing_unit)
        return list(set(neighs_list) - set([pu_pos]))

    def __pos_pu(self, processing_unit):
        for i, row in enumerate(self.grid):
            for j, col in enumerate(row):
                if col == processing_unit:
                    return i, j
        return None, None

    def _set_source(self, population):
        for individual in population.individuals:
            individual.source = self.local_node["id"]
        return population

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
        return OrderedDict({n_id: default_weight for n_id in node_ids})