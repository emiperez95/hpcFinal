import numpy as np
from helpers.singleton import Singleton
from helpers.configuration_container import ConfigurationContainer

@Singleton
class GridManager():
    def __init__(self):
        self.cc = ConfigurationContainer.instance()
        self.grid_x = self.cc.settings["general"]["distribution"]["grid"]["x_size"]
        self.grid_y = self.cc.settings["general"]["distribution"]["grid"]["y_size"]
        self.grid = np.zeros((self.grid_x, self.grid_y))

    def empty_spaces(self):
        '''Return ammount of empty spaces in grid.'''
        return len(self.__empty_spaces())

    def assign_worker(self, worker):
        """
        Params:
            worker: Worker to be assigned to grid
        Result:
            If there is an empty space in the grid, the worker is assigned.
        """
        empty_spaces = self.__empty_spaces()
        if empty_spaces:
            w_place = empty_spaces.pop(0)
            self.grid[w_place // self.grid_y, w_place % self.grid_y] = worker

    def grid_to_list(self):
        '''Returns a list version of the grid.'''
        return self.grid.flatten()

    def get_neightbours(self, processing_unit):
        '''
        Params:
            processing_unit: Procesing unit id.
        Return:
            List of neighbours of processing_unit.
        '''
        x, y = self.__pos_pu(processing_unit)
        neights = [(x+1, y), (x, y+1), (x-1, y), (x, y-1)]
        neights_list = []
        for nei in neights:
            neights_list.append(self.grid[nei[0]%self.grid_x, nei[1]%self.grid_y])
        pu_pos = self.__pos_pu(processing_unit)
        return list(set(neights_list) - set([pu_pos]))

    def __empty_spaces(self):
        return [i for (i, elem) in enumerate(self.grid_to_list()) if elem == 0]

    def __str__(self):
        return self.grid.__str__()

    def __pos_pu(self, processing_unit):
        for i, row in enumerate(self.grid):
            for j, col in enumerate(row):
                if col == processing_unit:
                    return i, j
        return None, None
