
import numpy as np
from helpers.singleton import Singleton
from helpers.configuration_container import ConfigurationContainer

@Singleton
class GridManager():
    def __init__(self):
        cc = ConfigurationContainer.instance()
        self.grid_x = cc.settings["general"]["distribution"]["grid"]["x_size"]
        self.grid_y = cc.settings["general"]["distribution"]["grid"]["y_size"]

        self.grid = np.zeros((self.grid_x, self.grid_y))

    def empty_spaces(self):
        return len(self.__empty_spaces())

    def assign_worker(self, worker):
        empty_spaces = self.__empty_spaces()
        if len(empty_spaces) > 0:
            w_place = empty_spaces.pop(0)
            self.grid[w_place // self.grid_y, w_place % self.grid_y] = worker

    def grid_to_list(self):
        return self.grid.flatten()

    def __empty_spaces(self):
        return [i for (i, elem) in enumerate(self.grid_to_list()) if elem == 0]

    def __str__(self):
        return self.grid.__str__()

    def get_neightbours(self, pu):
        x, y = self.__pos_pu(pu)
        neights = [(x+1, y), (x, y+1), (x-1, y), (x, y-1)]
        neights_list = []
        for nei in neights:
            neights_list.append(self.grid[nei[0]%self.grid_x, nei[1]%self.grid_y])
        pu_pos = self.__pos_pu(pu)
        return list(set(neights_list) - set([pu_pos]))
        
    def __pos_pu(self, pu):
        for i, row in enumerate(self.grid):
            for j, col in enumerate(row):
                if col == pu:
                    return i, j
    
    def load_grid(self, grid):
        self.grid = grid
        

    
