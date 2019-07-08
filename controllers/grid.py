
import numpy as np

class Grid_manager():
    def __init__(self, cc):
        self.grid_x = cc["grid"]["x_size"]
        self.grid_y = cc["grid"]["y_size"]
        self.grid_len = cc["grid"]["len"]
        self.grid = np.zeros((self.grid_x, self.grid_y))

    def empty_spaces(self):
        return len(self.__empty_spaces())

    def assign_worker(self, worker):
        print(worker)
        empty_spaces = self.__empty_spaces()
        if len(empty_spaces) > 0:
            w_place = empty_spaces.pop(0)
            self.grid[w_place // self.grid_y, w_place % self.grid_y] = worker

    def __grid_to_list(self):
        return self.grid.flatten()

    def __empty_spaces(self):
        return [i for (i, elem) in enumerate(self.__grid_to_list()) if elem == 0]

    def __str__(self):
        return self.grid.__str__()
    
        

    
