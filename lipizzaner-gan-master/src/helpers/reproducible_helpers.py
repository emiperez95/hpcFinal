import random
import sys
import numpy as np
import torch
# from helpers.log_helper import logging

def set_random_seed(seed, cuda):
    
    # logging.getLogger(__name__).info("Setting seed {} -> {}".format(seed, seed%2**32-1))
    seed_modulo = seed%(2**32-1)
    random.seed(seed_modulo)
    np.random.seed(seed_modulo)
    torch.manual_seed(seed_modulo)

    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed_modulo)
        torch.cuda.manual_seed_all(seed_modulo)


def get_heuristic_seed(seed, ip, port):
    """
    Heuristic method to obtain seed number based on client IP and port
    (Since it is desired to have different seed for different clients to ensure diversity)

    Added check for MPI, and adds rank to help diversiy when running multiple
    procesing units on the same node.

    Solved this on set_random_seed by using modulo maxint:
        -Handle the case of integer overflow
    """

    modulename = 'mpi4py'
    heuristic_seed = seed + int(ip.replace('.', '')) + 1000*port

    if modulename in sys.modules:
        from mpi4py import MPI
        heuristic_seed *= MPI.COMM_WORLD.Get_rank()

    return heuristic_seed
       
