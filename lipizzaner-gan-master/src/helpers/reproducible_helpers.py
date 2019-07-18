import random

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
    """
    # TODO Handle the case of integer overflow
    return seed + int(ip.replace('.', '')) + 1000*port
