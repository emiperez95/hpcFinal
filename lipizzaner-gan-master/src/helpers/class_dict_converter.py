'''
    This module provides helper functions to convert different
classes into dicts for easy transformations.
'''
from distribution.state_encoder import StateEncoder

def individual_to_dict(individual):
    individual_dict = {
        'id': individual.id,
        'parameters': individual.genome.encoded_parameters,
        'learning_rate': individual.learning_rate,
        'optimizer_state': StateEncoder.encode(individual.optimizer_state)
    }
    if individual.iteration is not None:
        individual_dict['iteration'] = individual.iteration

    return individual_dict
