from node import treeNode
from treeMap import generate_random_expression as spawn
import numpy as np
from copy import deepcopy
import pdb

def spawn_offspring(individuals:int, max_depth:int, constant_range:tuple, x_values:np.ndarray ):
    '''
    Generates an offspring of 'individuals' number, which can reach a depth of 'max_depth'
    An individual is a treeMap referring to a numpy expression valid for the specified problem
    Params:
    individuals (int): number of individuals
    max_depth (int): max depth of the expression generated
    x_vals (np.ndarray): the problem x values
    Returns:
    The offspring
    '''

    offspring=[]
    constants=np.linspace(constant_range[0],constant_range[1],10) + 1e-9 #avoid generating null costants
    constants = constants.tolist()
    variables = [f"x{index}" for index in range(x_values.shape[0])]
    
    for _ in range(individuals):
        variables_dict=dict(zip(variables,[0] * len(variables)))
        expr = spawn(max_depth,deepcopy(variables_dict),constants)
        y_values = [expr.validate_and_evaluate(dict(zip(variables,x))) for x in x_values.T]
        while False in y_values:
            expr = spawn(max_depth,deepcopy(variables_dict),constants)
            y_values = [expr.validate_and_evaluate(dict(zip(variables,x))) for x in x_values.T]
        
        offspring.append(expr)
        #print(f'Number of indifviduals:{len(offspring)}')

    return offspring