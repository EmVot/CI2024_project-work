import numpy as np
from evolutionary import spawn_offspring
import random

datapath='./data/'


if __name__ =='__main__':

    data = np.load(datapath+'problem_2.npz')

    individuals_nr=100
    x_values = data['x']
    y_values = data['y']
    const_range=(-5,5)
    max_depth = 4


    offspring=spawn_offspring(np.int32(individuals_nr),max_depth,const_range,x_values)

    for expr in offspring:
        print(expr)

