import numpy as np
from evolutionary import evolutionary_algorithm
datapath='./data/'


if __name__ =='__main__':

    problem_epressions={}
    
    for problemNr in range(1,9):

        dataname = 'problem_'+str(problemNr)+'.npz'
        data = np.load(datapath+dataname)
        print(dataname)
        individuals_nr=128
        y_val=data['y']
        y_range = np.mean(y_val) + np.std(y_val)
        x=data['x']
        const_range=(-y_range, y_range)
        max_depth = 4
        generation_nr = 100
        tau = 8

        best_ind=evolutionary_algorithm(individuals_nr,individuals_nr/2,generation_nr,tau,max_depth,data,const_range)
        
        problem_epressions[problemNr]=best_ind

    