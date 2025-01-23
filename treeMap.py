import numpy as np
from node import treeNode
import random
from operators import UNARY_OPERATORS, BINARY_OPERATORS
import pdb


class treeMap:

    def __init__(self,max_depth:int,varibales:list,constants:list,root:treeNode=None):

        self.max_depth=max_depth
        self.variables=varibales
        if root:
            self.root=root
        else:
            self.root=generate_random_expression(self.max_depth,self.variables,constants)
        
        self.fit = None
    
    def getRandomNode(self):

        nodeList=[]
        return random.choice(self.root.getNodes(nodeList))
    

    def validate_and_evaluate(self,x_values:np.ndarray):
        '''
        The expression is valid if both its syntax and the function domains are respected
        Params:
        x_values (np.ndarray): The problem samples data
        Returns:
        The evaluation (np.float64) if all is correct, False instead
        '''
        variables = [f"x{index}" for index in range(x_values.shape[0])]

        # data shape: (variablesNr, variableData)
        # => each measurement correspond to a single row of data.T, which shape is (variableData,variablesNr)

        y_estimate = [self.root.validate_and_evaluate(dict(zip(variables,x))) for x in x_values.T]

        if False in y_estimate:
            return False
        
        return y_estimate
    
    def fitness(self,problem):

        def mse(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        x_values = problem['x']
        y_values = problem['y']
        y_pred = self.validate_and_evaluate(x_values)

        return mse(y_values,y_pred)


def generate_random_expression(depth, variables, constants):


    def select_leaf_function(variables:dict, constants):
        '''
            This function picks a leaf based on probability, dependent on the number of 
            unpicked variables and the times a variable has been chosen.

            Parameters:
            - variables (dict): A dictionary where keys are variable names and values 
            represent the number of times each variable has been chosen.
            - constants (list): A list of possible constant values.

            Functionality:
            1. Choosing a Variable:
                - Variables that have been chosen less frequently have a higher probability of being picked.
                - The selection is done using probability weights that are inversely proportional 
                to the frequency of previous selections.
                - The function normalizes these probabilities and selects a variable accordingly.

            2. Choosing Between a Constant and a Variable:
                - If there are still unchosen variables, the probability of selecting a constant 
                is inversely proportional to the number of unchosen variables.
                - If all variables have been chosen at least once, the probability of selecting a 
                constant is fixed at 50%.
                - A random choice is made accordingly between constants and variables.

            Returns:
            - A selected leaf node, which could be either a variable (from variables) or a constant (from constants).

            Example Usage:
            variables = {"x": 2, "y": 0, "z": 3}
            constants = [1, 2, 3]
            selected_leaf = select_leaf_function(variables, constants)
            print(selected_leaf)  # Might print "y" if it has not been chosen yet, or a constant

            Related Functionality:
            - Used within a tree generation process where leaves can either be variables or constants.
            - Supports random expression generation with a balanced selection strategy.
        '''


        def choose_variable(variables:dict):

            # Probability to choose an unchosen variable antiproportional to its  choosing frequence
            frequencies = [v for v in variables.values()]
            
            probabilities = [1 / (f + 1) for f in frequencies]
            
            # Probabilities normalization
            p_sum=sum(probabilities)
            norm_probs = [p / p_sum for p in probabilities]
            
            # choice is made considering norm_probabilities
            chosen_variable = random.choices(list(variables.keys()), norm_probs, k=1)[0]
            variables[chosen_variable]+=1

            return chosen_variable

        unchosen_variables = [k for k, v in variables.items() if v == 0]
        
        if unchosen_variables:
            # probability of choosing a constant are antiproportional to the number of unchosen variables
            p = len(unchosen_variables) / len(variables)
        
        else:
            # if there are no left variables the probability of choosing a constant is fixed
            p = 0.5

        # we choose a random constant with p, a variable otherwise
        if random.random() > p:
            return random.choice(constants)
        else:
            return choose_variable(variables)
            

    
    # leaves must be constants or variables
    if depth == 0:
        return treeNode(select_leaf_function(variables,constants))
    
    # all others can be whatever
    else:
        p = random.random() < 0.9 #probability of choosing an operator (preferred)
        
        if p:
            if random.random() < 0.5:
                # Choose a random unary operator
                operation = random.choice(list(UNARY_OPERATORS.keys()))
                right_child = generate_random_expression(depth - 1, variables, constants)
                return treeNode(operation, right_child=right_child)
            else:
                # Choose a random binary operator
                operation = random.choice(list(BINARY_OPERATORS.keys()))
                
                #randomize the descent direction to avoid having all the variables at the left of the tree
                if random.random() < 0.5:
                    left_child = generate_random_expression(depth - 1, variables, constants)
                    right_child = generate_random_expression(depth - 1, variables, constants)
                else:
                    left_child = generate_random_expression(depth - 1, variables, constants)
                    right_child = generate_random_expression(depth - 1, variables, constants)

                return treeNode(operation, left_child=left_child, right_child=right_child)
        
        else:   #we put a leaf even if we do not reach the maximum depth
            return treeNode(select_leaf_function(variables,constants))