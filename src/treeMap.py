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
    
    def fitness(self,problem) -> np.float64:

        def mse(y_true, y_pred) -> np.float64:
            return np.mean((y_true - y_pred) ** 2,dtype=np.float64)

        x_values = problem['x']
        y_values = problem['y']
        y_pred = self.validate_and_evaluate(x_values)

        return mse(y_values,y_pred)

    def __str__(self):
        return self.root.__str__()
    
    def getDepth(self):

        def deep_traversing(node:treeNode,depth):
            if node is None:
                return 0
            else:
                return 1 + max(deep_traversing(node.left_child,depth),deep_traversing(node.right_child,depth))
        
        return deep_traversing(self.root,0)

def generate_random_expression(depth, variables, constants, variable_coefficient_range_dict):


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
            
            probabilities = [1 / (f**2 + 1) for f in frequencies]
            
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
            p = 1.8*(len(unchosen_variables) / len(variables))
        
        if len(unchosen_variables) <= 2:
            p=1
        
        elif len(unchosen_variables)==0:
            # if there are no left variables the probability of choosing a constant is fixed
            p = 0.5

        # we choose a random constant with p, a variable otherwise
        if random.random() > p:
            return random.choice(constants)
        else:
            return choose_variable(variables)
            
    # leaves must be constants or variables
    if depth == 0 or random.random() > 0.85:
        leaf=select_leaf_function(variables,constants)
        if leaf in variables:
            variable_coefficient_range=variable_coefficient_range_dict[leaf]
            return treeNode(leaf,coefficient=random.choice(np.linspace(variable_coefficient_range[0],variable_coefficient_range[1],25)))
        else:
            return treeNode(leaf)
        
        # all others can be whatever

    if random.random() < 0.5:
        # Choose a random unary operator
        operation = random.choice(list(UNARY_OPERATORS.keys()))
        right_child = generate_random_expression(depth - 1, variables, constants,variable_coefficient_range_dict)
        return treeNode(operation, right_child=right_child)
    else:
        # Choose a random binary operator
        operation = random.choice(list(BINARY_OPERATORS.keys()))
        
        #randomize the descent direction to avoid having all the variables at the left of the tree
        if random.random() < 0.5:
            left_child = generate_random_expression(depth - 1, variables, constants,variable_coefficient_range_dict)
            right_child = generate_random_expression(depth - 1, variables, constants,variable_coefficient_range_dict)
        else:
            right_child = generate_random_expression(depth - 1, variables, constants,variable_coefficient_range_dict)
            left_child = generate_random_expression(depth - 1, variables, constants,variable_coefficient_range_dict)

        return treeNode(operation, left_child=left_child, right_child=right_child)