from node import treeNode
import random
from operators import UNARY_OPERATORS, BINARY_OPERATORS
import pdb


def generate_random_expression(depth, variables, constants):
    '''
    Recursive generation of random expression for a fixed depth, ensuring all variables are used
    at least once in the tree.
    
    Params:
        depth: The current depth of the expression tree.
        variables: List of variables available (e.g. ['x', 'y', 'z']).
        constants: List of constants available.
        ensure_variable: If True, ensure that the expression will have at least one variable in the tree.
        variables_in_tree: List to keep track of variables used so far in the tree.
        
    Returns:
        treeNode: A node representing the root of the expression tree.
    '''

    def select_leaf_function(variables:dict, constants):
        '''
        This function picks a leaf based on probability, dependent to the number of the unpicked variables and the times a variable has been chosen
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